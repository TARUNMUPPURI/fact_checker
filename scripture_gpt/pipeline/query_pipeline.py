"""
query_pipeline.py
-----------------
LangGraph state machine for the Scripture GPT query pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Optional, TypedDict

import operator

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import anthropic
from langgraph.graph import StateGraph, END

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Models & costs
# ---------------------------------------------------------------------------
HAIKU  = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"

COST = {
    HAIKU:  {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000},
    SONNET: {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
}

VALID_VERDICTS = {"TRUE", "FALSE", "PARTIALLY_TRUE", "INSUFFICIENT_EVIDENCE"}
VALID_KANDA_KEYS = {
    "bala_kanda", "ayodhya_kanda", "aranya_kanda",
    "kishkindha_kanda", "sundara_kanda", "yuddha_kanda", "uttara_kanda",
}

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class QueryState(TypedDict):
    query:                str
    mode:                 str
    complexity:           str
    entities:             list[str]
    relevant_kandas:      list[str]
    claim:                str
    focus:                str
    tool_calls_made:      Annotated[list[dict], operator.add]
    tool_results:         Annotated[list[dict], operator.add]
    tool_rounds:          int
    evidence_by_kanda:    dict
    verdict:              str
    verdict_confidence:   float
    explanation:          str
    supporting_points:    list[str]
    contradicting_points: list[str]
    narrative:            str
    citations:            list[dict]
    messages:             Annotated[list[dict], operator.add]
    total_tokens:         int
    total_cost_usd:       float
    error:                str

# ---------------------------------------------------------------------------
# QueryResult
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    query:              str = ""
    mode:               str = ""
    complexity:         str = ""
    entities:           list[str] = field(default_factory=list)
    verdict:            str = ""
    verdict_confidence: float = 0.0
    verdict_explanation: str = ""
    narrative:          str = ""
    evidence_by_kanda:  dict = field(default_factory=dict)
    citations:          list[dict] = field(default_factory=list)
    answer:             str = ""
    total_tokens:       int = 0
    total_cost_usd:     float = 0.0
    latency_ms:         int = 0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_client = anthropic.Anthropic()

def _strip_fences(raw: str) -> str:
    s = re.sub(r"^```json\s*", "", raw.strip(), flags=re.IGNORECASE)
    return re.sub(r"```\s*$", "", s.strip())

def _parse_json(raw: str) -> Optional[dict]:
    try:
        return json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None

def _llm_cost(model: str, inp: int, out: int) -> float:
    c = COST.get(model, COST[HAIKU])
    return inp * c["input"] + out * c["output"]

def _call(model: str, messages: list, tools: list | None = None,
          max_tokens: int = 1024) -> anthropic.types.Message:
    kwargs = dict(model=model, max_tokens=max_tokens, messages=messages)
    if tools:
        kwargs["tools"] = tools
    return _client.messages.create(**kwargs)

# ---------------------------------------------------------------------------
# Module-level storage handles (set by QueryPipeline.__init__)
# ---------------------------------------------------------------------------
_chroma: Optional[object] = None
_neo4j:  Optional[object] = None
_aliases_data: dict = {}
_valid_char_ids: list[str] = []
_raw_sargas_dir: Path = Path("scripture_gpt/raw_sargas")

# ---------------------------------------------------------------------------
# Tool definitions (plain dicts for Anthropic SDK)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "vector_search",
        "description": (
            "Search scripture passages by semantic meaning. "
            "Use for conceptual queries about devotion, emotions, themes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query":            {"type": "string"},
                "kanda_filter":     {"type": "string"},
                "character_filter": {"type": "string"},
                "top_k":            {"type": "integer", "default": 8},
            },
            "required": ["query"],
        },
    },
    {
        "name": "graph_traverse",
        "description": (
            "Get all appearances of a character across the Ramayana in story order. "
            "Use when asked about a specific character."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "canonical_id": {"type": "string"},
                "kanda":        {"type": "string"},
            },
            "required": ["canonical_id"],
        },
    },
    {
        "name": "shloka_fetch",
        "description": (
            "Fetch the exact Sanskrit and English text of a specific sarga. "
            "Use to get citation text before quoting it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chunk_id":      {"type": "string"},
                "shloka_number": {"type": "integer"},
            },
            "required": ["chunk_id"],
        },
    },
    {
        "name": "entity_lookup",
        "description": (
            "Resolve any name or epithet to a canonical character ID. "
            "Use when you encounter an unfamiliar name like Anjaneya or Lankesh."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _exec_tool(name: str, inputs: dict) -> dict:
    try:
        if name == "vector_search":
            if _chroma is None:
                return {"error": "ChromaDB not initialised"}
            results = _chroma.search(
                query_text=inputs["query"],
                character_filter=inputs.get("character_filter"),
                kanda_filter=inputs.get("kanda_filter"),
                top_k=inputs.get("top_k", 8),
            )
            return {"results": results}

        if name == "graph_traverse":
            if _neo4j is None:
                return {"error": "Neo4j not initialised"}
            rows = _neo4j.get_character_sargas(inputs["canonical_id"])
            if inputs.get("kanda"):
                rows = [r for r in rows if r.get("kanda") == inputs["kanda"]]
            return {"sargas": rows}

        if name == "shloka_fetch":
            path = _raw_sargas_dir / f"{inputs['chunk_id']}.json"
            if not path.exists():
                return {"error": f"Sarga not found: {inputs['chunk_id']}"}
            data = json.loads(path.read_text(encoding="utf-8"))
            sn = inputs.get("shloka_number")
            if sn is not None:
                for s in data.get("shlokas", []):
                    if s.get("shloka_number") == sn:
                        return {"shloka": s, "chunk_id": inputs["chunk_id"]}
                return {"error": f"Shloka {sn} not found"}
            return {
                "chunk_id": inputs["chunk_id"],
                "text": data.get("text", "")[:2000],
                "text_sanskrit_full": data.get("text_sanskrit_full", "")[:1000],
                "shlokas": data.get("shlokas", [])[:5],
            }

        if name == "entity_lookup":
            name_q = inputs["name"].lower().strip()
            for cid, char in _aliases_data.get("characters", {}).items():
                if cid == name_q:
                    return {"canonical_id": cid, "canonical_name": char["canonical_name"]}
                all_names = (
                    [a.lower() for a in char.get("aliases", [])]
                    + [e.lower() for e in char.get("epithets", [])]
                    + [d.lower() for d in char.get("descriptive_references", [])]
                )
                if name_q in all_names:
                    return {"canonical_id": cid, "canonical_name": char["canonical_name"]}
            return {"error": f"Could not resolve: {inputs['name']}"}

        return {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        return {"error": str(exc)}

# ---------------------------------------------------------------------------
# NODE 1: classify_intent
# ---------------------------------------------------------------------------

def classify_intent(state: QueryState) -> dict:
    try:
        prompt = f"""Classify this query about the Valmiki Ramayana.

Query: {state['query']}

Valid character IDs: {json.dumps(_valid_char_ids[:40])}
Valid kanda keys: {json.dumps(sorted(VALID_KANDA_KEYS))}

Return ONLY JSON:
{{
  "mode": "FACTCHECK or EXPLORE",
  "complexity": "SIMPLE, MEDIUM, or COMPLEX",
  "entities": ["canonical_id list"],
  "relevant_kandas": ["kanda_keys where entities actually appear"],
  "claim": "exact claim if FACTCHECK, else empty string",
  "focus": "core subject of the query in one sentence",
  "confidence": 0.9
}}
SIMPLE=single entity single lookup, MEDIUM=single kanda, COMPLEX=multi-kanda or factcheck."""
        msg = _call(HAIKU, [{"role": "user", "content": prompt}])
        parsed = _parse_json(msg.content[0].text) or {}
        inp, out = msg.usage.input_tokens, msg.usage.output_tokens
        return {
            "mode":             parsed.get("mode", "EXPLORE"),
            "complexity":       parsed.get("complexity", "COMPLEX"),
            "entities":         parsed.get("entities", []),
            "relevant_kandas":  [k for k in parsed.get("relevant_kandas", []) if k in VALID_KANDA_KEYS],
            "claim":            parsed.get("claim", ""),
            "focus":            parsed.get("focus", state["query"]),
            "total_tokens":     inp + out,
            "total_cost_usd":   _llm_cost(HAIKU, inp, out),
            "messages":         [{"role": "assistant", "content": msg.content[0].text}],
        }
    except Exception as exc:
        log.error("classify_intent error: %s", exc)
        return {"complexity": "COMPLEX", "mode": "EXPLORE",
                "focus": state["query"], "error": str(exc)}

# ---------------------------------------------------------------------------
# NODE 2: direct_lookup
# ---------------------------------------------------------------------------

def direct_lookup(state: QueryState) -> dict:
    try:
        if not _neo4j or not state.get("entities"):
            return {"narrative": "No entity found for direct lookup.", "citations": []}
        entity = state["entities"][0]
        rows = _neo4j.get_character_sargas(entity)
        if not rows:
            return {"narrative": f"No sarga appearances found for '{entity}'.", "citations": []}
        citations = [
            {"kanda": r["kanda"], "chunk_id": r["chunk_id"],
             "shloka_number": None,
             "sanskrit_devanagari": "", "english": r.get("text_summary", ""),
             "detail": r.get("role", "")}
            for r in rows[:8]
        ]
        narrative = (
            f"{entity} appears in {len(rows)} sargas across "
            f"{len({r['kanda'] for r in rows})} kanda(s)."
        )
        return {"narrative": narrative, "citations": citations}
    except Exception as exc:
        log.error("direct_lookup error: %s", exc)
        return {"error": str(exc), "narrative": "", "citations": []}

# ---------------------------------------------------------------------------
# NODE 3: call_tools
# ---------------------------------------------------------------------------

def call_tools(state: QueryState) -> dict:
    try:
        context_msg = (
            f"Query: {state['query']}\n"
            f"Focus: {state.get('focus', '')}\n"
            f"Claim: {state.get('claim', '')}\n"
            f"Mode: {state.get('mode', 'EXPLORE')}\n"
            f"Kandas of interest: {state.get('relevant_kandas', [])}\n\n"
            "Use the available tools to gather evidence. When you have enough, stop calling tools."
        )
        history = state.get("messages", []) or []
        messages = [{"role": "user", "content": context_msg}] + history

        msg = _call(HAIKU, messages, tools=TOOLS, max_tokens=2048)
        inp, out = msg.usage.input_tokens, msg.usage.output_tokens

        new_tool_calls: list[dict] = []
        new_tool_results: list[dict] = []

        for block in msg.content:
            if block.type == "tool_use":
                result = _exec_tool(block.name, block.input)
                new_tool_calls.append({"name": block.name, "input": block.input})
                new_tool_results.append({"tool": block.name, "result": result})

        msg_text = next(
            (b.text for b in msg.content if b.type == "text"), ""
        )

        return {
            "tool_calls_made":  new_tool_calls,
            "tool_results":     new_tool_results,
            "tool_rounds":      state.get("tool_rounds", 0) + (1 if new_tool_calls else 0),
            "total_tokens":     inp + out,
            "total_cost_usd":   _llm_cost(HAIKU, inp, out),
            "messages":         [{"role": "assistant", "content": msg_text or str(msg.content)}],
            "_last_stop_reason": msg.stop_reason,
        }
    except Exception as exc:
        log.error("call_tools error: %s", exc)
        return {"error": str(exc)}

# ---------------------------------------------------------------------------
# NODE 4: summarise_evidence (async, asyncio.gather)
# ---------------------------------------------------------------------------

async def _summarise_kanda(kanda: str, focus: str, claim: str,
                            tool_results: list[dict]) -> tuple[str, dict]:
    relevant = [
        r for r in tool_results
        if kanda in str(r.get("result", ""))
    ]
    context = json.dumps(relevant[:6], ensure_ascii=False)[:3000]
    prompt = (
        f"Summarise evidence from {kanda} relevant to: {focus}\n"
        f"Claim to verify (if any): {claim}\n\n"
        f"Evidence:\n{context}\n\n"
        "Return ONLY JSON:\n"
        '{"summary":"...","supporting_events":[],'
        '"contradicting_events":[],"relevant_sargas":[],"has_evidence":true}'
    )
    try:
        msg = _call(HAIKU, [{"role": "user", "content": prompt}])
        parsed = _parse_json(msg.content[0].text) or {"has_evidence": False, "summary": ""}
    except Exception as exc:
        parsed = {"has_evidence": False, "summary": f"Error: {exc}"}
    return kanda, parsed


async def _gather_summaries(state: QueryState) -> dict:
    kandas = state.get("relevant_kandas") or list(VALID_KANDA_KEYS)
    focus  = state.get("focus", state["query"])
    claim  = state.get("claim", "")
    tool_results = state.get("tool_results", [])

    tasks = [_summarise_kanda(k, focus, claim, tool_results) for k in kandas]
    pairs = await asyncio.gather(*tasks)
    evidence = {k: v for k, v in pairs if v.get("has_evidence")}
    return {"evidence_by_kanda": evidence}


def summarise_evidence(state: QueryState) -> dict:
    try:
        return asyncio.run(_gather_summaries(state))
    except Exception as exc:
        log.error("summarise_evidence error: %s", exc)
        return {"error": str(exc), "evidence_by_kanda": {}}

# ---------------------------------------------------------------------------
# NODE 5: reason
# ---------------------------------------------------------------------------

def reason(state: QueryState) -> dict:
    try:
        evidence_str = json.dumps(state.get("evidence_by_kanda", {}),
                                  ensure_ascii=False)[:5000]
        mode = state.get("mode", "EXPLORE")

        if mode == "FACTCHECK":
            schema = (
                '{"verdict":"TRUE|FALSE|PARTIALLY_TRUE|INSUFFICIENT_EVIDENCE",'
                '"verdict_confidence":0.85,"explanation":"...",'
                '"supporting_points":[],"contradicting_points":[],'
                '"citations":[{"kanda":"","chunk_id":"","shloka_number":null,'
                '"sanskrit_devanagari":"","english":"","detail":""}]}'
            )
            prompt = (
                f"Claim: {state.get('claim', state['query'])}\n\n"
                f"Evidence by kanda:\n{evidence_str}\n\n"
                f"Return ONLY JSON:\n{schema}"
            )
        else:
            schema = (
                '{"narrative":"...","key_moments":[],"character_arc":"...",'
                '"citations":[{"kanda":"","chunk_id":"","shloka_number":null,'
                '"sanskrit_devanagari":"","english":"","detail":""}]}'
            )
            prompt = (
                f"Query: {state['query']}\n\n"
                f"Evidence by kanda:\n{evidence_str}\n\n"
                f"Return ONLY JSON:\n{schema}"
            )

        msg = _call(SONNET, [{"role": "user", "content": prompt}], max_tokens=2048)
        parsed = _parse_json(msg.content[0].text) or {}
        inp, out = msg.usage.input_tokens, msg.usage.output_tokens

        verdict = parsed.get("verdict", "INSUFFICIENT_EVIDENCE")
        if verdict not in VALID_VERDICTS:
            verdict = "INSUFFICIENT_EVIDENCE"

        return {
            "verdict":             verdict,
            "verdict_confidence":  float(parsed.get("verdict_confidence", 0.0)),
            "explanation":         parsed.get("explanation", ""),
            "supporting_points":   parsed.get("supporting_points", []),
            "contradicting_points": parsed.get("contradicting_points", []),
            "narrative":           parsed.get("narrative", ""),
            "citations":           parsed.get("citations", []),
            "total_tokens":        inp + out,
            "total_cost_usd":      _llm_cost(SONNET, inp, out),
            "messages":            [{"role": "assistant", "content": msg.content[0].text}],
        }
    except Exception as exc:
        log.error("reason error: %s", exc)
        return {"error": str(exc), "verdict": "INSUFFICIENT_EVIDENCE",
                "verdict_confidence": 0.0}

# ---------------------------------------------------------------------------
# NODE 6: verify_citations
# ---------------------------------------------------------------------------

def verify_citations(state: QueryState) -> dict:
    citations = list(state.get("citations", []))
    if not citations:
        return {}
    try:
        verified: list[dict] = []
        removed = 0
        for cite in citations:
            chunk_id = cite.get("chunk_id", "")
            if not chunk_id:
                continue
            fetched = _exec_tool("shloka_fetch", {"chunk_id": chunk_id,
                                                   "shloka_number": cite.get("shloka_number")})
            if "error" in fetched:
                removed += 1
                continue
            # Quick verification: check the claimed english text appears in fetched text
            claim_en = cite.get("english", "").lower()[:60]
            fetched_text = str(fetched).lower()
            if claim_en and claim_en not in fetched_text:
                # Ask Haiku to verify
                try:
                    prompt = (
                        f"Does this passage support the claim?\n"
                        f"Claim: {cite.get('detail','')}\n"
                        f"Passage: {str(fetched)[:800]}\n"
                        "Reply ONLY: YES or NO"
                    )
                    msg = _call(HAIKU, [{"role": "user", "content": prompt}], max_tokens=8)
                    if "NO" in msg.content[0].text.upper():
                        removed += 1
                        continue
                except Exception:
                    pass
            verified.append(cite)

        conf = max(0.0, float(state.get("verdict_confidence", 0.0)) - 0.1 * removed)
        return {"citations": verified, "verdict_confidence": conf}
    except Exception as exc:
        log.error("verify_citations error: %s", exc)
        return {"error": str(exc)}

# ---------------------------------------------------------------------------
# NODE 7: format_output
# ---------------------------------------------------------------------------

def format_output(state: QueryState) -> dict:
    mode = state.get("mode", "EXPLORE")
    citations = state.get("citations", [])

    cite_lines = []
    for c in citations[:5]:
        cite_lines.append(
            f"  [{c.get('kanda','?')} / {c.get('chunk_id','?')}] "
            f"{c.get('english','')[:120]}"
        )

    if mode == "FACTCHECK":
        verdict = state.get("verdict", "INSUFFICIENT_EVIDENCE")
        conf    = state.get("verdict_confidence", 0.0)
        expl    = state.get("explanation", "")
        answer  = f"VERDICT: {verdict} (confidence={conf:.2f})\n\n{expl}"
        if cite_lines:
            answer += "\n\nCitations:\n" + "\n".join(cite_lines)
    else:
        narrative = state.get("narrative", state.get("explanation", ""))
        answer    = narrative
        if cite_lines:
            answer += "\n\nKey passages:\n" + "\n".join(cite_lines)

    return {
        "answer":         answer,
        "total_cost_usd": round(state.get("total_cost_usd", 0.0), 4),
    }

# ---------------------------------------------------------------------------
# Conditional edge routers
# ---------------------------------------------------------------------------

def route_by_complexity(state: QueryState) -> str:
    if state.get("complexity") == "SIMPLE":
        return "direct_lookup"
    return "call_tools"


def should_continue_tools(state: QueryState) -> str:
    if state.get("tool_rounds", 0) >= 3:
        return "summarise_evidence"
    last = (state.get("messages") or [{}])[-1]
    content = last.get("content", "")
    # Check if last message contained tool_use (stored as string representation)
    if "tool_use" in str(content):
        return "call_tools"
    # Also check tool_calls_made in this round
    return "summarise_evidence"

# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

_graph = StateGraph(QueryState)

_graph.add_node("classify_intent",    classify_intent)
_graph.add_node("direct_lookup",      direct_lookup)
_graph.add_node("call_tools",         call_tools)
_graph.add_node("summarise_evidence", summarise_evidence)
_graph.add_node("reason",             reason)
_graph.add_node("verify_citations",   verify_citations)
_graph.add_node("format_output",      format_output)

_graph.set_entry_point("classify_intent")

_graph.add_conditional_edges(
    "classify_intent",
    route_by_complexity,
    {"direct_lookup": "direct_lookup", "call_tools": "call_tools"},
)
_graph.add_conditional_edges(
    "call_tools",
    should_continue_tools,
    {"call_tools": "call_tools", "summarise_evidence": "summarise_evidence"},
)
_graph.add_edge("direct_lookup",      "format_output")
_graph.add_edge("summarise_evidence", "reason")
_graph.add_edge("reason",             "verify_citations")
_graph.add_edge("verify_citations",   "format_output")
_graph.add_edge("format_output",      END)

app = _graph.compile()

# ---------------------------------------------------------------------------
# QueryPipeline
# ---------------------------------------------------------------------------

class QueryPipeline:

    def __init__(
        self,
        aliases_path: str | Path,
        raw_sargas_dir: str | Path,
        api_key: Optional[str] = None,
        chroma_persist_dir: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        global _aliases_data, _valid_char_ids, _raw_sargas_dir, _chroma, _neo4j

        _raw_sargas_dir = Path(raw_sargas_dir)

        aliases_path = Path(aliases_path)
        _aliases_data = json.loads(aliases_path.read_text(encoding="utf-8"))
        _valid_char_ids = sorted(_aliases_data.get("characters", {}).keys())

        if api_key:
            global _client
            _client = anthropic.Anthropic(api_key=api_key)

        # ChromaDB (optional — graceful degradation)
        try:
            from scripture_gpt.storage.chroma_ingest import ChromaIngestor
            persist = chroma_persist_dir or os.getenv("CHROMA_PERSIST_DIR", ".chroma")
            _chroma = ChromaIngestor(persist_dir=persist)
        except Exception as exc:
            log.warning("ChromaDB unavailable: %s", exc)

        # Neo4j (optional — graceful degradation)
        try:
            from scripture_gpt.storage.neo4j_ingest import Neo4jIngestor
            _neo4j = Neo4jIngestor(
                uri=neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                user=neo4j_user or os.getenv("NEO4J_USER", "neo4j"),
                password=neo4j_password or os.getenv("NEO4J_PASSWORD", ""),
            )
        except Exception as exc:
            log.warning("Neo4j unavailable: %s", exc)

    async def run(self, query: str) -> QueryResult:
        t0 = time.time()
        initial: QueryState = {
            "query": query, "mode": "", "complexity": "",
            "entities": [], "relevant_kandas": [],
            "claim": "", "focus": "",
            "tool_calls_made": [], "tool_results": [],
            "tool_rounds": 0, "evidence_by_kanda": {},
            "verdict": "", "verdict_confidence": 0.0,
            "explanation": "", "supporting_points": [],
            "contradicting_points": [], "narrative": "",
            "citations": [], "messages": [],
            "total_tokens": 0, "total_cost_usd": 0.0, "error": "",
        }
        final = await app.ainvoke(initial)
        latency = int((time.time() - t0) * 1000)
        return QueryResult(
            query=query,
            mode=final.get("mode", ""),
            complexity=final.get("complexity", ""),
            entities=final.get("entities", []),
            verdict=final.get("verdict", ""),
            verdict_confidence=final.get("verdict_confidence", 0.0),
            verdict_explanation=final.get("explanation", ""),
            narrative=final.get("narrative", ""),
            evidence_by_kanda=final.get("evidence_by_kanda", {}),
            citations=final.get("citations", []),
            answer=final.get("answer", ""),
            total_tokens=final.get("total_tokens", 0),
            total_cost_usd=final.get("total_cost_usd", 0.0),
            latency_ms=latency,
        )

    def run_sync(self, query: str) -> QueryResult:
        return asyncio.run(self.run(query))
