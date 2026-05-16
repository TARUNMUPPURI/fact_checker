"""
scripture_gpt/api/main.py
--------------------------
FastAPI application for Scripture GPT V1.

Endpoints:
    POST /query          — general query (explore or factcheck)
    POST /factcheck      — forced factcheck mode
    GET  /entity/{id}    — character metadata
    GET  /health         — liveness check
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
_ALIASES_PATH   = os.getenv("ALIASES_PATH",   str(_REPO_ROOT / "scripture_gpt" / "data" / "character_aliases.json"))
_RAW_SARGAS_DIR = os.getenv("RAW_SARGAS_DIR", str(_REPO_ROOT / "scripture_gpt" / "raw_sargas"))
_CHROMA_DIR     = os.getenv("CHROMA_PERSIST_DIR", str(_REPO_ROOT / "scripture_gpt" / ".chroma"))
_NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
_NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ---------------------------------------------------------------------------
# Module-level pipeline handle (set in lifespan)
# ---------------------------------------------------------------------------
_pipeline = None
_aliases_data: dict = {}


# ---------------------------------------------------------------------------
# Lifespan — initialise once, never at module level
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _aliases_data

    # Load aliases for /entity endpoint
    try:
        aliases_path = Path(_ALIASES_PATH)
        if aliases_path.exists():
            _aliases_data = json.loads(aliases_path.read_text(encoding="utf-8"))
        else:
            log.warning("character_aliases.json not found at: %s", _ALIASES_PATH)
    except Exception as exc:
        log.error("Failed to load aliases: %s", exc)

    # Initialise QueryPipeline
    try:
        from scripture_gpt.pipeline.query_pipeline import QueryPipeline
        _pipeline = QueryPipeline(
            aliases_path=_ALIASES_PATH,
            raw_sargas_dir=_RAW_SARGAS_DIR,
            chroma_persist_dir=_CHROMA_DIR,
            neo4j_uri=_NEO4J_URI,
            neo4j_user=_NEO4J_USER,
            neo4j_password=_NEO4J_PASSWORD,
        )
        log.info("QueryPipeline initialised.")
    except Exception as exc:
        log.error("QueryPipeline init failed (pipeline disabled): %s", exc)
        _pipeline = None

    yield

    # Shutdown: close Neo4j driver if accessible
    if _pipeline is not None:
        try:
            from scripture_gpt.storage.neo4j_ingest import Neo4jIngestor
            # Neo4jIngestor is held inside pipeline; close via the module global
            import scripture_gpt.pipeline.query_pipeline as qp
            if qp._neo4j is not None:
                qp._neo4j.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Scripture GPT",
    description="Valmiki Ramayana fact-checker and knowledge explorer.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500,
                       description="Natural language question about the Ramayana")


class CitationModel(BaseModel):
    kanda:               str = ""
    chunk_id:            str = ""
    shloka_number:       Optional[int] = None
    sanskrit_devanagari: str = ""
    english:             str = ""
    detail:              str = ""


class QueryResponse(BaseModel):
    query:          str
    mode:           str
    complexity:     str
    answer:         str
    citations:      list[CitationModel] = []
    total_cost_usd: float
    latency_ms:     int


class FactcheckRequest(BaseModel):
    claim: str = Field(..., min_length=3, max_length=500,
                       description="Claim to verify against the Valmiki Ramayana")


class FactcheckResponse(BaseModel):
    claim:                str
    verdict:              str
    verdict_confidence:   float
    explanation:          str
    supporting_points:    list[str] = []
    contradicting_points: list[str] = []
    citations:            list[CitationModel] = []
    total_cost_usd:       float
    latency_ms:           int


class EntityResponse(BaseModel):
    canonical_id:     str
    canonical_name:   str
    epithets:         list[str] = []
    primary_kandas:   list[str] = []
    key_events:       list[str] = []
    appears_in_sargas: int = 0


class HealthResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round4(v: float) -> float:
    return round(v, 4)


def _citations_from_result(raw: list[dict]) -> list[CitationModel]:
    out = []
    for c in (raw or []):
        out.append(CitationModel(
            kanda=c.get("kanda", ""),
            chunk_id=c.get("chunk_id", ""),
            shloka_number=c.get("shloka_number"),
            sanskrit_devanagari=c.get("sanskrit_devanagari", ""),
            english=c.get("english", ""),
            detail=c.get("detail", ""),
        ))
    return out


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    if _pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="Query pipeline is not available. Check server logs for initialisation errors.",
        )
    try:
        result = await _pipeline.run(request.query)
    except Exception as exc:
        log.error("/query error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {exc}",
        )

    return QueryResponse(
        query=result.query,
        mode=result.mode.lower() if result.mode else "explore",
        complexity=result.complexity or "COMPLEX",
        answer=result.answer,
        citations=_citations_from_result(result.citations),
        total_cost_usd=_round4(result.total_cost_usd),
        latency_ms=result.latency_ms,
    )


# ---------------------------------------------------------------------------
# POST /factcheck
# ---------------------------------------------------------------------------

@app.post("/factcheck", response_model=FactcheckResponse, tags=["Query"])
async def factcheck_endpoint(request: FactcheckRequest) -> FactcheckResponse:
    if _pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="Query pipeline is not available. Check server logs for initialisation errors.",
        )
    # Prefix forces classify_intent to treat this as FACTCHECK
    forced_query = f"FACTCHECK: {request.claim}"
    try:
        result = await _pipeline.run(forced_query)
    except Exception as exc:
        log.error("/factcheck error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Fact-check processing failed: {exc}",
        )

    return FactcheckResponse(
        claim=request.claim,
        verdict=result.verdict or "INSUFFICIENT_EVIDENCE",
        verdict_confidence=_round4(result.verdict_confidence),
        explanation=result.verdict_explanation,
        supporting_points=result.evidence_by_kanda.get("supporting_points", [])
                          if isinstance(result.evidence_by_kanda, dict) else [],
        contradicting_points=result.evidence_by_kanda.get("contradicting_points", [])
                             if isinstance(result.evidence_by_kanda, dict) else [],
        citations=_citations_from_result(result.citations),
        total_cost_usd=_round4(result.total_cost_usd),
        latency_ms=result.latency_ms,
    )


# ---------------------------------------------------------------------------
# GET /entity/{canonical_id}
# ---------------------------------------------------------------------------

@app.get("/entity/{canonical_id}", response_model=EntityResponse, tags=["Knowledge"])
async def entity_endpoint(canonical_id: str) -> EntityResponse:
    characters = _aliases_data.get("characters", {})
    char = characters.get(canonical_id)
    if char is None:
        raise HTTPException(
            status_code=404,
            detail=f"Character '{canonical_id}' not found in the dictionary.",
        )

    # Count Neo4j sarga appearances (optional — 0 if Neo4j unavailable)
    appears_count = 0
    try:
        import scripture_gpt.pipeline.query_pipeline as qp
        if qp._neo4j is not None:
            rows = qp._neo4j.get_character_sargas(canonical_id)
            appears_count = len(rows)
    except Exception:
        pass

    return EntityResponse(
        canonical_id=canonical_id,
        canonical_name=char.get("canonical_name", canonical_id),
        epithets=char.get("epithets", []),
        primary_kandas=char.get("primary_kandas", []),
        key_events=char.get("key_events", []),
        appears_in_sargas=appears_count,
    )
