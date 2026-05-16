"""
chunking_schema.py
------------------
Foundational data schema for Scripture GPT V1 (Valmiki Ramayana).
stdlib only — no external dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Kanda registry
# ---------------------------------------------------------------------------

KANDA_REGISTRY: dict[str, dict] = {
    "bala_kanda": {
        "number": 1,
        "english_name": "Book of Youth",
        "sarga_count": 77,
        "url_short": "bala",
        "contents_path": "utf8/baala/baala_contents.htm",
    },
    "ayodhya_kanda": {
        "number": 2,
        "english_name": "Book of Ayodhya",
        "sarga_count": 119,
        "url_short": "ayodhya",
        "contents_path": "utf8/ayodhya/ayodhya_contents.htm",
    },
    "aranya_kanda": {
        "number": 3,
        "english_name": "Book of the Forest",
        "sarga_count": 75,
        "url_short": "aranya",
        "contents_path": "utf8/aranya/aranya_contents.htm",
    },
    "kishkindha_kanda": {
        "number": 4,
        "english_name": "Book of Kishkindha",
        "sarga_count": 67,
        "url_short": "kishkindha",
        "contents_path": "utf8/kish/kishkindha_contents.htm",
    },
    "sundara_kanda": {
        "number": 5,
        "english_name": "Book of Beauty",
        "sarga_count": 68,
        "url_short": "sundara",
        "contents_path": "utf8/sundara/sundara_contents.htm",
    },
    "yuddha_kanda": {
        "number": 6,
        "english_name": "Book of War",
        "sarga_count": 128,
        "url_short": "yuddha",
        "contents_path": "utf8/yuddha/yuddha_contents.htm",
    },
    "uttara_kanda": {
        "number": 7,
        "english_name": "Book of the Aftermath",
        "sarga_count": 111,
        "url_short": "uttara",
        "contents_path": "utf8/uttara/uttara_contents.htm",
    },
}

# ---------------------------------------------------------------------------
# Controlled vocabulary constants
# ---------------------------------------------------------------------------

VALID_EVENT_TYPES = frozenset({
    "battle", "journey", "dialogue", "discovery", "ceremony",
    "death", "transformation", "revelation", "devotion", "other",
})

VALID_NARRATIVE_ARCS = frozenset({
    "opening", "rising_action", "climax", "falling_action", "resolution",
})


# ---------------------------------------------------------------------------
# ScriptureChunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScriptureChunk:
    # --- Identity ---
    chunk_id: str                          # e.g. "sundara_kanda_0014"
    kanda: str                             # e.g. "sundara_kanda"
    kanda_number: int                      # 1-7
    kanda_english: str                     # e.g. "Book of Beauty"
    sarga_number: int                      # within kanda
    sarga_number_global: int               # across all 645 sargas

    # --- Text content ---
    text: str = ""                         # English prose (joined paragraphs)
    text_sanskrit_full: str = ""           # Devanagari text (joined blocks)
    text_summary: str = ""                 # LLM-generated summary

    # --- Shloka range ---
    shloka_start: int = 1
    shloka_end: int = 1
    shloka_count: int = 0

    # --- Entities ---
    characters_present: List[str] = field(default_factory=list)
    characters_speaking: List[str] = field(default_factory=list)
    characters_primary: List[str] = field(default_factory=list)
    locations_mentioned: List[str] = field(default_factory=list)
    objects_mentioned: List[str] = field(default_factory=list)

    # --- Events & narrative ---
    event_ids: List[str] = field(default_factory=list)
    event_type: str = "other"              # see VALID_EVENT_TYPES
    narrative_arc: str = "rising_action"  # see VALID_NARRATIVE_ARCS
    emotional_tone: str = ""
    themes: List[str] = field(default_factory=list)

    # --- Navigation ---
    prev_sarga_id: Optional[str] = None
    next_sarga_id: Optional[str] = None
    related_sarga_ids: List[str] = field(default_factory=list)

    # --- Provenance ---
    translation_source: str = "valmikiramayan.net"
    is_interpolated: bool = False
    raw_source_url: str = ""
    embedding_model: str = ""
    chunk_version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helper: safe string for ChromaDB metadata (no None allowed)
# ---------------------------------------------------------------------------

def _safe_str(value) -> str:
    """Convert None or any value to str; ChromaDB rejects None in metadata."""
    if value is None:
        return ""
    return str(value)


def _safe_list_str(lst: list) -> str:
    """JSON-serialise a list to string for ChromaDB metadata."""
    if not lst:
        return "[]"
    return json.dumps(lst, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Converter: chunk_to_chromadb
# ---------------------------------------------------------------------------

def chunk_to_chromadb(chunk: ScriptureChunk) -> dict:
    """
    Convert a ScriptureChunk to a ChromaDB-compatible dict.

    Returns:
        {
            "id": str,
            "document": str,        # the text that gets embedded
            "metadata": dict        # flat key→str/int/float/bool, no None
        }
    Lists are JSON-serialised to strings because ChromaDB rejects list values
    in metadata.
    """
    return {
        "id": chunk.chunk_id,
        "document": chunk.text or chunk.text_summary or chunk.chunk_id,
        "metadata": {
            # Identity
            "chunk_id": _safe_str(chunk.chunk_id),
            "kanda": _safe_str(chunk.kanda),
            "kanda_number": chunk.kanda_number,
            "kanda_english": _safe_str(chunk.kanda_english),
            "sarga_number": chunk.sarga_number,
            "sarga_number_global": chunk.sarga_number_global,
            # Text
            "text_summary": _safe_str(chunk.text_summary),
            "text_sanskrit_full": _safe_str(chunk.text_sanskrit_full),
            # Shloka range
            "shloka_start": chunk.shloka_start,
            "shloka_end": chunk.shloka_end,
            "shloka_count": chunk.shloka_count,
            # Entities (lists → JSON strings)
            "characters_present": _safe_list_str(chunk.characters_present),
            "characters_speaking": _safe_list_str(chunk.characters_speaking),
            "characters_primary": _safe_list_str(chunk.characters_primary),
            "locations_mentioned": _safe_list_str(chunk.locations_mentioned),
            "objects_mentioned": _safe_list_str(chunk.objects_mentioned),
            # Events & narrative
            "event_ids": _safe_list_str(chunk.event_ids),
            "event_type": _safe_str(chunk.event_type),
            "narrative_arc": _safe_str(chunk.narrative_arc),
            "emotional_tone": _safe_str(chunk.emotional_tone),
            "themes": _safe_list_str(chunk.themes),
            # Navigation
            "prev_sarga_id": _safe_str(chunk.prev_sarga_id),
            "next_sarga_id": _safe_str(chunk.next_sarga_id),
            "related_sarga_ids": _safe_list_str(chunk.related_sarga_ids),
            # Provenance
            "translation_source": _safe_str(chunk.translation_source),
            "is_interpolated": bool(chunk.is_interpolated),
            "raw_source_url": _safe_str(chunk.raw_source_url),
            "embedding_model": _safe_str(chunk.embedding_model),
            "chunk_version": _safe_str(chunk.chunk_version),
        },
    }


# ---------------------------------------------------------------------------
# Converter: chunk_to_neo4j_cypher
# ---------------------------------------------------------------------------

def chunk_to_neo4j_cypher(chunk: ScriptureChunk) -> list[str]:
    """
    Convert a ScriptureChunk to a list of Cypher query strings.
    Uses MERGE (not CREATE) for all nodes to ensure idempotency.

    Returns a list of Cypher strings to be executed in order.
    """
    queries: list[str] = []

    # 1. MERGE Kanda node
    queries.append(
        f"MERGE (k:Kanda {{key: '{chunk.kanda}'}}) "
        f"SET k.number = {chunk.kanda_number}, "
        f"k.english_name = '{chunk.kanda_english}';"
    )

    # 2. MERGE Sarga (chunk) node
    escaped_url = chunk.raw_source_url.replace("'", "\\'")
    escaped_summary = chunk.text_summary.replace("'", "\\'")[:500]
    queries.append(
        f"MERGE (s:Sarga {{chunk_id: '{chunk.chunk_id}'}}) "
        f"SET s.sarga_number = {chunk.sarga_number}, "
        f"s.sarga_number_global = {chunk.sarga_number_global}, "
        f"s.shloka_start = {chunk.shloka_start}, "
        f"s.shloka_end = {chunk.shloka_end}, "
        f"s.shloka_count = {chunk.shloka_count}, "
        f"s.event_type = '{chunk.event_type}', "
        f"s.narrative_arc = '{chunk.narrative_arc}', "
        f"s.emotional_tone = '{chunk.emotional_tone}', "
        f"s.translation_source = '{chunk.translation_source}', "
        f"s.is_interpolated = {str(chunk.is_interpolated).lower()}, "
        f"s.raw_source_url = '{escaped_url}', "
        f"s.chunk_version = '{chunk.chunk_version}', "
        f"s.text_summary = '{escaped_summary}';"
    )

    # 3. MERGE relationship: Sarga -[:BELONGS_TO]-> Kanda
    queries.append(
        f"MATCH (s:Sarga {{chunk_id: '{chunk.chunk_id}'}}), "
        f"(k:Kanda {{key: '{chunk.kanda}'}}) "
        f"MERGE (s)-[:BELONGS_TO]->(k);"
    )

    # 4. MERGE Character nodes and relationships
    for char_id in chunk.characters_present:
        char_id_clean = char_id.replace("'", "\\'")
        queries.append(
            f"MERGE (c:Character {{canonical_id: '{char_id_clean}'}});"
        )
        queries.append(
            f"MATCH (s:Sarga {{chunk_id: '{chunk.chunk_id}'}}), "
            f"(c:Character {{canonical_id: '{char_id_clean}'}}) "
            f"MERGE (c)-[:APPEARS_IN]->(s);"
        )

    for char_id in chunk.characters_speaking:
        char_id_clean = char_id.replace("'", "\\'")
        queries.append(
            f"MATCH (s:Sarga {{chunk_id: '{chunk.chunk_id}'}}), "
            f"(c:Character {{canonical_id: '{char_id_clean}'}}) "
            f"MERGE (c)-[:SPEAKS_IN]->(s);"
        )

    for char_id in chunk.characters_primary:
        char_id_clean = char_id.replace("'", "\\'")
        queries.append(
            f"MATCH (s:Sarga {{chunk_id: '{chunk.chunk_id}'}}), "
            f"(c:Character {{canonical_id: '{char_id_clean}'}}) "
            f"MERGE (c)-[:PRIMARY_IN]->(s);"
        )

    # 5. MERGE Location nodes and relationships
    for loc in chunk.locations_mentioned:
        loc_clean = loc.replace("'", "\\'")
        queries.append(
            f"MERGE (l:Location {{canonical_id: '{loc_clean}'}});"
        )
        queries.append(
            f"MATCH (s:Sarga {{chunk_id: '{chunk.chunk_id}'}}), "
            f"(l:Location {{canonical_id: '{loc_clean}'}}) "
            f"MERGE (s)-[:SET_IN]->(l);"
        )

    # 6. MERGE Event nodes and relationships
    for event_id in chunk.event_ids:
        event_id_clean = event_id.replace("'", "\\'")
        queries.append(
            f"MERGE (e:Event {{event_id: '{event_id_clean}'}});"
        )
        queries.append(
            f"MATCH (s:Sarga {{chunk_id: '{chunk.chunk_id}'}}), "
            f"(e:Event {{event_id: '{event_id_clean}'}}) "
            f"MERGE (s)-[:CONTAINS_EVENT]->(e);"
        )

    # 7. MERGE navigation edges
    if chunk.prev_sarga_id:
        queries.append(
            f"MATCH (s:Sarga {{chunk_id: '{chunk.chunk_id}'}}), "
            f"(p:Sarga {{chunk_id: '{chunk.prev_sarga_id}'}}) "
            f"MERGE (p)-[:NEXT_SARGA]->(s);"
        )
    if chunk.next_sarga_id:
        queries.append(
            f"MATCH (s:Sarga {{chunk_id: '{chunk.chunk_id}'}}), "
            f"(n:Sarga {{chunk_id: '{chunk.next_sarga_id}'}}) "
            f"MERGE (s)-[:NEXT_SARGA]->(n);"
        )

    return queries


# ---------------------------------------------------------------------------
# Self-test (run with: python chunking_schema.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Verify KANDA_REGISTRY
    assert len(KANDA_REGISTRY) == 7, "Expected 7 kandas"
    sarga_total = sum(v["sarga_count"] for v in KANDA_REGISTRY.values())
    assert sarga_total == 645, f"Expected 645 total sargas, got {sarga_total}"

    # Build a sample chunk
    sample = ScriptureChunk(
        chunk_id="sundara_kanda_0014",
        kanda="sundara_kanda",
        kanda_number=5,
        kanda_english="Book of Beauty",
        sarga_number=14,
        sarga_number_global=352,
        text="Hanuman searched the inner apartments of Lanka for Sita.",
        text_sanskrit_full="",
        text_summary="Hanuman searches Lanka.",
        shloka_start=1,
        shloka_end=40,
        shloka_count=40,
        characters_present=["hanuman", "sita"],
        characters_speaking=["hanuman"],
        characters_primary=["hanuman"],
        locations_mentioned=["lanka"],
        event_ids=["searches_ashoka_grove"],
        event_type="discovery",
        narrative_arc="rising_action",
        emotional_tone="anxious",
        themes=["devotion", "search"],
        prev_sarga_id="sundara_kanda_0013",
        next_sarga_id="sundara_kanda_0015",
        raw_source_url="https://www.valmikiramayan.net/sundara/sarga14/sundaraitd14.htm",
        embedding_model="text-embedding-3-large",
        chunk_version="1.0.0",
    )

    # Test ChromaDB converter
    chroma = chunk_to_chromadb(sample)
    assert chroma["id"] == "sundara_kanda_0014"
    assert isinstance(chroma["document"], str) and chroma["document"]
    for k, v in chroma["metadata"].items():
        assert v is not None, f"ChromaDB metadata key '{k}' is None — forbidden"

    # Test Neo4j converter
    cypher_queries = chunk_to_neo4j_cypher(sample)
    assert len(cypher_queries) > 0
    for q in cypher_queries:
        assert "MERGE" in q or "MATCH" in q, f"Non-MERGE/MATCH query: {q}"
        assert "CREATE" not in q, f"Forbidden CREATE found: {q}"

    # Test None safety
    null_chunk = ScriptureChunk(
        chunk_id="test_null",
        kanda="bala_kanda",
        kanda_number=1,
        kanda_english="Book of Youth",
        sarga_number=1,
        sarga_number_global=1,
        prev_sarga_id=None,
        next_sarga_id=None,
    )
    null_chroma = chunk_to_chromadb(null_chunk)
    assert null_chroma["metadata"]["prev_sarga_id"] == ""
    assert null_chroma["metadata"]["next_sarga_id"] == ""

    print("[OK] KANDA_REGISTRY: 7 kandas, 645 sargas total")
    print("[OK] ScriptureChunk dataclass instantiated")
    print(f"[OK] chunk_to_chromadb: {len(chroma['metadata'])} metadata keys, no None values")
    print(f"[OK] chunk_to_neo4j_cypher: {len(cypher_queries)} Cypher statements, all use MERGE")
    print("[OK] None -> empty-string conversion verified")
    print("All chunking_schema checks passed.")
