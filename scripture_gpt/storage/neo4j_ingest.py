"""
neo4j_ingest.py
---------------
Ingests extracted sarga metadata and character aliases into Neo4j
as a knowledge graph. All nodes and relationships use MERGE for idempotency.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripture_gpt.data.chunking_schema import KANDA_REGISTRY

log = logging.getLogger(__name__)


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


class Neo4jIngestor:

    def __init__(self, uri: str, user: str, password: str):
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            self._driver.verify_connectivity()
        except Exception as exc:
            raise ConnectionError(
                f"Neo4j unreachable at '{uri}': {exc}"
            ) from exc

    def close(self) -> None:
        self._driver.close()

    # ------------------------------------------------------------------
    def ingest_all(self, extracted_dir: str | Path, aliases_path: str | Path) -> None:
        """
        Strict ingestion order:
          1. Character nodes
          2. Location nodes
          3. Sarga nodes
          4. PRECEDES relationships
          5. APPEARS_IN relationships
          6. LOCATED_IN relationships
        """
        extracted_dir = Path(extracted_dir)
        aliases_path = Path(aliases_path)

        aliases_data = _load_json(aliases_path)
        characters: dict = aliases_data.get("characters", {})
        locations: dict = aliases_data.get("locations", {})

        files = sorted(
            f for f in extracted_dir.glob("*.json")
            if f.name != "review_queue.json"
        )
        total = len(files)

        extracted_by_id: dict[str, dict] = {}
        for f in files:
            try:
                extracted_by_id[f.stem] = _load_json(f)
            except Exception as exc:
                log.error("Could not read %s: %s", f.name, exc)

        print(f"\nNeo4j ingest: {total} sargas | {len(characters)} characters | {len(locations)} locations")

        print("  [1/6] Character nodes...")
        self._ingest_characters(characters)

        print("  [2/6] Location nodes...")
        self._ingest_locations(locations)

        print("  [3/6] Sarga nodes...")
        self._ingest_sargas(extracted_by_id, total)

        print("  [4/6] PRECEDES relationships...")
        self._ingest_precedes(extracted_by_id)

        print("  [5/6] APPEARS_IN relationships...")
        self._ingest_appears_in(extracted_by_id)

        print("  [6/6] LOCATED_IN relationships...")
        self._ingest_located_in(extracted_by_id)

        print("Neo4j ingest complete.")

    # ------------------------------------------------------------------
    def _ingest_characters(self, characters: dict) -> None:
        with self._driver.session() as session:
            for cid, char in characters.items():
                session.run(
                    "MERGE (c:Character {canonical_id: $cid}) "
                    "SET c.canonical_name=$name, c.gender=$gender, "
                    "    c.species=$species, c.role=$role",
                    cid=cid,
                    name=char.get("canonical_name", cid),
                    gender=char.get("gender", "unknown"),
                    species=char.get("species", "other"),
                    role=char.get("role", ""),
                )

    def _ingest_locations(self, locations: dict) -> None:
        with self._driver.session() as session:
            for lid, loc in locations.items():
                session.run(
                    "MERGE (l:Location {canonical_id: $lid}) "
                    "SET l.canonical_name=$name",
                    lid=lid,
                    name=loc.get("canonical_name", lid),
                )

    def _ingest_sargas(self, extracted_by_id: dict, total: int) -> None:
        items = sorted(extracted_by_id.items())
        with self._driver.session() as session:
            for idx, (chunk_id, data) in enumerate(items, start=1):
                print(f"    [{idx:3d}/{total}] {chunk_id}...", end="\r", flush=True)
                try:
                    kanda = data.get("kanda", "")
                    kreg = KANDA_REGISTRY.get(kanda, {})
                    session.run(
                        "MERGE (s:Sarga {chunk_id: $chunk_id}) "
                        "SET s.kanda=$kanda, s.kanda_number=$knum, "
                        "    s.sarga_number=$snum, s.sarga_number_global=$sglob, "
                        "    s.event_type=$etype, s.narrative_arc=$arc, "
                        "    s.text_summary=$summary, s.is_interpolated=$interp, "
                        "    s.shloka_start=$sstart, s.shloka_end=$send",
                        chunk_id=chunk_id,
                        kanda=kanda,
                        knum=kreg.get("number", 0),
                        snum=int(data.get("sarga_number", 0)),
                        sglob=int(data.get("sarga_number_global", 0)),
                        etype=data.get("event_type", "other"),
                        arc=data.get("narrative_arc", "rising_action"),
                        summary=str(data.get("text_summary", ""))[:500],
                        interp=bool(data.get("is_interpolated", False)),
                        sstart=int(data.get("shloka_start", 1)),
                        send=int(data.get("shloka_end", 1)),
                    )
                except Exception as exc:
                    log.error("Sarga %s failed: %s", chunk_id, exc)
        print()

    def _ingest_precedes(self, extracted_by_id: dict) -> None:
        ordered = sorted(
            extracted_by_id.items(),
            key=lambda x: int(x[1].get("sarga_number_global", 0)),
        )
        with self._driver.session() as session:
            for i in range(len(ordered) - 1):
                cur_id, cur_data = ordered[i]
                nxt_id, nxt_data = ordered[i + 1]
                if cur_data.get("kanda") == nxt_data.get("kanda"):
                    try:
                        session.run(
                            "MATCH (a:Sarga {chunk_id:$cur}),(b:Sarga {chunk_id:$nxt}) "
                            "MERGE (a)-[:PRECEDES]->(b)",
                            cur=cur_id, nxt=nxt_id,
                        )
                    except Exception as exc:
                        log.warning("PRECEDES %s->%s: %s", cur_id, nxt_id, exc)

    def _ingest_appears_in(self, extracted_by_id: dict) -> None:
        with self._driver.session() as session:
            for chunk_id, data in extracted_by_id.items():
                speaking = set(data.get("characters_speaking", []))
                primary = set(data.get("characters_primary", []))
                for cid in data.get("characters_present", []):
                    try:
                        session.run(
                            "MATCH (c:Character {canonical_id:$cid}),"
                            "      (s:Sarga {chunk_id:$cid2}) "
                            "MERGE (c)-[r:APPEARS_IN {chunk_id:$cid2}]->(s) "
                            "SET r.role=$role, r.speaks=$speaks",
                            cid=cid, cid2=chunk_id,
                            role="primary" if cid in primary else "supporting",
                            speaks=cid in speaking,
                        )
                    except Exception as exc:
                        log.warning("APPEARS_IN %s->%s: %s", cid, chunk_id, exc)

    def _ingest_located_in(self, extracted_by_id: dict) -> None:
        with self._driver.session() as session:
            for chunk_id, data in extracted_by_id.items():
                for lid in data.get("locations_mentioned", []):
                    try:
                        session.run(
                            "MATCH (s:Sarga {chunk_id:$cid}),"
                            "      (l:Location {canonical_id:$lid}) "
                            "MERGE (s)-[:LOCATED_IN]->(l)",
                            cid=chunk_id, lid=lid,
                        )
                    except Exception as exc:
                        log.warning("LOCATED_IN %s->%s: %s", chunk_id, lid, exc)

    # ------------------------------------------------------------------
    def get_character_sargas(self, canonical_id: str) -> list[dict]:
        """Return all sargas a character appears in, ordered chronologically."""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (c:Character {canonical_id:$cid})-[r:APPEARS_IN]->(s:Sarga) "
                "RETURN s.chunk_id AS chunk_id, s.kanda AS kanda, "
                "       s.sarga_number AS sarga_number, "
                "       s.text_summary AS text_summary, "
                "       r.role AS role, "
                "       s.sarga_number_global AS sarga_number_global "
                "ORDER BY s.sarga_number_global ASC",
                cid=canonical_id,
            )
            return [dict(record) for record in result]
