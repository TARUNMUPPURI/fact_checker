"""
chroma_ingest.py
----------------
Ingests extracted sarga metadata into ChromaDB using OpenAI
text-embedding-3-large for embeddings.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripture_gpt.data.chunking_schema import (
    KANDA_REGISTRY,
    ScriptureChunk,
    chunk_to_chromadb,
)

log = logging.getLogger(__name__)

EMBED_MODEL = "text-embedding-3-large"
EMBED_BATCH_SIZE = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_meta_value(v) -> str | int | float | bool:
    """Coerce value to ChromaDB-compatible scalar — None → '', lists → JSON str."""
    if v is None:
        return ""
    if isinstance(v, list):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    return str(v)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# ChromaIngestor
# ---------------------------------------------------------------------------

class ChromaIngestor:

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "scripture_sargas",
    ):
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise ConnectionError(
                f"ChromaDB unreachable at '{persist_dir}': {exc}"
            ) from exc

        try:
            from openai import OpenAI
            self._oai = OpenAI()
        except Exception as exc:
            raise ConnectionError(
                f"OpenAI client could not be initialised: {exc}"
            ) from exc

        self._collection_name = collection_name

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI embeddings API for a batch of texts."""
        response = self._oai.embeddings.create(
            model=EMBED_MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_all(self, extracted_dir: str | Path, raw_sargas_dir: str | Path) -> None:
        """
        Ingest every extraction JSON in extracted_dir.
        Fetches the corresponding raw sarga JSON for the full English text.
        """
        extracted_dir = Path(extracted_dir)
        raw_sargas_dir = Path(raw_sargas_dir)

        files = sorted(f for f in extracted_dir.glob("*.json")
                       if f.name != "review_queue.json")
        total = len(files)
        log.info("ChromaDB: %d extraction files to ingest", total)

        # Process in batches of EMBED_BATCH_SIZE
        batch_ids: list[str] = []
        batch_texts: list[str] = []
        batch_meta: list[dict] = []

        def _flush():
            if not batch_ids:
                return
            embeddings = self._embed_batch(batch_texts)
            self._collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_meta,
            )
            batch_ids.clear()
            batch_texts.clear()
            batch_meta.clear()

        for idx, ext_path in enumerate(files, start=1):
            chunk_id = ext_path.stem
            print(f"  Ingesting [{idx:3d}/{total}] {chunk_id}...", end="\r", flush=True)

            try:
                extracted = _load_json(ext_path)
                raw_path = raw_sargas_dir / f"{chunk_id}.json"
                raw_text = ""
                if raw_path.exists():
                    raw_sarga = _load_json(raw_path)
                    raw_text = raw_sarga.get("text", "")

                ok = self._prepare_batch(
                    chunk_id, extracted, raw_text,
                    batch_ids, batch_texts, batch_meta,
                )
                if not ok:
                    continue

            except Exception as exc:
                log.error("  [%s] failed: %s", chunk_id, exc)
                continue

            if len(batch_ids) >= EMBED_BATCH_SIZE:
                _flush()

        _flush()
        print(f"\nChromaDB ingest complete: {total} sargas processed.")

    def ingest_one(
        self,
        chunk_id: str,
        extracted_data: dict,
        raw_text: str,
    ) -> bool:
        """
        Ingest a single sarga. Returns True on success.
        """
        batch_ids: list[str] = []
        batch_texts: list[str] = []
        batch_meta: list[dict] = []

        ok = self._prepare_batch(chunk_id, extracted_data, raw_text,
                                  batch_ids, batch_texts, batch_meta)
        if not ok or not batch_ids:
            return False
        try:
            embeddings = self._embed_batch(batch_texts)
            self._collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_meta,
            )
            return True
        except Exception as exc:
            log.error("ingest_one failed for %s: %s", chunk_id, exc)
            return False

    # ------------------------------------------------------------------
    # Prepare one record for batching
    # ------------------------------------------------------------------

    def _prepare_batch(
        self,
        chunk_id: str,
        extracted: dict,
        raw_text: str,
        batch_ids: list,
        batch_texts: list,
        batch_meta: list,
    ) -> bool:
        # Build a minimal ScriptureChunk from extracted data for chunk_to_chromadb
        try:
            kanda = extracted.get("kanda", "")
            kreg = KANDA_REGISTRY.get(kanda, {})
            chunk = ScriptureChunk(
                chunk_id=chunk_id,
                kanda=kanda,
                kanda_number=kreg.get("number", 0),
                kanda_english=extracted.get("kanda_english", kreg.get("english_name", "")),
                sarga_number=int(extracted.get("sarga_number", 0)),
                sarga_number_global=int(extracted.get("sarga_number_global", 0)),
                text=raw_text,
                text_summary=extracted.get("text_summary", ""),
                characters_present=extracted.get("characters_present", []),
                characters_speaking=extracted.get("characters_speaking", []),
                characters_primary=extracted.get("characters_primary", []),
                locations_mentioned=extracted.get("locations_mentioned", []),
                objects_mentioned=extracted.get("objects_mentioned", []),
                event_ids=extracted.get("event_ids", []),
                event_type=extracted.get("event_type", "other"),
                narrative_arc=extracted.get("narrative_arc", "rising_action"),
                emotional_tone=extracted.get("emotional_tone", ""),
                themes=extracted.get("themes", []),
                is_interpolated=bool(extracted.get("is_interpolated", False)),
            )
        except Exception as exc:
            log.error("Could not build ScriptureChunk for %s: %s", chunk_id, exc)
            return False

        chroma_doc = chunk_to_chromadb(chunk)
        text = chroma_doc["document"]
        if not text.strip():
            log.warning("%s: empty text — skipping embedding", chunk_id)
            return False

        # Sanitise metadata — all values must be scalar (str/int/float/bool)
        meta = {k: _safe_meta_value(v) for k, v in chroma_doc["metadata"].items()}

        batch_ids.append(chroma_doc["id"])
        batch_texts.append(text)
        batch_meta.append(meta)
        return True

    # ------------------------------------------------------------------
    # Search helper
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        character_filter: Optional[str] = None,
        kanda_filter: Optional[str] = None,
        top_k: int = 8,
    ) -> list[dict]:
        """
        Semantic search over the collection.

        Args:
            query_text: natural language query
            character_filter: canonical_id substring; filters by
                characters_present metadata field
            kanda_filter: exact kanda key to restrict results
            top_k: number of results to return

        Returns:
            list of {chunk_id, text, metadata, distance}
        """
        # Build where clause
        conditions: list[dict] = []
        if kanda_filter:
            conditions.append({"kanda": {"$eq": kanda_filter}})
        if character_filter:
            # ChromaDB string contains
            conditions.append(
                {"characters_present": {"$contains": character_filter}}
            )

        where: Optional[dict] = None
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        # Embed query
        embed_resp = self._oai.embeddings.create(
            model=EMBED_MODEL,
            input=[query_text],
        )
        query_embedding = embed_resp.data[0].embedding

        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        output: list[dict] = []
        for i, cid in enumerate(results["ids"][0]):
            output.append({
                "chunk_id": cid,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return output
