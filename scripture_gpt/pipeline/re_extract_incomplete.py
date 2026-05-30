"""
re_extract_incomplete.py
------------------------
Fixes all 461 "partial" sargas without touching the 74 successful ones.

Two strategies:
  Phase 1 — SHORT sargas (raw text <= 6,000 chars):
    The LLM already saw the full text in the original run and answered correctly,
    but the old 21-character whitelist stripped valid answers (e.g. Narada, Valmiki).
    FIX: Re-parse the existing raw_llm_response on disk using the NEW, expanded
    whitelist (35 characters).  Zero API calls.  Completely free.

  Phase 2 — LONG sargas (raw text > 6,000 chars):
    The original prompt truncated the text to 6,000 chars so the LLM genuinely
    missed events/characters in the latter 80%+ of the chapter.
    FIX: Re-call the LLM with the full text (up to 45,000 chars).

After completion, writes a manifest file listing all updated chunk IDs so that
ingest_all.py can selectively re-embed only those sargas in ChromaDB.

Usage:
    # Dry-run (no API calls, no file writes):
    uv run python scripture_gpt/pipeline/re_extract_incomplete.py --dry-run

    # Full run:
    uv run python scripture_gpt/pipeline/re_extract_incomplete.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripture_gpt.pipeline.llm_extractor import LLMExtractor, ExtractionError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("re_extract_incomplete")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHORT_LIMIT = 6_000          # chars; <= this means LLM already saw the full text

# gpt-4o-mini pricing (USD per 1M tokens)
INPUT_COST_PER_M  = 0.150
OUTPUT_COST_PER_M = 0.600
CHARS_PER_TOKEN   = 4.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kanda_from_chunk_id(chunk_id: str) -> str:
    """Derive kanda key from chunk_id. e.g. 'bala_kanda_0001' -> 'bala_kanda'"""
    parts = chunk_id.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else ""


def _save(result_dict: dict, output_dir: Path, orig_ext: dict, raw_data: dict, chunk_id: str) -> None:
    """Write the updated extraction JSON, preserving sarga metadata."""
    path = output_dir / f"{chunk_id}.json"

    # Preserve sarga-identity fields that live in the raw JSON but not in ExtractionResult
    for field_name, fallback in [
        ("kanda",               _kanda_from_chunk_id(chunk_id)),
        ("kanda_english",       raw_data.get("kanda_english", orig_ext.get("kanda_english", ""))),
        ("sarga_number",        raw_data.get("sarga_number",  orig_ext.get("sarga_number", 0))),
        ("sarga_number_global", raw_data.get("sarga_number_global", orig_ext.get("sarga_number_global", 0))),
    ]:
        if not result_dict.get(field_name):
            result_dict[field_name] = fallback

    result_dict["extracted_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.write_text(json.dumps(result_dict, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Re-extract partial sargas with improved whitelist and full-text prompts."
    )
    parser.add_argument("--extracted-dir",  default="scripture_gpt/output/extracted",
                        help="Directory of existing extraction JSONs")
    parser.add_argument("--raw-sargas-dir", default="scripture_gpt/raw_sargas",
                        help="Directory of raw sarga JSONs")
    parser.add_argument("--aliases",        default="scripture_gpt/data/character_aliases.json",
                        help="Path to character_aliases.json")
    parser.add_argument("--manifest",       default="scripture_gpt/output/re_extracted_manifest.txt",
                        help="Output file: list of updated chunk IDs for selective ChromaDB re-ingest")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds between API calls (default: 0.3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without making API calls or writing files")
    args = parser.parse_args(argv)

    ext_dir = Path(args.extracted_dir)
    raw_dir = Path(args.raw_sargas_dir)

    # ------------------------------------------------------------------
    # Step 1: Classify all partial sargas into short vs long
    # ------------------------------------------------------------------
    all_ext_files = sorted(f for f in ext_dir.glob("*.json") if f.name != "review_queue.json")

    short_partials: list[tuple] = []
    long_partials:  list[tuple] = []

    for f in all_ext_files:
        raw_p = raw_dir / f.name
        if not raw_p.exists():
            log.warning("No matching raw sarga for %s — skipping", f.stem)
            continue
        try:
            ext_data = _load_json(f)
            raw_data = _load_json(raw_p)
        except Exception as exc:
            log.warning("Could not read %s: %s — skipping", f.stem, exc)
            continue

        if ext_data.get("extraction_status") != "partial":
            continue  # skip 'success' and 'failed' (none exist, but defensive)

        raw_text = raw_data.get("text", "")
        entry = (f, raw_p, f.stem, raw_text, ext_data, raw_data)
        if len(raw_text) <= SHORT_LIMIT:
            short_partials.append(entry)
        else:
            long_partials.append(entry)

    # ------------------------------------------------------------------
    # Step 2: Cost estimate
    # ------------------------------------------------------------------
    est_input_tokens = sum(
        (min(len(e[3]), 45000) / CHARS_PER_TOKEN) + 875
        for e in long_partials
    )
    est_output_tokens = len(long_partials) * 600
    est_cost = (
        (est_input_tokens  / 1_000_000) * INPUT_COST_PER_M +
        (est_output_tokens / 1_000_000) * OUTPUT_COST_PER_M
    )

    print()
    print("=" * 65)
    print("  Re-Extract Incomplete Sargas")
    print("=" * 65)
    print(f"  Phase 1 — short partials  (FREE, re-parse from disk) : {len(short_partials)}")
    print(f"  Phase 2 — long partials   (API re-call, full text)   : {len(long_partials)}")
    print(f"  Estimated API cost                                     : ${est_cost:.4f}")
    print(f"  Manifest will be written to                            : {args.manifest}")
    print(f"  Dry run                                                : {args.dry_run}")
    print("=" * 65)
    print()

    if args.dry_run:
        print("Dry-run mode — no files written, no API calls made.")
        print()
        print("Short partial examples:")
        for (_, _, cid, rt, ed, _) in short_partials[:5]:
            print(f"  {cid}  ({len(rt)} chars)  unknowns: {ed.get('unknown_characters', [])[:3]}")
        print()
        print("Long partial examples (top 5 by length):")
        for (_, _, cid, rt, _, _) in sorted(long_partials, key=lambda x: len(x[3]), reverse=True)[:5]:
            print(f"  {cid}  ({len(rt):,} chars)")
        return

    # ------------------------------------------------------------------
    # Step 3: Load extractor (uses the updated 35-character aliases)
    # ------------------------------------------------------------------
    extractor = LLMExtractor(str(args.aliases))

    updated_ids: list[str] = []
    counts = {"free": 0, "api": 0, "failed": 0}
    total_api_tokens = 0
    total_api_cost   = 0.0

    # ------------------------------------------------------------------
    # Phase 1: Short sargas — re-parse raw_llm_response (ZERO API cost)
    # ------------------------------------------------------------------
    print(f"--- Phase 1: Re-parsing {len(short_partials)} short sargas (no API calls) ---")
    print()

    for idx, (ext_path, raw_p, chunk_id, raw_text, ext_data, raw_data) in enumerate(short_partials, 1):
        raw_llm    = ext_data.get("raw_llm_response", "")
        tokens_used = ext_data.get("tokens_used", 0)

        if not raw_llm:
            log.warning("[%3d/%d] %s — empty raw_llm_response, cannot re-parse", idx, len(short_partials), chunk_id)
            counts["failed"] += 1
            continue

        parsed = LLMExtractor._parse_json(raw_llm)
        if parsed is None:
            log.warning("[%3d/%d] %s — raw_llm_response is not valid JSON, skipping", idx, len(short_partials), chunk_id)
            counts["failed"] += 1
            continue

        result = LLMExtractor._build_result(chunk_id, parsed, raw_llm, tokens_used)
        result = extractor._validate(result)

        out = asdict(result)
        _save(out, ext_dir, ext_data, raw_data, chunk_id)
        updated_ids.append(chunk_id)
        counts["free"] += 1

        sym    = "[OK]" if result.extraction_status == "success" else "[~] "
        unkn   = len(result.unknown_characters)
        chars  = result.characters_present[:4]
        print(
            f"  [{idx:3d}/{len(short_partials)}] {chunk_id:<28} {sym}"
            f"  conf={result.confidence:.2f}"
            f"  chars={chars}"
            + (f"  [{unkn} still unknown]" if unkn else ""),
            flush=True,
        )

    # ------------------------------------------------------------------
    # Phase 2: Long sargas — full API re-call with expanded text
    # ------------------------------------------------------------------
    print()
    print(f"--- Phase 2: Re-extracting {len(long_partials)} long sargas via API ---")
    print()

    for idx, (ext_path, raw_p, chunk_id, raw_text, ext_data, raw_data) in enumerate(long_partials, 1):
        kanda_english = raw_data.get("kanda_english", ext_data.get("kanda_english", ""))
        sarga_number  = int(raw_data.get("sarga_number", ext_data.get("sarga_number", 0)))

        try:
            result = extractor.extract(
                chunk_id=chunk_id,
                sarga_text=raw_text,          # full text — llm_extractor now caps at 45,000 chars
                kanda_english=kanda_english,
                sarga_number=sarga_number,
            )
        except ExtractionError as exc:
            log.error("[%3d/%d] %s — ExtractionError: %s", idx, len(long_partials), chunk_id, exc)
            counts["failed"] += 1
            continue
        except Exception as exc:
            log.error("[%3d/%d] %s — Unexpected error: %s", idx, len(long_partials), chunk_id, exc)
            counts["failed"] += 1
            continue

        # Cost tracking
        tok = result.tokens_used
        total_api_tokens += tok
        # Approximate split: 85% input, 15% output
        cost = (tok * 0.85 / 1_000_000) * INPUT_COST_PER_M + \
               (tok * 0.15 / 1_000_000) * OUTPUT_COST_PER_M
        total_api_cost += cost

        out = asdict(result)
        _save(out, ext_dir, ext_data, raw_data, chunk_id)
        updated_ids.append(chunk_id)
        counts["api"] += 1

        sym   = "[OK]" if result.extraction_status == "success" else "[~] "
        unkn  = len(result.unknown_characters)
        chars = result.characters_present[:4]
        print(
            f"  [{idx:3d}/{len(long_partials)}] {chunk_id:<28} {sym}"
            f"  conf={result.confidence:.2f}"
            f"  tok={tok}"
            f"  chars={chars}"
            + (f"  [{unkn} still unknown]" if unkn else ""),
            flush=True,
        )

        if args.delay > 0:
            time.sleep(args.delay)

    # ------------------------------------------------------------------
    # Step 4: Write manifest for selective ChromaDB re-ingest
    # ------------------------------------------------------------------
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(updated_ids), encoding="utf-8")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("  Re-extraction complete")
    print("=" * 65)
    print(f"  Free re-parses (Phase 1) : {counts['free']}")
    print(f"  API re-calls   (Phase 2) : {counts['api']}")
    print(f"  Failed / skipped         : {counts['failed']}")
    print(f"  Total API tokens used    : {total_api_tokens:,}")
    print(f"  Total API cost           : ${total_api_cost:.4f}")
    print(f"  Manifest written         : {manifest_path}  ({len(updated_ids)} IDs)")
    print("=" * 65)
    print()
    print("  Next step — selectively re-ingest only updated sargas into ChromaDB:")
    print(f"  uv run python scripture_gpt/storage/ingest_all.py --skip-neo4j --only-ids-file {manifest_path}")
    print()


if __name__ == "__main__":
    main()
