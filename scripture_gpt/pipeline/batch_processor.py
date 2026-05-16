"""
batch_processor.py
------------------
Batch-processes raw sarga JSON files through the LLM extractor and
writes structured extraction results with a review queue.
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

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripture_gpt.pipeline.llm_extractor import (
    ExtractionError,
    ExtractionResult,
    LLMExtractor,
)

# ---------------------------------------------------------------------------
# Cost constants (Claude Haiku)
# ---------------------------------------------------------------------------
INPUT_COST_PER_TOKEN = 0.80 / 1_000_000
OUTPUT_COST_PER_TOKEN = 4.00 / 1_000_000

# Estimate: typical sarga produces ~800 input + ~400 output tokens
_EST_INPUT_TOKENS = 800
_EST_OUTPUT_TOKENS = 400
_EST_COST_PER_SARGA = (
    _EST_INPUT_TOKENS * INPUT_COST_PER_TOKEN
    + _EST_OUTPUT_TOKENS * OUTPUT_COST_PER_TOKEN
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch_processor")


# ---------------------------------------------------------------------------
# BatchProcessor
# ---------------------------------------------------------------------------

class BatchProcessor:

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        aliases_path: str | Path,
        confidence_threshold: float = 0.7,
        delay: float = 0.3,
        dry_run: bool = False,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.aliases_path = Path(aliases_path)
        self.confidence_threshold = confidence_threshold
        self.delay = delay
        self.dry_run = dry_run

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._review_queue_path = self.output_dir / "review_queue.json"

        self._extractor: LLMExtractor | None = None  # lazy-init (skipped in dry_run)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        start_time = time.time()

        # --- Scan inputs ---
        all_files = sorted(self.input_dir.glob("*.json"))
        total = len(all_files)

        already_done = [
            f for f in all_files
            if (self.output_dir / f.name).exists()
        ]
        pending_files = [
            f for f in all_files
            if f not in set(already_done)
        ]
        pending = len(pending_files)
        done = len(already_done)

        est_cost = pending * _EST_COST_PER_SARGA

        print(
            f"\n{'='*60}\n"
            f"  Scripture GPT -- Batch Extractor\n"
            f"  Input : {self.input_dir}\n"
            f"  Output: {self.output_dir}\n"
            f"{'='*60}\n"
            f"  Total sargas   : {total}\n"
            f"  Already done   : {done}\n"
            f"  Pending        : {pending}\n"
            f"  Estimated cost : ${est_cost:.4f} USD\n"
            f"  Dry run        : {self.dry_run}\n"
            f"{'='*60}\n"
        )

        if self.dry_run:
            print("Dry-run mode: no API calls made.")
            # Ensure review_queue.json exists even in dry-run
            self._write_review_queue([])
            return

        if pending == 0:
            print("Nothing to do.")
            self._write_review_queue(self._load_existing_review_queue())
            return

        # --- Lazy-init extractor ---
        self._extractor = LLMExtractor(str(self.aliases_path))

        # --- Process ---
        counts = {"success": 0, "partial": 0, "failed": 0}
        total_tokens = 0
        total_cost = 0.0
        review_entries: list[dict] = self._load_existing_review_queue()

        for idx, src_path in enumerate(pending_files, start=1):
            chunk_id = src_path.stem
            result = self._process_one(src_path, chunk_id, idx, pending)

            if result is None:
                counts["failed"] += 1
                continue

            counts[result.extraction_status] += 1
            total_tokens += result.tokens_used

            actual_cost = result.tokens_used * (
                INPUT_COST_PER_TOKEN + OUTPUT_COST_PER_TOKEN
            ) / 2  # rough split
            total_cost += actual_cost

            # Flag for review?
            needs_review = (
                result.confidence < self.confidence_threshold
                or bool(result.unknown_characters)
                or bool(result.unknown_locations)
            )
            if needs_review:
                review_entries.append({
                    "chunk_id": chunk_id,
                    "confidence": result.confidence,
                    "unknown_characters": result.unknown_characters,
                    "unknown_locations": result.unknown_locations,
                    "extraction_status": result.extraction_status,
                })

            # Persist result immediately (crash-safe)
            self._save_result(result)

            # Persist review queue after every sarga
            self._write_review_queue(review_entries)

            if self.delay > 0:
                time.sleep(self.delay)

        elapsed = time.time() - start_time
        print(
            f"\n{'='*60}\n"
            f"  Extraction complete in {elapsed:.1f}s\n"
            f"  Success  : {counts['success']}\n"
            f"  Partial  : {counts['partial']}\n"
            f"  Failed   : {counts['failed']}\n"
            f"  Tokens   : {total_tokens:,}\n"
            f"  Cost     : ${total_cost:.4f} USD\n"
            f"  Review Q : {len(review_entries)} entries\n"
            f"{'='*60}\n"
        )

    # ------------------------------------------------------------------
    # Process one sarga
    # ------------------------------------------------------------------

    def _process_one(
        self,
        src_path: Path,
        chunk_id: str,
        idx: int,
        total_pending: int,
    ) -> ExtractionResult | None:

        # Load raw sarga
        try:
            raw = json.loads(src_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.error("[%3d/%d] %s  -- failed to read source: %s", idx, total_pending, chunk_id, exc)
            return None

        sarga_text: str = raw.get("text", "")
        kanda_english: str = raw.get("kanda_english", "")
        sarga_number: int = int(raw.get("sarga_number", 0))

        if not sarga_text.strip():
            log.warning("[%3d/%d] %s  -- empty text, skipping", idx, total_pending, chunk_id)
            return None

        # Call extractor
        try:
            result = self._extractor.extract(
                chunk_id=chunk_id,
                sarga_text=sarga_text,
                kanda_english=kanda_english,
                sarga_number=sarga_number,
            )
        except ExtractionError as exc:
            log.error("[%3d/%d] %s  -- ExtractionError: %s", idx, total_pending, chunk_id, exc)
            return None
        except Exception as exc:
            log.error("[%3d/%d] %s  -- Unexpected error: %s", idx, total_pending, chunk_id, exc)
            return None

        # Console log line
        status_sym = "[OK]" if result.extraction_status == "success" else "[~]"
        char_preview = result.characters_present[:4]
        unknown_note = f"  [{len(result.unknown_characters)} unknown]" if result.unknown_characters else ""
        print(
            f"[{idx:3d}/{total_pending}] {chunk_id}  {status_sym}"
            f"  conf={result.confidence:.2f}"
            f"  chars={char_preview}"
            f"{unknown_note}",
            flush=True,
        )

        return result

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_result(self, result: ExtractionResult) -> None:
        out_path = self.output_dir / f"{result.chunk_id}.json"
        data = asdict(result)
        data["extracted_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_review_queue(self, entries: list[dict]) -> None:
        self._review_queue_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_existing_review_queue(self) -> list[dict]:
        if self._review_queue_path.exists():
            try:
                data = json.loads(
                    self._review_queue_path.read_text(encoding="utf-8")
                )
                return data if isinstance(data, list) else []
            except Exception:
                pass
        return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Batch-extract metadata from raw sarga JSONs via Claude Haiku",
    )
    parser.add_argument(
        "--input", required=True,
        help="Directory containing raw sarga JSON files",
    )
    parser.add_argument(
        "--output", required=True,
        help="Directory to write extraction results",
    )
    parser.add_argument(
        "--aliases", required=True,
        help="Path to character_aliases.json",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan and cost estimate without making API calls",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.7,
        help="Confidence below this value flags a sarga for review (default: 0.7)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.3,
        help="Seconds between API calls (default: 0.3)",
    )
    args = parser.parse_args(argv)

    processor = BatchProcessor(
        input_dir=args.input,
        output_dir=args.output,
        aliases_path=args.aliases,
        confidence_threshold=args.confidence_threshold,
        delay=args.delay,
        dry_run=args.dry_run,
    )
    processor.run()


if __name__ == "__main__":
    main()
