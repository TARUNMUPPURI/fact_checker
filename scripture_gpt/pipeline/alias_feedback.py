"""
alias_feedback.py
-----------------
Four sub-commands for maintaining character_aliases.json using the
review queue produced by batch_processor.py.

Sub-commands:
  analyze  -- report on unknown characters in the review queue
  suggest  -- ask Claude Haiku to propose dictionary entries
  merge    -- merge approved suggestions into character_aliases.json
  rerun    -- re-run extraction only on flagged sargas
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

MODEL = "claude-haiku-4-5-20251001"

VALID_KANDA_KEYS = {
    "bala_kanda", "ayodhya_kanda", "aranya_kanda",
    "kishkindha_kanda", "sundara_kanda", "yuddha_kanda", "uttara_kanda",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str | Path) -> dict | list:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _load_review_queue(path: str | Path) -> list[dict]:
    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"review_queue must be a JSON array: {path}")
    return data


def _strip_fences(raw: str) -> str:
    """Remove ```json ... ``` markdown fences from LLM response."""
    cleaned = re.sub(r"^```json\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned.strip())
    return cleaned


def _parse_json_response(raw: str) -> dict | None:
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Sub-command: analyze
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace) -> None:
    """
    Read review_queue.json and print a report on every unique unknown character:
    how many sargas it appeared in, first 3 chunk_ids, co-occurring unknowns.
    """
    queue = _load_review_queue(args.review_queue)

    if not queue:
        print("Review queue is empty — nothing to analyze.")
        return

    # Collect stats per unknown character name
    name_to_chunks: dict[str, list[str]] = defaultdict(list)
    name_to_cooccurring: dict[str, set[str]] = defaultdict(set)

    for entry in queue:
        chunk_id = entry.get("chunk_id", "?")
        unknowns = entry.get("unknown_characters", [])
        for name in unknowns:
            name_to_chunks[name].append(chunk_id)
            for other in unknowns:
                if other != name:
                    name_to_cooccurring[name].add(other)

    if not name_to_chunks:
        print("No unknown characters found in the review queue.")
        return

    print(f"\n{'='*60}")
    print(f"  Unknown Character Report  ({len(name_to_chunks)} unique names)")
    print(f"{'='*60}\n")

    for name in sorted(name_to_chunks, key=lambda n: -len(name_to_chunks[n])):
        chunks = name_to_chunks[name]
        preview = chunks[:3]
        co = sorted(name_to_cooccurring[name])
        print(f"  Character : {name}")
        print(f"  Sargas    : {len(chunks)}")
        print(f"  First 3   : {preview}")
        if co:
            print(f"  Co-occurs : {co}")
        print()

    print(f"  Total flagged sargas in queue : {len(queue)}")
    print()


# ---------------------------------------------------------------------------
# Sub-command: suggest
# ---------------------------------------------------------------------------

def cmd_suggest(args: argparse.Namespace) -> None:
    """
    For each unknown character not already in aliases, call Claude Haiku to
    suggest a dictionary entry, then save all suggestions to a JSON file.
    """
    import anthropic

    queue = _load_review_queue(args.review_queue)
    aliases_data = _load_json(args.aliases)
    existing_chars: set[str] = set(aliases_data.get("characters", {}).keys())

    # Map: unknown name → list of (chunk_id, text_preview)
    name_to_previews: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for entry in queue:
        chunk_id = entry.get("chunk_id", "?")
        for name in entry.get("unknown_characters", []):
            if name not in existing_chars:
                # Store chunk_id as context reference; actual text loaded below
                name_to_previews[name].append((chunk_id, ""))

    if not name_to_previews:
        print("No unknown characters to suggest entries for.")
        return

    # Attempt to load text previews from raw sarga files
    raw_sargas_dir = Path(args.review_queue).parent.parent.parent / "raw_sargas"
    for name, entries in name_to_previews.items():
        updated: list[tuple[str, str]] = []
        for chunk_id, _ in entries[:3]:
            sarga_path = raw_sargas_dir / f"{chunk_id}.json"
            preview = ""
            if sarga_path.exists():
                try:
                    sarga = json.loads(sarga_path.read_text(encoding="utf-8"))
                    text = sarga.get("text", "")
                    # Extract 400-char window around first occurrence of name
                    idx = text.lower().find(name.lower())
                    if idx >= 0:
                        start = max(0, idx - 100)
                        preview = text[start: start + 400].strip()
                    else:
                        preview = text[:400].strip()
                except Exception:
                    pass
            updated.append((chunk_id, preview))
        name_to_previews[name] = updated

    client = anthropic.Anthropic()
    suggestions: dict[str, dict] = {}

    for name, preview_entries in name_to_previews.items():
        context_parts = [
            f"[{cid}]: {preview}"
            for cid, preview in preview_entries
            if preview
        ]
        context = "\n\n".join(context_parts) if context_parts else "(no text preview available)"

        prompt = (
            f"You are a Valmiki Ramayana scholar. Character '{name}' was found "
            f"in these passages:\n{context}\n\n"
            "Return ONLY JSON with these fields: canonical_id, canonical_name, "
            "gender, species, role, aliases, epithets, descriptive_references, "
            "primary_kandas, key_events, confidence.\n"
            "canonical_id must be lowercase letters and underscores only.\n"
            "No preamble, no markdown fences."
        )

        try:
            message = client.messages.create(
                model=MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text
            parsed = _parse_json_response(raw)

            if parsed is None:
                suggestions[name] = {"error": "JSON parse failed", "raw": raw[:500]}
                print(f"  [FAIL] {name} -- could not parse response")
                continue

            # Sanitise
            cid = str(parsed.get("canonical_id", "")).strip().lower()
            cid = re.sub(r"[^a-z_]", "_", cid)
            parsed["canonical_id"] = cid

            # Keep only valid kanda keys
            parsed["primary_kandas"] = [
                k for k in parsed.get("primary_kandas", [])
                if k in VALID_KANDA_KEYS
            ]

            conf = float(parsed.get("confidence", 0.0))
            suggestions[name] = parsed
            print(f"  [OK] {name} -> '{cid}'  (conf={conf:.2f})")

        except Exception as exc:
            suggestions[name] = {"error": str(exc)}
            print(f"  [ERROR] {name} -- {exc}")

    output_data = {
        "_instructions": (
            "Review each entry, remove incorrect ones, "
            "then run the merge command."
        ),
        "suggestions": suggestions,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSuggestions written to: {out_path}")


# ---------------------------------------------------------------------------
# Sub-command: merge
# ---------------------------------------------------------------------------

def cmd_merge(args: argparse.Namespace) -> None:
    """
    Merge approved suggestions into character_aliases.json.
    Writes to a temp file first, validates JSON, then renames atomically
    to prevent corruption on write failure.
    """
    approved_data = _load_json(args.approved)
    suggestions: dict = approved_data.get("suggestions", approved_data)

    aliases_path = Path(args.aliases)
    aliases_data = _load_json(aliases_path)
    existing_chars: dict = aliases_data.setdefault("characters", {})

    merged = 0
    for name, entry in suggestions.items():
        # Skip if error key present
        if "error" in entry:
            print(f"  [SKIP] '{name}' — has error: {entry['error']}")
            continue

        cid = str(entry.get("canonical_id", "")).strip()

        # Skip if empty canonical_id
        if not cid:
            print(f"  [SKIP] '{name}' — empty canonical_id")
            continue

        # Skip if already in aliases
        if cid in existing_chars:
            print(f"  [SKIP] '{cid}' — already in aliases")
            continue

        # Remove confidence field before saving
        clean_entry = {k: v for k, v in entry.items() if k != "confidence"}
        existing_chars[cid] = clean_entry
        merged += 1
        print(f"  [ADD]  '{cid}' — {entry.get('canonical_name', cid)}")

    total = len(existing_chars)

    # Write to temp file first, validate, then rename
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".json", dir=aliases_path.parent, prefix=".aliases_tmp_"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(aliases_data, f, ensure_ascii=False, indent=2)

        # Validate the temp file
        _load_json(tmp_path)

        # Atomic rename
        Path(tmp_path).replace(aliases_path)
        print(f"\nMerged {merged} new character(s). Total: {total}")

    except Exception as exc:
        # Clean up temp on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise RuntimeError(f"Merge failed, aliases.json NOT modified: {exc}") from exc


# ---------------------------------------------------------------------------
# Sub-command: rerun
# ---------------------------------------------------------------------------

def cmd_rerun(args: argparse.Namespace) -> None:
    """
    Re-run the LLM extraction only on sargas flagged in the review queue.
    Deletes existing extraction output for those sargas, copies raw files to
    a temp directory, runs BatchProcessor, and reports resolution counts.
    """
    from scripture_gpt.pipeline.batch_processor import BatchProcessor

    queue = _load_review_queue(args.review_queue)
    if not queue:
        print("Review queue is empty — nothing to rerun.")
        return

    flagged_ids: list[str] = [entry["chunk_id"] for entry in queue if "chunk_id" in entry]
    output_dir = Path(args.output)
    input_dir = Path(args.input)

    print(f"Flagged sargas to reprocess: {len(flagged_ids)}")

    # Delete existing extraction outputs for flagged sargas only
    deleted = 0
    for cid in flagged_ids:
        out_file = output_dir / f"{cid}.json"
        if out_file.exists():
            out_file.unlink()
            deleted += 1

    print(f"Deleted {deleted} existing extraction result(s).")

    # Copy only flagged raw sarga files to a temp directory
    with tempfile.TemporaryDirectory(prefix="scripture_gpt_rerun_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        copied = 0
        for cid in flagged_ids:
            src = input_dir / f"{cid}.json"
            if src.exists():
                shutil.copy2(src, tmp_path / f"{cid}.json")
                copied += 1
            else:
                log.warning("Raw sarga not found: %s", src)

        print(f"Copied {copied} raw sarga(s) to temp directory.")

        if copied == 0:
            print("No source files found — aborting rerun.")
            return

        # Run BatchProcessor on the temp directory
        processor = BatchProcessor(
            input_dir=tmp_path,
            output_dir=output_dir,
            aliases_path=args.aliases,
            dry_run=False,
        )
        processor.run()

    # Count resolved: sargas now with extraction_status == "success"
    resolved = 0
    for cid in flagged_ids:
        out_file = output_dir / f"{cid}.json"
        if out_file.exists():
            try:
                data = json.loads(out_file.read_text(encoding="utf-8"))
                if data.get("extraction_status") == "success":
                    resolved += 1
            except Exception:
                pass

    print(f"\nRerun complete: {copied} processed, {resolved} resolved.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alias_feedback.py",
        description=(
            "Maintain character_aliases.json using the extraction review queue. "
            "Four sub-commands: analyze, suggest, merge, rerun."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- analyze ---
    p_analyze = sub.add_parser(
        "analyze",
        help="Report on unknown characters found in the review queue",
        description=(
            "Read review_queue.json and print a report showing each unique "
            "unknown character, how many sargas it appeared in, the first 3 "
            "chunk_ids, and co-occurring unknowns."
        ),
    )
    p_analyze.add_argument(
        "--review-queue", required=True, metavar="PATH",
        help="Path to review_queue.json produced by batch_processor",
    )

    # --- suggest ---
    p_suggest = sub.add_parser(
        "suggest",
        help="Ask Claude Haiku to propose new character dictionary entries",
        description=(
            "For each unknown character not already in character_aliases.json, "
            "call Claude Haiku with context from the sarga text and save proposed "
            "entries to an output JSON file for human review."
        ),
    )
    p_suggest.add_argument(
        "--review-queue", required=True, metavar="PATH",
        help="Path to review_queue.json",
    )
    p_suggest.add_argument(
        "--aliases", required=True, metavar="PATH",
        help="Path to character_aliases.json",
    )
    p_suggest.add_argument(
        "--output", required=True, metavar="PATH",
        help="Path to write suggestions JSON file",
    )

    # --- merge ---
    p_merge = sub.add_parser(
        "merge",
        help="Merge approved suggestions into character_aliases.json",
        description=(
            "Read an approved suggestions file (human-reviewed output of 'suggest'), "
            "validate each entry, and merge new characters into character_aliases.json. "
            "Writes to a temp file first to prevent corruption on failure."
        ),
    )
    p_merge.add_argument(
        "--approved", required=True, metavar="PATH",
        help="Path to the approved suggestions JSON file",
    )
    p_merge.add_argument(
        "--aliases", required=True, metavar="PATH",
        help="Path to character_aliases.json to update",
    )

    # --- rerun ---
    p_rerun = sub.add_parser(
        "rerun",
        help="Re-run LLM extraction on sargas flagged in the review queue",
        description=(
            "Re-process only the sargas listed in review_queue.json: deletes their "
            "existing extraction results, copies raw sarga files to a temp directory, "
            "runs BatchProcessor, and reports how many were resolved."
        ),
    )
    p_rerun.add_argument(
        "--review-queue", required=True, metavar="PATH",
        help="Path to review_queue.json",
    )
    p_rerun.add_argument(
        "--input", required=True, metavar="PATH",
        help="Directory containing raw sarga JSON files",
    )
    p_rerun.add_argument(
        "--output", required=True, metavar="PATH",
        help="Directory containing extraction results (flagged files will be deleted and rewritten)",
    )
    p_rerun.add_argument(
        "--aliases", required=True, metavar="PATH",
        help="Path to character_aliases.json (updated before rerun for best results)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "analyze": cmd_analyze,
        "suggest": cmd_suggest,
        "merge": cmd_merge,
        "rerun": cmd_rerun,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
