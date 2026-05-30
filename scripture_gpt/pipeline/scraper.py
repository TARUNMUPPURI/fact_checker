"""
scraper.py
----------
Valmiki Ramayana scraper — downloads all 645 sargas from
valmikiramayan.net and saves structured JSON to raw_sargas/.

Navigation strategy (the site uses HTML framesets):
  1. Fetch kanda contents page  →  collect _frame.htm links
  2. Fetch each _frame.htm      →  read <frame src="..."> for content URL
  3. Fetch content page         →  parse Devanagari / IAST / English

URL shape discovered from live site:
  Contents : https://www.valmikiramayan.net/utf8/{kanda}/xxx_contents.htm
  Frame    : https://www.valmikiramayan.net/utf8/{kanda}/sarga{N}/{kanda}_{N}_frame.htm
  Content  : https://www.valmikiramayan.net/utf8/{kanda}/sarga{N}/{kanda}sans{N}.htm
             (relative to frame page; src attr in <frame> tag)

Usage:
    python scripture_gpt/pipeline/scraper.py --kanda sundara_kanda
    python scripture_gpt/pipeline/scraper.py --all
    python scripture_gpt/pipeline/scraper.py --kanda bala_kanda --delay 2.0 --output mydir/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Path bootstrap — importable from any cwd
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripture_gpt.data.chunking_schema import KANDA_REGISTRY

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://www.valmikiramayan.net"
USER_AGENT = "ScriptureGPT-Research/1.0 (educational project)"

# IAST diacritic characters used on this site
IAST_CHARS = set("\u0101\u012b\u016b\u1e43\u015b\u1e63\u1e6d\u1e0d\u1e47"
                 "\u0100\u012a\u016a\u015a\u1e62\u1e6c\u1e0c\u1e46\u1e5b\u1e5a")

# Shloka reference patterns to strip from extracted text
_REF_RE = re.compile(
    r"\[\s*\d+\s*[-\u2013]\s*\d+\s*[-\u2013]\s*\d+\s*\]"  # [1-14-3]
    r"|\(\s*\d+\s*\)"                                         # (1)
    r"|\bverse\s+\d+\b"                                       # verse 14
    r"|\bshloka\s+\d+\b",                                     # shloka 3
    flags=re.IGNORECASE,
)

# Global sarga number offsets
SARGA_OFFSETS: dict[str, int] = {
    "bala_kanda": 0,
    "ayodhya_kanda": 77,
    "aranya_kanda": 196,
    "kishkindha_kanda": 271,
    "sundara_kanda": 338,
    "yuddha_kanda": 406,
    "uttara_kanda": 534,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scraper")

# ---------------------------------------------------------------------------
# Text classification helpers
# ---------------------------------------------------------------------------

def _is_devanagari(text: str) -> bool:
    """True if text contains any Devanagari codepoint (U+0900-U+097F)."""
    return any("\u0900" <= ch <= "\u097F" for ch in text)


def _is_iast(text: str) -> bool:
    """True if text contains IAST diacritics."""
    return bool(IAST_CHARS & set(text))


def _is_pratipada(text: str) -> bool:
    """True if text appears to be a word-by-word meaning (e.g. word = meaning)."""
    return text.count(" = ") >= 1 or (text.count("=") >= 2)


def _clean(text: str) -> str:
    """Normalise whitespace and strip shloka reference markers."""
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = _REF_RE.sub("", text)
    # Remove 'Verse Locator' and other common site artifacts
    text = re.sub(r"\bVerse Locator\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def _fetch(url: str, session: requests.Session) -> Optional[BeautifulSoup]:
    """GET url; return BeautifulSoup or None on 404 / network error."""
    try:
        resp = session.get(url, timeout=20)
        if resp.status_code == 404:
            log.warning("404 Not Found: %s -- skipping", url)
            return None
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as exc:
        log.warning("Request error %s: %s -- skipping", url, exc)
        return None


# ---------------------------------------------------------------------------
# Step 1 — collect frame URLs from the kanda contents page
# ---------------------------------------------------------------------------

def get_frame_urls(kanda_key: str, session: requests.Session) -> list[str]:
    """
    Fetch the kanda contents page and return all sarga frame URLs
    (links containing '_frame.htm' that also contain 'sarga').
    """
    contents_path = KANDA_REGISTRY[kanda_key]["contents_path"]
    contents_url = f"{BASE_URL}/{contents_path}"
    log.info("Fetching contents page: %s", contents_url)

    soup = _fetch(contents_url, session)
    if soup is None:
        raise RuntimeError(f"Could not fetch contents page: {contents_url}")

    seen: set[str] = set()
    ordered: list[str] = []
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        if "_frame.htm" in href and "sarga" in href:
            full = urljoin(contents_url, href)
            if full not in seen:
                seen.add(full)
                ordered.append(full)

    log.info("Found %d frame links for %s", len(ordered), kanda_key)
    return ordered


# ---------------------------------------------------------------------------
# Step 2 — resolve frame page to content URL
# ---------------------------------------------------------------------------

def resolve_content_url(frame_url: str, session: requests.Session) -> Optional[str]:
    """
    Fetch the frame page and return the URL of the main content frame
    (the <frame name='main'> or first <frame> src).
    """
    soup = _fetch(frame_url, session)
    if soup is None:
        return None

    # HTML 4 frameset: <frame src="...">
    for frame in soup.find_all("frame"):
        src = frame.get("src", "")
        if src and "footer" not in src.lower():
            return urljoin(frame_url, src)

    # Fallback: <iframe>
    for iframe in soup.find_all("iframe"):
        src = iframe.get("src", "")
        if src:
            return urljoin(frame_url, src)

    log.warning("No content frame found in: %s", frame_url)
    return None


# ---------------------------------------------------------------------------
# Step 3 — parse the content page
# ---------------------------------------------------------------------------

def parse_content(soup: BeautifulSoup) -> dict:
    """
    Extract Devanagari, IAST, and English blocks from a sarga content page.

    Returns:
        {
            "shlokas": list of {shloka_number, sanskrit_devanagari, iast, english},
            "text": str,               # joined English
            "text_sanskrit_full": str, # joined Devanagari
        }
    """
    devanagari_blocks: list[str] = []
    english_blocks: list[str] = []

    shlokas = []
    current_shloka = None

    for p in soup.find_all("p"):
        raw = p.get_text(separator=" ")
        cleaned = _clean(raw)
        if not cleaned:
            continue

        if _is_devanagari(cleaned):
            if current_shloka:
                shlokas.append(current_shloka)
            
            current_shloka = {
                "sanskrit_devanagari": cleaned,
                "iast": "",
                "pratipada_pieces": [],
                "english_pieces": []
            }
            devanagari_blocks.append(cleaned)
            
        elif _is_iast(cleaned):
            if current_shloka:
                if current_shloka["iast"]:
                    current_shloka["iast"] += "\n" + cleaned
                else:
                    current_shloka["iast"] = cleaned
                    
        elif len(cleaned) > 10:
            if _is_pratipada(cleaned):
                if current_shloka:
                    current_shloka["pratipada_pieces"].append(cleaned)
            else:
                english_blocks.append(cleaned)
                if current_shloka:
                    current_shloka["english_pieces"].append(cleaned)

    if current_shloka:
        shlokas.append(current_shloka)

    # Format the shlokas
    final_shlokas = []
    for i, sh in enumerate(shlokas):
        final_shlokas.append({
            "shloka_number": i + 1,
            "sanskrit_devanagari": sh["sanskrit_devanagari"],
            "iast": sh["iast"],
            "pratipada": "\n\n".join(sh["pratipada_pieces"]),
            "english": "\n\n".join(sh["english_pieces"])
        })

    text = "\n\n".join(english_blocks)
    text_sanskrit_full = "\n\n".join(devanagari_blocks)

    # Fallback: never store an empty text field
    if not text.strip():
        fallback = _clean(soup.get_text(separator=" "))
        if fallback:
            log.warning("No English prose found — using raw body text as fallback")
            text = fallback

    return {
        "shlokas": final_shlokas,
        "text": text,
        "text_sanskrit_full": text_sanskrit_full,
    }


# ---------------------------------------------------------------------------
# Sarga number extraction
# ---------------------------------------------------------------------------

_SARGA_RE = re.compile(r"sarga(\d+)", re.IGNORECASE)


def _sarga_number_from_url(url: str) -> Optional[int]:
    m = _SARGA_RE.search(url)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Core: scrape one kanda
# ---------------------------------------------------------------------------

def scrape_kanda(
    kanda_key: str,
    output_dir: Path,
    session: requests.Session,
    delay: float = 1.5,
    force: bool = False,
) -> int:
    """Scrape all sargas for one kanda. Returns count of newly saved files."""
    meta = KANDA_REGISTRY[kanda_key]
    sarga_count: int = meta["sarga_count"]
    kanda_english: str = meta["english_name"]
    offset: int = SARGA_OFFSETS[kanda_key]
    saved = 0

    # Step 1: collect all frame URLs from the contents page
    frame_urls = get_frame_urls(kanda_key, session)
    total_found = len(frame_urls)

    if total_found == 0:
        log.error("No frame URLs found for %s — aborting kanda", kanda_key)
        return 0

    if total_found != sarga_count:
        log.warning(
            "%s: expected %d sargas from registry but found %d frame links",
            kanda_key, sarga_count, total_found,
        )

    for idx, frame_url in enumerate(frame_urls, start=1):
        sarga_number = _sarga_number_from_url(frame_url)
        if sarga_number is None:
            log.warning("Could not extract sarga number from: %s", frame_url)
            continue

        chunk_id = f"{kanda_key}_{sarga_number:04d}"
        out_path = output_dir / f"{chunk_id}.json"

        # Resume support
        if out_path.exists() and not force:
            log.info("[%d/%d] %s  (exists -- skipping)", idx, total_found, chunk_id)
            continue

        # Step 2: resolve frame → content URL
        time.sleep(delay)
        content_url = resolve_content_url(frame_url, session)
        if content_url is None:
            log.warning("[%d/%d] %s  [SKIP]  (no content URL)", idx, total_found, chunk_id)
            continue

        # Step 3: fetch and parse content page
        time.sleep(delay)
        content_soup = _fetch(content_url, session)
        if content_soup is None:
            log.warning("[%d/%d] %s  [SKIP]  (content fetch failed)", idx, total_found, chunk_id)
            continue

        parsed = parse_content(content_soup)

        chunk = {
            "chunk_id": chunk_id,
            "kanda": kanda_key,
            "kanda_english": kanda_english,
            "sarga_number": sarga_number,
            "sarga_number_global": offset + sarga_number,
            "shlokas": parsed["shlokas"],
            "text": parsed["text"],
            "text_sanskrit_full": parsed["text_sanskrit_full"],
            "source_url": content_url,
            "scraped_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        out_path.write_text(
            json.dumps(chunk, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        saved += 1
        char_count = len(chunk["text"])
        print(f"  [{idx}/{total_found}] {chunk_id}  [OK]  {char_count} chars", flush=True)

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Valmiki Ramayana from valmikiramayan.net",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--kanda",
        choices=list(KANDA_REGISTRY.keys()),
        metavar="KANDA_KEY",
        help="Scrape one kanda (e.g. sundara_kanda)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Scrape all 7 kandas",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripture_gpt/raw_sargas"),
        help="Output directory (default: scripture_gpt/raw_sargas/)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds between requests (default: 1.5)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files instead of skipping them",
    )
    args = parser.parse_args(argv)

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    session = _make_session()
    kandas = list(KANDA_REGISTRY.keys()) if args.all else [args.kanda]

    total_saved = 0
    for kanda_key in kandas:
        meta = KANDA_REGISTRY[kanda_key]
        print(
            f"\n{'='*60}\n"
            f"  Kanda {meta['number']}/7 -- {meta['english_name']} ({kanda_key})\n"
            f"  Sargas: {meta['sarga_count']}  |  Delay: {args.delay}s\n"
            f"{'='*60}"
        )
        n = scrape_kanda(kanda_key, output_dir, session, delay=args.delay, force=args.force)
        total_saved += n
        print(f"\n  -> {n} new file(s) saved for {kanda_key}")

    print(f"\nDone. Total new sargas saved: {total_saved}")


if __name__ == "__main__":
    main()
