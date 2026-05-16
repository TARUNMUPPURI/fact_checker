"""
pipeline/scraper.py

Robust scraper for valmikiramayan.net
- No hardcoded sarga URLs
- Navigates via kanda → sarga → frame → actual content
- Extracts ONLY English translation (class="tat")
- Outputs one JSON per sarga (your schema)

Usage:
  python pipeline/scraper.py --kanda sundara_kanda --output data/raw_sargas/
"""

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.valmikiramayan.net"

KANDA_CONFIG = {
    "bala_kanda": {
        "english": "Book of Youth",
        "path": "utf8/baala/baala_contents.htm",
    },
    "ayodhya_kanda": {
        "english": "Book of Ayodhya",
        "path": "utf8/ayodhya/ayodhya_contents.htm",
    },
    "aranya_kanda": {
        "english": "Book of the Forest",
        "path": "utf8/aranya/aranya_contents.htm",
    },
    "kishkindha_kanda": {
        "english": "Book of Kishkindha",
        "path": "utf8/kish/kishkindha_contents.htm",
    },
    "sundara_kanda": {
        "english": "Book of Beauty",
        "path": "utf8/sundara/sundara_contents.htm",
    },
    "yuddha_kanda": {
        "english": "Book of War",
        "path": "utf8/yuddha/yuddha_contents.htm",
    },
}

GLOBAL_SARGA_OFFSETS = {
    "bala_kanda": 0,
    "ayodhya_kanda": 77,
    "aranya_kanda": 196,
    "kishkindha_kanda": 271,
    "sundara_kanda": 338,
    "yuddha_kanda": 406,
}


class ValmikiScraper:

    def __init__(self, output_dir: str, delay: float = 1.5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ScriptureGPT-Research/1.0 (educational use)"
        })

    # ------------------------------------------------------------------ #
    # HTTP
    # ------------------------------------------------------------------ #

    def _get_soup(self, url: str) -> BeautifulSoup:
        for attempt in range(3):
            try:
                resp = self.session.get(url, timeout=15)
                resp.raise_for_status()
                return BeautifulSoup(resp.text, "html.parser")
            except requests.RequestException as e:
                print(f"[Retry {attempt+1}] {url} → {e}")
                time.sleep(2 * (attempt + 1))
        raise Exception(f"Failed to fetch: {url}")

    # ------------------------------------------------------------------ #
    # Navigation
    # ------------------------------------------------------------------ #

    def get_sarga_frame_links(self, kanda_url: str):
        soup = self._get_soup(kanda_url)

        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]

            if "_frame.htm" in href and "sarga" in href:
                full = urljoin(kanda_url, href)
                links.append(full)

        return links

    def resolve_frame(self, frame_url: str):
        soup = self._get_soup(frame_url)

        frame = soup.find("frame")
        if frame and frame.get("src"):
            return urljoin(frame_url, frame["src"])

        iframe = soup.find("iframe")
        if iframe and iframe.get("src"):
            return urljoin(frame_url, iframe["src"])

        print(f"[WARN] Could not resolve frame: {frame_url}")
        return None

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #

    def extract_translation(self, url: str):
        soup = self._get_soup(url)

        paragraphs = []

        for p in soup.find_all("p", class_="tat"):
            text = p.get_text(" ", strip=True)

            if not text:
                continue

            # clean artifacts
            text = re.sub(r"\[\d+[-\d]*\]", "", text)
            text = re.sub(r"\(\d+\)", "", text)
            text = re.sub(r"\s+", " ", text).strip()

            if len(text) > 40:
                paragraphs.append(text)

        return "\n\n".join(paragraphs)

    def extract_sarga_number(self, url: str):
        match = re.search(r"sarga(\d+)", url)
        return int(match.group(1)) if match else None

    # ------------------------------------------------------------------ #
    # Main
    # ------------------------------------------------------------------ #

    def scrape_kanda(self, kanda_key: str):
        config = KANDA_CONFIG[kanda_key]

        kanda_url = urljoin(BASE_URL, config["path"])
        kanda_english = config["english"]

        print(f"\nScraping {kanda_english}...")

        frame_links = self.get_sarga_frame_links(kanda_url)

        for frame_url in frame_links:

            try:
                content_url = self.resolve_frame(frame_url)
                if not content_url:
                    continue

                text = self.extract_translation(content_url)
                if not text or len(text) < 200:
                    print(f"[SKIP] Low content: {content_url}")
                    continue

                sarga_number = self.extract_sarga_number(frame_url)
                if not sarga_number:
                    print(f"[SKIP] No sarga number: {frame_url}")
                    continue

                chunk_id = f"{kanda_key}_{sarga_number:04d}"
                global_num = GLOBAL_SARGA_OFFSETS[kanda_key] + sarga_number

                data = {
                    "chunk_id": chunk_id,
                    "kanda": kanda_key,
                    "kanda_english": kanda_english,
                    "sarga_number": sarga_number,
                    "sarga_number_global": global_num,
                    "text": text,
                    "source_url": content_url,
                    "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }

                out_path = self.output_dir / f"{chunk_id}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                print(f"✓ {chunk_id}")

                time.sleep(self.delay)

            except Exception as e:
                print(f"[ERROR] {frame_url} → {e}")

    # ------------------------------------------------------------------ #
    # CLI
    # ------------------------------------------------------------------ #


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--kanda", choices=list(KANDA_CONFIG.keys()))
    parser.add_argument("--delay", type=float, default=1.5)

    args = parser.parse_args()

    scraper = ValmikiScraper(output_dir=args.output, delay=args.delay)

    if not args.kanda:
        parser.error("Provide --kanda")

    scraper.scrape_kanda(args.kanda)