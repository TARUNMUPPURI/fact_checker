import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

kandas = [
    "bala_kanda",
    "ayodhya_kanda",
    "aranya_kanda",
    "kishkindha_kanda",
    "sundara_kanda",
    "yuddha_kanda",
    "uttara_kanda"
]

def scrape_kanda(kanda):
    print(f"Starting {kanda}...")
    cmd = [
        "python", "scripture_gpt/pipeline/scraper.py",
        "--kanda", kanda,
        "--output", "scripture_gpt/raw_sargas/",
        "--delay", "0.2"
    ]
    subprocess.run(cmd)
    print(f"Finished {kanda}")

with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(scrape_kanda, kandas)
