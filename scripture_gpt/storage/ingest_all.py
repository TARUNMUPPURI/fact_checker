"""
ingest_all.py
-------------
CLI script that runs ChromaIngestor then Neo4jIngestor in sequence.
Exits with code 1 if either step fails.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest extracted sarga data into ChromaDB and Neo4j",
    )
    parser.add_argument(
        "--extracted-dir",
        default="scripture_gpt/output/extracted",
        help="Directory containing extraction result JSONs (default: scripture_gpt/output/extracted)",
    )
    parser.add_argument(
        "--raw-sargas-dir",
        default="scripture_gpt/raw_sargas",
        help="Directory containing raw sarga JSONs (default: scripture_gpt/raw_sargas)",
    )
    parser.add_argument(
        "--aliases",
        default="scripture_gpt/data/character_aliases.json",
        help="Path to character_aliases.json",
    )
    parser.add_argument(
        "--chroma-dir",
        default=os.getenv("CHROMA_PERSIST_DIR", "scripture_gpt/.chroma"),
        help="ChromaDB persistence directory (default: $CHROMA_PERSIST_DIR)",
    )
    parser.add_argument(
        "--neo4j-uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j URI (default: $NEO4J_URI)",
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.getenv("NEO4J_USER", "neo4j"),
        help="Neo4j username (default: $NEO4J_USER)",
    )
    parser.add_argument(
        "--neo4j-password",
        default=os.getenv("NEO4J_PASSWORD", ""),
        help="Neo4j password (default: $NEO4J_PASSWORD)",
    )
    parser.add_argument(
        "--skip-chroma", action="store_true",
        help="Skip ChromaDB ingestion",
    )
    parser.add_argument(
        "--skip-neo4j", action="store_true",
        help="Skip Neo4j ingestion",
    )
    args = parser.parse_args()

    exit_code = 0

    # ------------------------------------------------------------------ #
    # Step 1: ChromaDB
    # ------------------------------------------------------------------ #
    if not args.skip_chroma:
        print("\n" + "="*60)
        print("  STEP 1/2 — ChromaDB Ingestion")
        print("="*60)
        try:
            from scripture_gpt.storage.chroma_ingest import ChromaIngestor
            ingestor = ChromaIngestor(persist_dir=args.chroma_dir)
            ingestor.ingest_all(
                extracted_dir=args.extracted_dir,
                raw_sargas_dir=args.raw_sargas_dir,
            )
        except ConnectionError as exc:
            print(f"\n[ERROR] ChromaDB connection failed: {exc}", file=sys.stderr)
            exit_code = 1
        except Exception as exc:
            print(f"\n[ERROR] ChromaDB ingestion failed: {exc}", file=sys.stderr)
            exit_code = 1
    else:
        print("Skipping ChromaDB (--skip-chroma)")

    # ------------------------------------------------------------------ #
    # Step 2: Neo4j
    # ------------------------------------------------------------------ #
    if not args.skip_neo4j:
        print("\n" + "="*60)
        print("  STEP 2/2 — Neo4j Ingestion")
        print("="*60)
        try:
            from scripture_gpt.storage.neo4j_ingest import Neo4jIngestor
            ingestor = Neo4jIngestor(
                uri=args.neo4j_uri,
                user=args.neo4j_user,
                password=args.neo4j_password,
            )
            try:
                ingestor.ingest_all(
                    extracted_dir=args.extracted_dir,
                    aliases_path=args.aliases,
                )
            finally:
                ingestor.close()
        except ConnectionError as exc:
            print(f"\n[ERROR] Neo4j connection failed: {exc}", file=sys.stderr)
            exit_code = 1
        except Exception as exc:
            print(f"\n[ERROR] Neo4j ingestion failed: {exc}", file=sys.stderr)
            exit_code = 1
    else:
        print("Skipping Neo4j (--skip-neo4j)")

    print("\n" + "="*60)
    if exit_code == 0:
        print("  Ingest complete — all steps succeeded.")
    else:
        print("  Ingest finished with errors (see above).")
    print("="*60 + "\n")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
