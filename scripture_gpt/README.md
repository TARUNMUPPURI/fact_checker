# Scripture GPT V1 — Valmiki Ramayana Fact-Checker

Scripture GPT is an AI-powered fact-checking system for Hindu scriptures. Version 1 focuses exclusively on the Valmiki Ramayana: given a natural-language claim (e.g., *"Hanuman never disobeyed Rama"*), the system retrieves semantically relevant evidence from all seven Kandas using a hybrid ChromaDB vector store and Neo4j knowledge graph, then passes the evidence to Claude (Anthropic) for a structured verdict complete with Sanskrit shloka citations, kanda/sarga/shloka references, and a confidence score — with an optional GPT-4o mini verification pass for cross-model consensus.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Fill in your API keys and connection strings in .env

# 3. (Coming soon) Run the ingestion pipeline, then start the API / UI
```

## Project Layout

| Path | Purpose |
|------|---------|
| `data/` | Character alias maps, chunking schema |
| `pipeline/` | Scraper, extractor, batch processor |
| `storage/` | ChromaDB & Neo4j ingestion scripts |
| `api/` | FastAPI application |
| `ui/` | Streamlit front-end |
| `raw_sargas/` | Scraped sarga JSON files |
| `output/extracted/` | Extraction results |
