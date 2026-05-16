$py = "C:\Users\tarun\AppData\Local\Programs\Python\Python312\python.exe"

# Step 1
Write-Host "STEP 1 - Install dependencies"
& $py -c "import anthropic, langgraph, chromadb, neo4j, fastapi, streamlit; print('All imports OK')"
$importResult = $?
if (-not $importResult) { Write-Host "STEP 1 FAIL: Imports failed"; exit 1 }
& $py -c "import langchain" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "STEP 1 FAIL: langchain is installed"
    exit 1
}
Write-Host "STEP 1 PASS"

# Step 2
Write-Host "STEP 2 - Config"
if (-not (Test-Path ".env")) {
    Copy-Item "scripture_gpt/.env.example" ".env" -ErrorAction Ignore
    Copy-Item ".env.example" ".env" -ErrorAction Ignore
}
Write-Host "STEP 2 PASS"

# Step 3
Write-Host "STEP 3 - Scrape"
& $py scripture_gpt/pipeline/scraper.py --kanda sundara_kanda --output scripture_gpt/raw_sargas/ --delay 0.1
& $py -c "import glob; f = glob.glob('scripture_gpt/raw_sargas/sundara_kanda_*.json'); assert len(f) >= 68, f'Got {len(f)}'; print('Scraper OK: >= 68 sargas')"
if ($LASTEXITCODE -ne 0) { Write-Host "STEP 3 FAIL"; exit 1 }
Write-Host "STEP 3 PASS"

# Step 4
Write-Host "STEP 4 - Batch extraction"
& $py scripture_gpt/pipeline/batch_processor.py --input scripture_gpt/raw_sargas --output scripture_gpt/output/extracted --aliases scripture_gpt/data/character_aliases.json --dry-run
if ($LASTEXITCODE -ne 0) { Write-Host "STEP 4 FAIL"; exit 1 }
Write-Host "STEP 4 PASS"

# Step 5
Write-Host "STEP 5 - Graph"
& $py -c "from scripture_gpt.pipeline.query_pipeline import app, QueryState; nodes = set(app.nodes.keys()) - {'__start__'}; expected = {'classify_intent','direct_lookup','call_tools','summarise_evidence','reason','verify_citations','format_output'}; assert not expected - nodes, f'Missing: {expected - nodes}'; print('LangGraph graph OK')"
if ($LASTEXITCODE -ne 0) { Write-Host "STEP 5 FAIL"; exit 1 }
Write-Host "STEP 5 PASS"

# Step 6
Write-Host "STEP 6 - FastAPI Routes"
& $py -c "from scripture_gpt.api.main import app; routes = [r.path for r in app.routes];`nfor r in ['/query','/factcheck','/entity/{canonical_id}','/health']: assert r in routes;`nprint('FastAPI routes OK')"
if ($LASTEXITCODE -ne 0) { Write-Host "STEP 6 FAIL"; exit 1 }
Write-Host "STEP 6 PASS"

# Step 7
Write-Host "STEP 7 - UI checks"
& $py -c "import ast; src = open('scripture_gpt/ui/app.py', encoding='utf-8').read(); ast.parse(src);`nfor check in ['st.set_page_config','INSUFFICIENT_EVIDENCE','unsafe_allow_html']: assert check in src, f'Missing: {check}';`nassert 'langchain' not in src.lower(); assert 'langgraph' not in src.lower(); print('Streamlit UI OK')"
if ($LASTEXITCODE -ne 0) { Write-Host "STEP 7 FAIL"; exit 1 }
Write-Host "STEP 7 PASS"

Write-Host "STEPS 1-7 COMPLETE"
