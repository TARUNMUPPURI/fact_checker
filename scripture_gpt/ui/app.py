"""
scripture_gpt/ui/app.py
-----------------------
Streamlit front-end for Scripture GPT V1.
All data sourced via HTTP calls to the FastAPI backend.
"""

import os

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 60

VERDICT_COLORS = {
    "TRUE":                  "#2e7d32",   # green
    "FALSE":                 "#c62828",   # red
    "PARTIALLY_TRUE":        "#e65100",   # amber
    "INSUFFICIENT_EVIDENCE": "#546e7a",   # grey
}

VERDICT_LABELS = {
    "TRUE":                  "TRUE",
    "FALSE":                 "FALSE",
    "PARTIALLY_TRUE":        "PARTIALLY TRUE",
    "INSUFFICIENT_EVIDENCE": "INSUFFICIENT EVIDENCE",
}

KANDA_COLORS = {
    "bala_kanda":        "#1565C0",
    "ayodhya_kanda":     "#6A1B9A",
    "aranya_kanda":      "#2E7D32",
    "kishkindha_kanda":  "#E65100",
    "sundara_kanda":     "#00838F",
    "yuddha_kanda":      "#B71C1C",
    "uttara_kanda":      "#4E342E",
}

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Scripture GPT",
    layout="wide",
    page_icon="🕉️",
)

# ---------------------------------------------------------------------------
# Shared CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  .verdict-card {
    padding: 18px 24px;
    border-radius: 10px;
    margin-bottom: 16px;
    color: white;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 1px;
  }
  .devanagari-box {
    background: #f5f5f5;
    border-left: 4px solid #90A4AE;
    padding: 10px 14px;
    border-radius: 4px;
    font-size: 1.1rem;
    line-height: 1.8;
    color: #263238;
    margin-bottom: 6px;
  }
  .kanda-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    color: white;
    font-size: 0.8rem;
    margin: 2px 4px 2px 0;
    font-weight: 600;
  }
  .epithet-tag {
    display: inline-block;
    background: #E3F2FD;
    color: #1565C0;
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 0.82rem;
    margin: 2px 3px 2px 0;
  }
  .footnote {
    color: #9E9E9E;
    font-size: 0.78rem;
    margin-top: 12px;
  }
  .citation-header {
    font-weight: 600;
    color: #37474F;
    border-bottom: 1px solid #ECEFF1;
    padding-bottom: 4px;
    margin-bottom: 8px;
  }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

for key in ("factcheck_result", "explore_result", "entity_result"):
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _has_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in text)


def _safe_markdown(text: str) -> None:
    """Render text; use unsafe_allow_html only when Devanagari is present."""
    if text and _has_devanagari(text):
        st.markdown(text, unsafe_allow_html=True)
    else:
        st.markdown(text)


def api_post(path: str, payload: dict) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE_URL}{path}",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 500:
            st.error(
                "The fact-checker encountered an error. "
                "Please try rephrasing your query."
            )
            return None
        if resp.status_code == 404:
            st.error(f"Not found (HTTP 404).")
            return None
        if not resp.ok:
            st.error(f"API error {resp.status_code}: {resp.text[:200]}")
            return None
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not connect to the backend at {API_BASE_URL}. "
            "Make sure the FastAPI server is running."
        )
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out after 60 seconds. Please try again.")
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        return None


def api_get(path: str) -> dict | None:
    try:
        resp = requests.get(
            f"{API_BASE_URL}{path}",
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 404:
            st.error(f"Character not found. Check the name or canonical ID.")
            return None
        if resp.status_code == 500:
            st.error(
                "The fact-checker encountered an error. "
                "Please try rephrasing your query."
            )
            return None
        if not resp.ok:
            st.error(f"API error {resp.status_code}: {resp.text[:200]}")
            return None
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not connect to the backend at {API_BASE_URL}. "
            "Make sure the FastAPI server is running."
        )
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Shared citation renderer
# ---------------------------------------------------------------------------

def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    st.markdown("#### 📜 Citations")
    for i, cite in enumerate(citations, start=1):
        kanda   = cite.get("kanda", "").replace("_", " ").title()
        cid     = cite.get("chunk_id", "")
        shloka  = cite.get("shloka_number")
        skt     = cite.get("sanskrit_devanagari", "")
        english = cite.get("english", "")
        detail  = cite.get("detail", "")

        shloka_label = f"Shloka {shloka}" if shloka else ""
        header = " · ".join(filter(None, [kanda, cid, shloka_label]))

        with st.expander(f"Citation {i} — {header}", expanded=(i == 1)):
            st.markdown(f'<div class="citation-header">{header}</div>',
                        unsafe_allow_html=True)
            if skt and _has_devanagari(skt):
                st.markdown(
                    f'<div class="devanagari-box">{skt}</div>',
                    unsafe_allow_html=True,
                )
            if english:
                st.markdown(english)
            if detail:
                st.markdown(f"*{detail}*")


# ---------------------------------------------------------------------------
# Page 1 — Fact Checker
# ---------------------------------------------------------------------------

def page_factcheck() -> None:
    st.title("🕉️ Scripture GPT — Fact Checker")
    st.caption("Verify claims against the complete Valmiki Ramayana (645 sargas)")

    claim = st.text_input(
        "Enter a claim to verify",
        placeholder="Hanuman never acted without Rama's permission",
        key="factcheck_input",
    )
    submit = st.button("Check Claim", type="primary", key="factcheck_submit")

    if submit and claim.strip():
        with st.spinner("Consulting the Valmiki Ramayana…"):
            data = api_post("/factcheck", {"claim": claim.strip()})
        if data:
            st.session_state.factcheck_result = data

    result = st.session_state.factcheck_result
    if result is None:
        return

    verdict  = result.get("verdict", "INSUFFICIENT_EVIDENCE")
    conf     = float(result.get("verdict_confidence", 0.0))
    expl     = result.get("explanation", "")
    sup      = result.get("supporting_points", [])
    con      = result.get("contradicting_points", [])
    citations = result.get("citations", [])
    cost     = result.get("total_cost_usd", 0)
    latency  = result.get("latency_ms", 0)

    color = VERDICT_COLORS.get(verdict, VERDICT_COLORS["INSUFFICIENT_EVIDENCE"])
    label = VERDICT_LABELS.get(verdict, verdict)

    st.markdown(
        f'<div class="verdict-card" style="background:{color};">'
        f'VERDICT: {label}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"**Confidence:** {conf:.0%}")
    st.progress(min(max(conf, 0.0), 1.0))

    if expl:
        st.markdown("#### Explanation")
        st.markdown(expl)

    col1, col2 = st.columns(2)
    with col1:
        if sup:
            st.markdown("#### ✅ Supporting Points")
            for pt in sup:
                st.markdown(f'<span style="color:#2e7d32">• {pt}</span>',
                            unsafe_allow_html=True)
    with col2:
        if con:
            st.markdown("#### ❌ Contradicting Points")
            for pt in con:
                st.markdown(f'<span style="color:#c62828">• {pt}</span>',
                            unsafe_allow_html=True)

    render_citations(citations)

    st.markdown(
        f'<div class="footnote">Cost: ${cost:.4f} | Latency: {latency} ms</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page 2 — Explore
# ---------------------------------------------------------------------------

def page_explore() -> None:
    st.title("🔍 Explore the Ramayana")
    st.caption("Ask any question about characters, events, or themes")

    query = st.text_input(
        "Your question",
        placeholder="Ask anything about the Ramayana…",
        key="explore_input",
    )
    submit = st.button("Search", type="primary", key="explore_submit")

    if submit and query.strip():
        with st.spinner("Searching the scriptures…"):
            data = api_post("/query", {"query": query.strip()})
        if data:
            st.session_state.explore_result = data

    result = st.session_state.explore_result
    if result is None:
        return

    answer    = result.get("answer", "")
    citations = result.get("citations", [])
    cost      = result.get("total_cost_usd", 0)
    latency   = result.get("latency_ms", 0)
    mode      = result.get("mode", "explore")
    complexity = result.get("complexity", "")

    st.markdown(f"*Mode: {mode.upper()} · Complexity: {complexity}*")
    st.markdown("---")

    if answer:
        _safe_markdown(answer)

    render_citations(citations)

    st.markdown(
        f'<div class="footnote">Cost: ${cost:.4f} | Latency: {latency} ms</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page 3 — Character Explorer
# ---------------------------------------------------------------------------

def page_character() -> None:
    st.title("👤 Character Explorer")
    st.caption("Explore characters by name or epithet")

    name = st.text_input(
        "Character name or epithet",
        placeholder="Enter name or epithet (e.g. Anjaneya, Maruti)",
        key="entity_input",
    )
    submit = st.button("Look Up", type="primary", key="entity_submit")

    if submit and name.strip():
        # Use name directly as canonical_id (lowercase + underscores)
        canonical_id = name.strip().lower().replace(" ", "_")
        with st.spinner(f"Looking up '{name}'…"):
            data = api_get(f"/entity/{canonical_id}")
        if data:
            st.session_state.entity_result = data

    result = st.session_state.entity_result
    if result is None:
        return

    cname   = result.get("canonical_name", result.get("canonical_id", ""))
    epithets = result.get("epithets", [])
    kandas  = result.get("primary_kandas", [])
    events  = result.get("key_events", [])
    n_sargas = result.get("appears_in_sargas", 0)

    st.markdown(f"## {cname}")

    if epithets:
        tags_html = "".join(
            f'<span class="epithet-tag">{e}</span>' for e in epithets
        )
        st.markdown(tags_html, unsafe_allow_html=True)

    st.markdown(f"**Appears in {n_sargas} sarga(s)**")
    st.markdown("---")

    if kandas:
        st.markdown("**Primary Kandas**")
        badges = ""
        for k in kandas:
            color = KANDA_COLORS.get(k, "#455A64")
            label = k.replace("_", " ").title()
            badges += (
                f'<span class="kanda-badge" '
                f'style="background:{color};">{label}</span>'
            )
        st.markdown(badges, unsafe_allow_html=True)
        st.markdown("")

    if events:
        st.markdown("**Key Events**")
        for i, ev in enumerate(events, start=1):
            st.markdown(f"{i}. {ev}")


# ---------------------------------------------------------------------------
# Page 4 — About
# ---------------------------------------------------------------------------

def page_about() -> None:
    st.title("ℹ️ About Scripture GPT")

    st.markdown("""
Scripture GPT is an AI-powered fact-checker and knowledge explorer for the
**Valmiki Ramayana** — the original Sanskrit epic, the Ādi Kāvya.

It retrieves evidence across all **7 kandas and 645 sargas**, reasons over
that evidence using large language models, and returns a structured verdict
with Sanskrit citations.

---

### 📚 Data Source

- **Text:** Valmiki Ramayana (original Sanskrit + word-for-word English
  translation by Desiraju Hanumanta Rao & K. M. K. Murthy)
- **Source:** [valmikiramayan.net](https://www.valmikiramayan.net)

---

### 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Query pipeline | Python · Lang&#8203;Graph state machine |
| LLM reasoning | Anthropic Claude (Haiku + Sonnet) |
| Vector store | ChromaDB + OpenAI text-embedding-3-large |
| Knowledge graph | Neo4j |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Data | 645 structured sarga JSONs |

---

### ⚖️ How to Interpret Verdicts

| Verdict | Meaning |
|---------|---------|
| ✅ **TRUE** | Strong textual evidence supports the claim across multiple sargas |
| ❌ **FALSE** | Clear textual evidence contradicts the claim |
| 🟡 **PARTIALLY TRUE** | The claim is true in some kandas/contexts but not universally |
| ⬜ **INSUFFICIENT EVIDENCE** | The scraped corpus doesn't contain enough evidence to decide |

> The confidence score reflects how consistent and numerous the supporting
> passages are — not the certainty of the LLM itself.

---

### ⚠️ Limitations

- V1 covers the Valmiki Ramayana only (not Adhyatma Ramayana, Kamba Ramayana, etc.)
- Sanskrit citations are machine-verified but should be cross-checked with
  the original for scholarly use.
- Uttara Kanda interpolation questions are flagged but not filtered.
""")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Hanuman_flag.svg/120px-Hanuman_flag.svg.png",
    width=80,
)
st.sidebar.title("Scripture GPT")
st.sidebar.caption("Valmiki Ramayana · V1")

page = st.sidebar.radio(
    "Navigate",
    options=["Fact Checker", "Explore", "Character Explorer", "About"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**API:** `{API_BASE_URL}`")

# Health check in sidebar
try:
    hresp = requests.get(f"{API_BASE_URL}/health", timeout=3)
    if hresp.ok:
        st.sidebar.success("API online", icon="🟢")
    else:
        st.sidebar.warning("API returned non-OK", icon="🟡")
except Exception:
    st.sidebar.error("API offline", icon="🔴")

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if page == "Fact Checker":
    page_factcheck()
elif page == "Explore":
    page_explore()
elif page == "Character Explorer":
    page_character()
elif page == "About":
    page_about()
