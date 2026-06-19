# streamlit_app.py

import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="PDF RAG",
    page_icon="📄",
    layout="wide",
)

st.title("📄 PDF RAG Assistant")

# ------------------------------------------------------------------
# Helper — fetch document metadata once per session, cache it
# ------------------------------------------------------------------

@st.cache_data(ttl=30)
def fetch_document_metadata() -> list[dict]:
    try:
        r = requests.get(f"{API_BASE}/documents/metadata", timeout=5)
        if r.ok:
            return r.json().get("documents", [])
    except Exception:
        pass
    return []


# ------------------------------------------------------------------
# Layout: sidebar for filters, main column for upload + Q&A
# ------------------------------------------------------------------

sidebar = st.sidebar
main    = st

# ==================================================================
# SIDEBAR — metadata filters
# ==================================================================

sidebar.header("🔎 Search Filters")
sidebar.caption(
    "Narrow retrieval to a subset of your documents. "
    "Leave blank to search all."
)

docs_meta  = fetch_document_metadata()
filenames  = sorted({d["filename"] for d in docs_meta if d.get("filename")})
categories = sorted({d["category"] for d in docs_meta if d.get("category")})
authors    = sorted({d["author"]   for d in docs_meta if d.get("author")})
years      = sorted({d["year"]     for d in docs_meta if d.get("year")})
all_tags   = sorted({
    tag
    for d in docs_meta
    for tag in (d.get("tags") or [])
})

filter_filename = sidebar.selectbox(
    "📄 Filename",
    options=["(all)"] + filenames,
)

filter_category = sidebar.selectbox(
    "🗂 Category",
    options=["(all)"] + categories,
)

filter_author = sidebar.selectbox(
    "✍️ Author",
    options=["(all)"] + authors,
)

filter_tags = sidebar.multiselect(
    "🏷 Tags  (match ANY)",
    options=all_tags,
)

year_range = None
if years:
    year_min, year_max = int(min(years)), int(max(years))
    if year_min != year_max:
        year_range = sidebar.slider(
            "📅 Year range",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max),
        )
    else:
        sidebar.caption(f"📅 Only one year in collection: {year_min}")
        year_range = (year_min, year_max)

# Build the filters dict sent to the API (None → omit → ignored server-side)
active_filters = {}
if filter_filename != "(all)":
    active_filters["filename"] = filter_filename
if filter_category != "(all)":
    active_filters["category"] = filter_category
if filter_author != "(all)":
    active_filters["author"] = filter_author
if filter_tags:
    active_filters["tags"] = filter_tags
if year_range and years:
    if year_range[0] != int(min(years)) or year_range[1] != int(max(years)):
        active_filters["year_from"] = year_range[0]
        active_filters["year_to"]   = year_range[1]

if active_filters:
    sidebar.success(f"Active filters: {list(active_filters.keys())}")
else:
    sidebar.info("No filters — searching all documents.")

sidebar.divider()

# ==================================================================
# MAIN — Upload
# ==================================================================

main.header("📤 Upload a PDF")

uploaded_file = main.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file is not None:

    with main.expander("📝 Optional document metadata", expanded=True):
        m_col1, m_col2 = main.columns(2)

        with m_col1:
            meta_category = main.text_input(
                "Category",
                placeholder="e.g. annual_report, contract, manual",
            )
            meta_author = main.text_input(
                "Author",
                placeholder="e.g. Alice Smith",
            )

        with m_col2:
            meta_year = main.number_input(
                "Year",
                min_value=1900,
                max_value=2100,
                value=None,
                placeholder="e.g. 2024",
            )
            meta_tags = main.text_input(
                "Tags  (comma-separated)",
                placeholder="e.g. finance, budget, Q4",
            )

    if main.button("⬆️ Ingest PDF"):

        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        file_path = uploads_dir / uploaded_file.name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with main.spinner("Embedding (dense + sparse) and storing…"):

            with open(file_path, "rb") as f:
                form_data = {
                    "category": (None, meta_category or ""),
                    "author":   (None, meta_author   or ""),
                    "tags":     (None, meta_tags      or ""),
                }
                if meta_year:
                    form_data["year"] = (None, str(int(meta_year)))

                response = requests.post(
                    f"{API_BASE}/ingest",
                    files={"file": (uploaded_file.name, f, "application/pdf")},
                    data={k: v[1] for k, v in form_data.items()},
                )

        if response.ok:
            result = response.json()
            main.success(
                f"✅ Ingested **{result['chunks']}** chunks from `{result['source']}`"
            )
            st.cache_data.clear()   # refresh the sidebar filter dropdowns
        else:
            main.error(response.text)

# ==================================================================
# MAIN — Ingested documents
# ==================================================================

main.divider()

with main.expander("📚 Ingested documents"):
    if docs_meta:
        for d in docs_meta:
            tags_str = ", ".join(d.get("tags") or []) or "—"
            main.markdown(
                f"**{d['filename']}**  "
                f"| category: `{d.get('category') or '—'}`  "
                f"| author: `{d.get('author') or '—'}`  "
                f"| year: `{d.get('year') or '—'}`  "
                f"| tags: `{tags_str}`"
            )
    else:
        main.caption("No documents ingested yet.")

# ==================================================================
# MAIN — Ask
# ==================================================================

main.divider()
main.header("💬 Ask a Question")

question = main.text_input("Question")

q_col1, q_col2 = main.columns(2)

with q_col1:
    top_k = main.slider(
        "Top K (chunks → LLM, after reranking)",
        min_value=1, max_value=20, value=5,
    )

with q_col2:
    fetch_k = main.slider(
        "Fetch K (candidates before reranking)",
        min_value=max(top_k, 5), max_value=50, value=30,
    )

use_hybrid = main.toggle(
    "Hybrid search (dense + sparse → RRF)",
    value=True,
    help="ON: combines semantic and keyword retrieval. OFF: dense only.",
)

if main.button("Ask ▶", type="primary"):

    if not question.strip():
        main.warning("Please enter a question.")

    else:
        mode_label = "hybrid" if use_hybrid else "dense-only"

        with main.spinner(f"Retrieving ({mode_label}), reranking, generating…"):
            response = requests.post(
                f"{API_BASE}/query",
                json={
                    "question":   question,
                    "top_k":      top_k,
                    "fetch_k":    fetch_k,
                    "use_hybrid": use_hybrid,
                    "filters":    active_filters,
                },
            )

        if response.ok:
            result = response.json()

            main.subheader("Answer")
            main.write(result.get("answer", ""))

            sources = result.get("sources", [])
            if sources:
                main.subheader("Sources")
                for src in sources:
                    main.write(f"- {src}")

            used_mode = result.get("retrieval_mode", "?")
            main.caption(
                f"Mode: **{used_mode}** | "
                f"filters: **{list(active_filters.keys()) or 'none'}** | "
                f"→ reranked to top {top_k}"
            )

            matches = result.get("matches", [])
            if matches:
                with main.expander(f"🔍 {len(matches)} chunks sent to LLM"):
                    for i, m in enumerate(matches, start=1):

                        if m.get("rrf_score") is not None:
                            ret_score = f"RRF `{m['rrf_score']:.4f}`"
                        elif m.get("vector_score") is not None:
                            ret_score = f"cosine `{m['vector_score']:.3f}`"
                        else:
                            ret_score = "—"

                        rerank_score = (
                            f"`{m['rerank_score']:.3f}`"
                            if m.get("rerank_score") is not None else "—"
                        )

                        main.markdown(
                            f"**#{i}** — *{m.get('source', '')}*  \n"
                            f"retrieval: {ret_score} &nbsp;&nbsp; rerank: {rerank_score}"
                        )
                        main.caption(m.get("text", "")[:400])

        else:
            main.error(response.text)