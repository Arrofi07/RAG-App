# streamlit_app.py

import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="AI Advisor for Studying in Germany",
    page_icon="🤖",
    layout="wide",
)

# ------------------------------------------------------------------
# Session state initialisation
# ------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role": str, "content": str}

if "debug_info" not in st.session_state:
    st.session_state.debug_info = []  # parallel list, one entry per assistant turn


# ------------------------------------------------------------------
# Helper
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


# ==================================================================
# SIDEBAR
# ==================================================================

with st.sidebar:
    st.header("📄 Documents")

    # --- Upload ---
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        with st.expander("📝 Metadata (optional)"):
            meta_category = st.text_input("Category", placeholder="e.g. report, contract")
            meta_author   = st.text_input("Author",   placeholder="e.g. Alice Smith")
            meta_year     = st.number_input("Year", min_value=1900, max_value=2100,
                                            value=None, placeholder="e.g. 2024")
            meta_tags     = st.text_input("Tags (comma-separated)",
                                          placeholder="e.g. finance, Q4")

        if st.button("⬆️ Ingest", use_container_width=True):
            uploads_dir = Path("uploads")
            uploads_dir.mkdir(exist_ok=True)
            fp = uploads_dir / uploaded_file.name

            with open(fp, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Ingesting…"):
                with open(fp, "rb") as f:
                    form_data = {
                        "category": meta_category or "",
                        "author":   meta_author   or "",
                        "tags":     meta_tags     or "",
                    }
                    if meta_year:
                        form_data["year"] = str(int(meta_year))

                    resp = requests.post(
                        f"{API_BASE}/ingest",
                        files={"file": (uploaded_file.name, f, "application/pdf")},
                        data=form_data,
                    )

            if resp.ok:
                res = resp.json()
                st.success(f"✅ {res['chunks']} chunks from `{res['source']}`")
                st.cache_data.clear()
            else:
                st.error(resp.text)

    st.divider()

    # --- Ingested documents ---
    docs_meta = fetch_document_metadata()

    with st.expander(f"📚 Source Documents ({len(docs_meta)})", expanded=False):
        if docs_meta:
            for d in docs_meta:
                tags_str = ", ".join(d.get("tags") or []) or "—"
                st.markdown(
                    f"**{d['filename']}**  \n"
                    f"`{d.get('category') or '—'}` | "
                    f"`{d.get('author') or '—'}` | "
                    f"`{d.get('year') or '—'}` | "
                    f"tags: `{tags_str}`"
                )
        else:
            st.caption("No documents yet.")

    st.divider()

    # --- Search filters ---
    st.header("🔎 Filters")

    filenames  = sorted({d["filename"] for d in docs_meta if d.get("filename")})
    categories = sorted({d["category"] for d in docs_meta if d.get("category")})
    authors    = sorted({d["author"]   for d in docs_meta if d.get("author")})
    years      = sorted({d["year"]     for d in docs_meta if d.get("year")})
    all_tags   = sorted({t for d in docs_meta for t in (d.get("tags") or [])})

    f_filename = st.selectbox("📄 File",     ["all"] + filenames)
    f_category = st.selectbox("🗂 Category", ["all"] + categories)
    f_author   = st.selectbox("✍️ Author",   ["all"] + authors)
    f_tags     = st.multiselect("🏷 Tags (match any)", all_tags)

    year_range = None
    if len(years) >= 2:
        year_range = st.slider(
            "📅 Year",
            min_value=int(min(years)), max_value=int(max(years)),
            value=(int(min(years)), int(max(years))),
        )

    active_filters: dict = {}
    if f_filename != "all":   active_filters["filename"] = f_filename
    if f_category != "all":   active_filters["category"] = f_category
    if f_author   != "all":   active_filters["author"]   = f_author
    if f_tags:                  active_filters["tags"]      = f_tags
    if year_range and len(years) >= 2:
        if year_range != (int(min(years)), int(max(years))):
            active_filters["year_from"] = year_range[0]
            active_filters["year_to"]   = year_range[1]

    if active_filters:
        st.success(f"Filtering by: {', '.join(active_filters)}")

    st.divider()

    # --- Retrieval settings ---
    st.header("⚙️ Settings")

    top_k = st.slider("Top K (→ LLM)", min_value=1, max_value=20, value=5)
    fetch_k = st.slider("Fetch K (pre-rerank)", min_value=5, max_value=50, value=30)
    use_hybrid = st.toggle("Hybrid search", value=True,
                           help="Dense + sparse → RRF. OFF = dense only.")
    show_debug = st.toggle("Show debug info", value=False,
                           help="Show rewritten question and chunk scores.")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages   = []
        st.session_state.debug_info = []
        st.rerun()


# ==================================================================
# MAIN — chat interface
# ==================================================================

st.title("🤖 AI Advisor for Studying in Germany")

if not st.session_state.messages:
    st.caption("Upload a PDF in the sidebar, then start chatting below.")

# Render existing conversation
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

    # Debug panel after each assistant turn
    if msg["role"] == "assistant" and show_debug:
        turn_idx = sum(
            1 for m in st.session_state.messages[:i+1] if m["role"] == "assistant"
        ) - 1

        if turn_idx < len(st.session_state.debug_info):
            info = st.session_state.debug_info[turn_idx]

            with st.expander("🔍 Debug info for this answer"):

                if info.get("rewritten_question"):
                    st.info(f"**Query rewritten to:** {info['rewritten_question']}")

                st.caption(
                    f"Mode: **{info.get('retrieval_mode', '?')}** | "
                    f"filters: **{list(active_filters.keys()) or 'none'}** | "
                    f"sources: {', '.join(info.get('sources', []))}"
                )

                for j, m in enumerate(info.get("matches", []), 1):
                    if m.get("rrf_score") is not None:
                        score = f"RRF `{m['rrf_score']:.4f}`"
                    elif m.get("vector_score") is not None:
                        score = f"cosine `{m['vector_score']:.3f}`"
                    else:
                        score = "—"

                    rerank_s = (
                        f"`{m['rerank_score']:.3f}`"
                        if m.get("rerank_score") is not None else "—"
                    )

                    st.markdown(
                        f"**#{j}** *{m.get('source','')}* — "
                        f"{score} → rerank {rerank_s}"
                    )
                    st.caption(m.get("text", "")[:300])

# Chat input (sticks to bottom of page)
if prompt := st.chat_input("Ask something about your documents…"):

    # Show the user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the API
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            resp = requests.post(
                f"{API_BASE}/query",
                json={
                    "question":   prompt,
                    "history":    st.session_state.messages[:-1],  # exclude current turn
                    "top_k":      top_k,
                    "fetch_k":    fetch_k,
                    "use_hybrid": use_hybrid,
                    "filters":    active_filters,
                },
            )

        if resp.ok:
            result = resp.json()
            answer = result.get("answer", "")

            st.markdown(answer)

            # Save assistant turn
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Save debug info for this turn
            st.session_state.debug_info.append({
                "rewritten_question": result.get("rewritten_question"),
                "retrieval_mode":     result.get("retrieval_mode"),
                "sources":            result.get("sources", []),
                "matches":            result.get("matches", []),
            })

            # Inline source attribution under the answer
            sources = result.get("sources", [])
            if sources:
                st.caption("Sources: " + " · ".join(f"`{s}`" for s in sources))

        else:
            error_msg = f"Error: {resp.text}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.session_state.debug_info.append({})