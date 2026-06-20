# streamlit_app.py

import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------

if "messages"   not in st.session_state:
    st.session_state.messages   = []
if "debug_info" not in st.session_state:
    st.session_state.debug_info = []


# ------------------------------------------------------------------
# Helpers
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

    # ── Upload ──────────────────────────────────────────────────────
    st.header("Documents")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        with st.expander("Metadata (optional)"):
            meta_category = st.text_input("Category", placeholder="e.g. report, contract")
            meta_author   = st.text_input("Author",   placeholder="e.g. Alice Smith")
            meta_year     = st.number_input("Year", min_value=1900, max_value=2100,
                                            value=None, placeholder="e.g. 2024")
            meta_tags     = st.text_input("Tags (comma-separated)",
                                          placeholder="e.g. finance, Q4")

        use_contextual = st.toggle(
            "Contextual enrichment",
            value=False,
            help=(
                "Before embedding each chunk, an LLM generates a short context "
                "sentence that situates the chunk within the document. "
                "This makes retrieval significantly more accurate for ambiguous "
                "chunks (tables, numbered lists, cross-references). "
                "⚠️ Adds ~1 LLM call per chunk — slower ingest."
            ),
        )

        if use_contextual:
            st.info(
                "Contextual enrichment is ON. Ingest will be slower — "
                "one LLM call per chunk to generate context prefixes.",
                icon="⏳",
            )

        if st.button("⬆️ Ingest", use_container_width=True):
            uploads_dir = Path("uploads")
            uploads_dir.mkdir(exist_ok=True)
            fp = uploads_dir / uploaded_file.name

            with open(fp, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(
                "Ingesting with contextual enrichment…"
                if use_contextual
                else "Ingesting…"
            ):
                with open(fp, "rb") as f:
                    form_data = {
                        "category":       meta_category or "",
                        "author":         meta_author   or "",
                        "tags":           meta_tags     or "",
                        "use_contextual": "true" if use_contextual else "false",
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
                enriched_label = " (contextually enriched ✨)" if res.get("contextual_enriched") else ""
                st.success(f"✅ {res['chunks']} chunks from `{res['source']}`{enriched_label}")
                st.cache_data.clear()
            else:
                st.error(resp.text)

    st.divider()

    # ── Ingested documents ───────────────────────────────────────────
    docs_meta = fetch_document_metadata()

    with st.expander(f"📚 Ingested ({len(docs_meta)})", expanded=False):
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

    # ── Filters ──────────────────────────────────────────────────────
    st.header("🔎 Filters")

    filenames  = sorted({d["filename"] for d in docs_meta if d.get("filename")})
    categories = sorted({d["category"] for d in docs_meta if d.get("category")})
    authors    = sorted({d["author"]   for d in docs_meta if d.get("author")})
    years      = sorted({d["year"]     for d in docs_meta if d.get("year")})
    all_tags   = sorted({t for d in docs_meta for t in (d.get("tags") or [])})

    f_filename = st.selectbox("File",     ["(all)"] + filenames)
    f_category = st.selectbox("Category", ["(all)"] + categories)
    f_author   = st.selectbox("Author",   ["(all)"] + authors)
    f_tags     = st.multiselect("Tags (match any)", all_tags)

    year_range = None
    if len(years) >= 2:
        year_range = st.slider(
            "Year",
            min_value=int(min(years)),
            max_value=int(max(years)),
            value=(int(min(years)), int(max(years))),
        )

    active_filters: dict = {}
    if f_filename != "(all)":   active_filters["filename"] = f_filename
    if f_category != "(all)":   active_filters["category"] = f_category
    if f_author   != "(all)":   active_filters["author"]   = f_author
    if f_tags:                  active_filters["tags"]      = f_tags
    if year_range and len(years) >= 2:
        if year_range != (int(min(years)), int(max(years))):
            active_filters["year_from"] = year_range[0]
            active_filters["year_to"]   = year_range[1]

    if active_filters:
        st.success(f"Filtering: {', '.join(active_filters)}")

    st.divider()

    # ── Settings ─────────────────────────────────────────────────────
    st.header("Settings")

    top_k   = st.slider("Top K (→ LLM)",        min_value=1,  max_value=20, value=5)
    fetch_k = st.slider("Fetch K (pre-rerank)",  min_value=5,  max_value=50, value=30)

    use_hybrid = st.toggle(
        "Hybrid search",
        value=True,
        help="Dense + sparse → RRF. OFF = dense only.",
    )

    use_window = st.toggle(
        "Window expansion",
        value=True,
        help=(
            "After retrieval, each matched chunk is expanded with its "
            "neighboring chunks before being sent to the LLM. "
            "This gives the LLM more surrounding context, reducing "
            "answers that are cut off mid-thought. "
            "Only effective if the document was ingested with this app "
            "(v6+) which stores prev/next chunk in the payload."
        ),
    )

    show_debug = st.toggle("Show debug info", value=False)

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages   = []
        st.session_state.debug_info = []
        st.rerun()


# ==================================================================
# MAIN — chat
# ==================================================================

st.title("🤖 RAG Chatbot")

if not st.session_state.messages:
    st.caption(
        "Upload a PDF in the sidebar, then start chatting. "
        "Enable **Contextual enrichment** before ingesting for best retrieval accuracy. "
        "Enable **Window expansion** to give the LLM more context per answer."
    )

# Render existing conversation
assistant_turn_idx = 0

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

    if msg["role"] == "assistant":
        if show_debug and assistant_turn_idx < len(st.session_state.debug_info):
            info = st.session_state.debug_info[assistant_turn_idx]

            with st.expander("🔍 Debug — this answer"):

                # Query rewriting
                if info.get("rewritten_question"):
                    st.info(f"**Query rewritten to:** {info['rewritten_question']}")

                # Mode / window
                flags = []
                if info.get("window_expanded"):
                    flags.append("window expanded ✅")
                else:
                    flags.append("window expansion off")

                st.caption(
                    f"Retrieval: **{info.get('retrieval_mode', '?')}** | "
                    f"{' | '.join(flags)} | "
                    f"sources: {', '.join(info.get('sources', []))}"
                )

                # Per-chunk scores + expansion diff
                for j, m in enumerate(info.get("matches", []), 1):
                    badges = []
                    if m.get("was_enriched"):
                        badges.append("✨ enriched")

                    sent = m.get("text_sent_to_llm") or ""
                    orig = m.get("text", "")
                    if sent and sent != orig:
                        extra = len(sent) - len(orig)
                        badges.append(f"🪟 +{extra} chars window")

                    badge_str = "  ".join(badges)

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
                        f"**#{j}** *{m.get('source', '')}* — "
                        f"{score} → rerank {rerank_s}  {badge_str}"
                    )

                    # Show what the LLM actually received if it differs
                    if sent and sent != orig:
                        tab_orig, tab_sent = st.tabs(["Matched text", "Sent to LLM"])
                        with tab_orig:
                            st.caption(orig[:400])
                        with tab_sent:
                            st.caption(sent[:600])
                    else:
                        st.caption(orig[:400])

        assistant_turn_idx += 1

# Chat input
if prompt := st.chat_input("Ask something about your documents…"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            resp = requests.post(
                f"{API_BASE}/query",
                json={
                    "question":              prompt,
                    "history":               st.session_state.messages[:-1],
                    "top_k":                 top_k,
                    "fetch_k":               fetch_k,
                    "use_hybrid":            use_hybrid,
                    "use_window_expansion":  use_window,
                    "filters":               active_filters,
                },
            )

        if resp.ok:
            result = resp.json()
            answer = result.get("answer", "")

            st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.debug_info.append({
                "rewritten_question": result.get("rewritten_question"),
                "retrieval_mode":     result.get("retrieval_mode"),
                "window_expanded":    result.get("window_expanded"),
                "sources":            result.get("sources", []),
                "matches":            result.get("matches", []),
            })

            sources = result.get("sources", [])
            if sources:
                st.caption("Sources: " + " · ".join(f"`{s}`" for s in sources))

            # Small contextual indicators under the answer
            badges = []
            if result.get("rewritten_question"):
                badges.append("query rewritten")
            if result.get("window_expanded"):
                badges.append("window expanded")
            if badges:
                st.caption(" · ".join(badges))

        else:
            error_msg = f"Error: {resp.text}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.session_state.debug_info.append({})