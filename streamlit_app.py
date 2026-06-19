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
    layout="centered",
)

st.title("📄 PDF RAG Assistant")

# --------------------------------------------------
# Upload PDF
# --------------------------------------------------

st.header("Upload a PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"],
)

if uploaded_file is not None:

    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)

    file_path = uploads_dir / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Ingest PDF"):

        with st.spinner("Embedding and storing document..."):

            with open(file_path, "rb") as f:
                response = requests.post(
                    f"{API_BASE}/ingest",
                    files={
                        "file": (
                            uploaded_file.name,
                            f,
                            "application/pdf",
                        )
                    },
                )

        if response.ok:
            result = response.json()

            st.success(
                f"Successfully ingested {result['chunks']} chunks."
            )

        else:
            st.error(response.text)

# --------------------------------------------------
# Ingested documents
# --------------------------------------------------
 
st.divider()
 
with st.expander("📚 Ingested documents"):
    docs_resp = requests.get(f"{API_BASE}/documents")
 
    if docs_resp.ok:
        docs = docs_resp.json().get("documents", [])
        if docs:
            for d in docs:
                st.write(f"- {d}")
        else:
            st.caption("No documents ingested yet.")
    else:
        st.caption("Could not reach API.")

# --------------------------------------------------
# Ask Questions
# --------------------------------------------------
 
st.divider()
 
st.header("Ask a Question")
 
question = st.text_input("Question")
 
col1, col2 = st.columns(2)
 
with col1:
    top_k = st.slider(
        "Top K (chunks → LLM, after reranking)",
        min_value=1,
        max_value=20,
        value=5,
    )
 
with col2:
    fetch_k = st.slider(
        "Fetch K (candidates before reranking)",
        min_value=max(top_k, 5),
        max_value=50,
        value=30,
    )
 
use_hybrid = st.toggle(
    "Hybrid search (dense + BM25 sparse → RRF)",
    value=True,
    help=(
        "ON → combines semantic (dense) and keyword (sparse) retrieval "
        "using Reciprocal Rank Fusion before reranking.\n\n"
        "OFF → dense-only (useful to compare quality)."
    ),
)
 
if st.button("Ask", type="primary"):
 
    if not question.strip():
        st.warning("Please enter a question.")
 
    else:
 
        mode_label = "hybrid" if use_hybrid else "dense-only"
 
        with st.spinner(f"Searching ({mode_label}) and generating answer..."):
 
            response = requests.post(
                f"{API_BASE}/query",
                json={
                    "question":   question,
                    "top_k":      top_k,
                    "fetch_k":    fetch_k,
                    "use_hybrid": use_hybrid,
                },
            )
 
        if response.ok:
 
            result = response.json()
 
            st.subheader("Answer")
            st.write(result.get("answer", ""))
 
            sources = result.get("sources", [])
            if sources:
                st.subheader("Sources")
                for src in sources:
                    st.write(f"- {src}")
 
            # Show retrieval mode actually used
            used_mode = result.get("retrieval_mode", "?")
            st.caption(f"Retrieval mode: **{used_mode}** → reranked to top {top_k}")
 
            # Score breakdown per chunk
            matches = result.get("matches", [])
            if matches:
                with st.expander(
                    f"🔍 Show {len(matches)} chunks sent to LLM (scores)"
                ):
                    for i, m in enumerate(matches, start=1):
 
                        # Show whichever retrieval score is present
                        if m.get("rrf_score") is not None:
                            retrieval_score = f"RRF `{m['rrf_score']:.4f}`"
                        elif m.get("vector_score") is not None:
                            retrieval_score = f"cosine `{m['vector_score']:.3f}`"
                        else:
                            retrieval_score = "—"
 
                        rerank_score = (
                            f"`{m['rerank_score']:.3f}`"
                            if m.get("rerank_score") is not None
                            else "—"
                        )
 
                        st.markdown(
                            f"**#{i}** — *{m.get('source', '')}*  \n"
                            f"retrieval: {retrieval_score} &nbsp;&nbsp; "
                            f"rerank: {rerank_score}"
                        )
                        st.caption(m.get("text", "")[:400])
 
        else:
            st.error(response.text)
