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
# Ask Questions
# --------------------------------------------------

st.divider()

st.header("Ask a Question")

question = st.text_input(
    "Question",
)

top_k = st.slider(
    "Top K (chunks sent to the LLM, after reranking)",
    min_value=1,
    max_value=20,
    value=5,
)

fetch_k = st.slider(
    "Fetch K (candidates pulled from vector search, before reranking)",
    min_value=top_k,
    max_value=50,
    value=30,
)

if st.button("Ask"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching..."):

            response = requests.post(
                f"{API_BASE}/query",
                json={
                    "question": question,
                    "top_k": top_k,
                    "fetch_k": fetch_k,
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

            matches = result.get("matches", [])

            if matches:

                with st.expander(
                    f"Show the {len(matches)} chunks sent to the LLM "
                    "(with vector vs. rerank scores)"
                ):
                    for i, m in enumerate(matches, start=1):
                        st.markdown(
                            f"**#{i}** — *{m.get('source', '')}* "
                            f"&nbsp;&nbsp; vector: `{m.get('vector_score'):.3f}` "
                            f"&nbsp;&nbsp; rerank: `{m.get('rerank_score'):.3f}`"
                        )
                        st.caption(m.get("text", "")[:400])

        else:

            st.error(response.text)
