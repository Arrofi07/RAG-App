# main.py

import os
import uuid
import logging
import tempfile
from typing import Optional
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from google import genai

from data_loader import load_and_chunk_pdf, embed, EMBED_DIM
from vector_db import QdrantStorage
from reranker import rerank
from custom_types import (
    MetaFilter,
    IngestResult,
    DocumentListResult,
    QueryResult,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# -------------------------
# Configuration
# -------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="RAG PDF API", version="4.0.0")

store = QdrantStorage(dim=EMBED_DIM)

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]


# -------------------------
# Request models
# -------------------------

class QueryRequest(BaseModel):
    question:    str
    top_k:       int            = 5
    fetch_k:     int            = 30
    use_hybrid:  bool           = True
    filters:     MetaFilter     = MetaFilter()   # all fields None by default → no filter


# -------------------------
# Health check
# -------------------------

@app.get("/")
def root():
    return {"status": "running", "version": "4.0.0 — metadata filters enabled"}


# -------------------------
# Ingest PDF
# -------------------------

@app.post("/ingest", response_model=IngestResult)
async def ingest_pdf(
    file:     UploadFile     = File(...),
    category: Optional[str]  = Form(None),
    author:   Optional[str]  = Form(None),
    year:     Optional[int]  = Form(None),
    tags:     Optional[str]  = Form(None),   # comma-separated, e.g. "finance,q4,draft"
):
    """
    Ingest a PDF alongside optional metadata.

    Metadata is stored in every chunk's Qdrant payload and can be used
    to filter queries later via the `filters` field in QueryRequest.

    Form fields (all optional)
    --------------------------
    category : free-text label  e.g. "annual_report", "contract"
    author   : document author  e.g. "Alice Smith"
    year     : publication year e.g. 2024  (integer, enables range queries)
    tags     : comma-separated  e.g. "finance,budget,Q4"
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Parse comma-separated tags into a list
    tag_list: list[str] = (
        [t.strip() for t in tags.split(",") if t.strip()]
        if tags else []
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        pdf_path = tmp.name

    try:
        chunks = load_and_chunk_pdf(pdf_path)

        if not chunks:
            raise HTTPException(status_code=400, detail="No text found in PDF.")

        embeddings  = embed(chunks)
        dense_vecs  = embeddings["dense"]
        sparse_vecs = embeddings["sparse"]

        source_id = file.filename

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]

        payloads = [
            {
                # Core fields
                "source":        source_id,
                "text":          chunks[i],
                "filename":      source_id,
                "chunk_id":      i,
                "document_type": "pdf",
                "uploaded_at":   datetime.utcnow().isoformat(),
                # User-supplied metadata (None values are stored but filtered
                # away cleanly by _build_filter when searching)
                "category":      category,
                "author":        author,
                "year":          year,
                "tags":          tag_list,
            }
            for i in range(len(chunks))
        ]

        store.upsert(
            ids=ids,
            dense_vectors=dense_vecs,
            sparse_vectors=sparse_vecs,
            payloads=payloads,
        )

        log.info(
            "Ingested %d chunks from '%s' | category=%s author=%s year=%s tags=%s",
            len(chunks), source_id, category, author, year, tag_list,
        )

        return IngestResult(success=True, source=source_id, chunks=len(chunks))

    finally:
        Path(pdf_path).unlink(missing_ok=True)


# -------------------------
# Query
# -------------------------

@app.post("/query", response_model=QueryResult)
def query(req: QueryRequest):
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    q_embeddings = embed([question])
    q_dense      = q_embeddings["dense"][0]
    q_sparse     = q_embeddings["sparse"][0]

    # ── Stage 1: Retrieval + metadata filter ────────────────────────────────
    # MetaFilter fields that are None are silently ignored by _build_filter,
    # so passing MetaFilter() (all None) is the same as no filter at all.
    if req.use_hybrid:
        candidates = store.search_hybrid_candidates(
            dense_vector=q_dense,
            sparse_vector=q_sparse,
            fetch_k=req.fetch_k,
            meta=req.filters,
        )
        retrieval_mode = "hybrid"
    else:
        candidates = store.search_candidates(
            query_vector=q_dense,
            fetch_k=req.fetch_k,
            meta=req.filters,
        )
        retrieval_mode = "dense"

    log.info(
        "Query | mode=%s candidates=%d filters=%s",
        retrieval_mode, len(candidates), req.filters.model_dump(exclude_none=True),
    )

    # ── Stage 2: Rerank ─────────────────────────────────────────────────────
    reranked = rerank(question, candidates, top_k=req.top_k)

    contexts = [c["text"] for c in reranked]

    sources = []
    for c in reranked:
        if c["source"] and c["source"] not in sources:
            sources.append(c["source"])

    # ── Stage 3: Generate ───────────────────────────────────────────────────
    context_block = "\n\n".join(f"- {c}" for c in contexts)

    prompt = f"""You are a helpful assistant.

Answer ONLY using the provided context.
If the answer is not contained in the context, say that you don't know.

Context:

{context_block}

Question:

{question}
"""

    response = None
    for model_name in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            break
        except Exception as e:
            log.warning("%s failed: %s", model_name, e)

    if response is None:
        raise HTTPException(status_code=503, detail="All LLM backends unavailable.")

    answer = ""
    if hasattr(response, "text") and response.text:
        answer = response.text.strip()

    return QueryResult(
        answer=answer,
        sources=sources,
        num_contexts=len(contexts),
        retrieval_mode=retrieval_mode,
        matches=[
            {
                "text":         c["text"],
                "source":       c["source"],
                "rrf_score":    c.get("rrf_score"),
                "vector_score": c.get("vector_score"),
                "rerank_score": c.get("rerank_score"),
            }
            for c in reranked
        ],
    )


# -------------------------
# List ingested documents
# -------------------------

@app.get("/documents", response_model=DocumentListResult)
def list_documents():
    docs = store.list_documents()
    # Return just filenames for the DocumentListResult schema;
    # full metadata available in the raw list for future endpoints
    return DocumentListResult(documents=[d["filename"] for d in docs])


@app.get("/documents/metadata")
def list_documents_metadata():
    """Full metadata per document — used by the UI to populate filter dropdowns."""
    return {"documents": store.list_documents()}