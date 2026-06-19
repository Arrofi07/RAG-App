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
    ChatMessage,
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

app = FastAPI(title="RAG Chatbot API", version="5.0.0")

store = QdrantStorage(dim=EMBED_DIM)

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]

# How many past turns to include in the prompt.
# Older turns are dropped to avoid bloating the context window.
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))


# -------------------------
# Request models
# -------------------------

class QueryRequest(BaseModel):
    question:   str
    history:    list[ChatMessage] = []   # full conversation so far, oldest first
    top_k:      int         = 5
    fetch_k:    int         = 30
    use_hybrid: bool        = True
    filters:    MetaFilter  = MetaFilter()


# -------------------------
# LLM helper
# -------------------------

def _call_llm(prompt: str) -> str:
    """Try each Gemini model in order, return the first successful response."""
    for model_name in GEMINI_MODELS:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if hasattr(resp, "text") and resp.text:
                return resp.text.strip()
        except Exception as e:
            log.warning("%s failed: %s", model_name, e)

    raise HTTPException(status_code=503, detail="All LLM backends unavailable.")


# -------------------------
# Query rewriting
# -------------------------

def _rewrite_question(question: str, history: list[ChatMessage]) -> str:
    """
    Use the LLM to turn a follow-up question into a fully self-contained one.

    Why this matters: the embedding model has no memory — if the user asks
    "what about Q4?" after a question about revenue, the vector search would
    retrieve chunks about Q4 in general, not Q4 revenue. Rewriting first
    gives the retriever a precise, context-rich query to work with.

    When there is no history we skip the LLM call entirely and return the
    original question unchanged.
    """
    if not history:
        return question

    # Build a compact transcript of the last N turns
    recent = history[-(MAX_HISTORY_TURNS):]
    transcript = "\n".join(
        f"{m.role.upper()}: {m.content}" for m in recent
    )

    prompt = f"""You are a query rewriter for a RAG system.

Given the conversation history and the user's latest message, rewrite the
latest message into a single, fully self-contained question that can be
understood without the conversation history.

Rules:
- Output ONLY the rewritten question, no explanation or preamble.
- If the latest message is already self-contained, return it unchanged.
- Preserve the original intent and language of the user.

Conversation history:
{transcript}

Latest message: {question}

Rewritten question:"""

    try:
        return _call_llm(prompt)
    except Exception:
        # If rewriting fails for any reason, fall back to the original
        log.warning("Query rewriting failed, using original question.")
        return question


# -------------------------
# Health check
# -------------------------

@app.get("/")
def root():
    return {"status": "running", "version": "5.0.0 — chatbot with memory"}


# -------------------------
# Ingest PDF
# -------------------------

@app.post("/ingest", response_model=IngestResult)
async def ingest_pdf(
    file:     UploadFile    = File(...),
    category: Optional[str] = Form(None),
    author:   Optional[str] = Form(None),
    year:     Optional[int] = Form(None),
    tags:     Optional[str] = Form(None),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    tag_list = (
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
        source_id   = file.filename

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]

        payloads = [
            {
                "source":        source_id,
                "text":          chunks[i],
                "filename":      source_id,
                "chunk_id":      i,
                "document_type": "pdf",
                "uploaded_at":   datetime.utcnow().isoformat(),
                "category":      category,
                "author":        author,
                "year":          year,
                "tags":          tag_list,
            }
            for i in range(len(chunks))
        ]

        store.upsert(
            ids=ids,
            dense_vectors=embeddings["dense"],
            sparse_vectors=embeddings["sparse"],
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
# Chat query
# -------------------------

@app.post("/query", response_model=QueryResult)
def query(req: QueryRequest):
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # ── Stage 1: Query rewriting ─────────────────────────────────────────
    # Resolve references to previous turns ("what about the second one?")
    # into a standalone question before it hits the embedding model.
    rewritten = _rewrite_question(question, req.history)

    if rewritten != question:
        log.info("Rewritten: '%s' → '%s'", question, rewritten)

    # ── Stage 2: Hybrid retrieval ────────────────────────────────────────
    q_emb    = embed([rewritten])
    q_dense  = q_emb["dense"][0]
    q_sparse = q_emb["sparse"][0]

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

    # ── Stage 3: Rerank ──────────────────────────────────────────────────
    reranked = rerank(rewritten, candidates, top_k=req.top_k)

    contexts = [c["text"] for c in reranked]

    sources = []
    for c in reranked:
        if c["source"] and c["source"] not in sources:
            sources.append(c["source"])

    # ── Stage 4: Generate with history ───────────────────────────────────
    context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))

    # Include the last N turns of history so the LLM can produce coherent
    # follow-up answers (e.g. "As I mentioned earlier..." or "Adding to the
    # previous answer...").
    recent_history = req.history[-(MAX_HISTORY_TURNS):]
    history_block = "\n".join(
        f"{m.role.upper()}: {m.content}" for m in recent_history
    ) if recent_history else ""

    prompt = f"""You are a helpful assistant that answers questions based on document context.

Instructions:
- Answer using ONLY the provided context.
- If the context does not contain the answer, say you don't know.
- You may refer to the conversation history to give coherent follow-up answers.
- Be concise and cite context numbers like [1], [2] where relevant.

Retrieved context:

{context_block}
{"Conversation history:" + chr(10) + history_block if history_block else ""}

Current question: {question}

Answer:"""

    answer = _call_llm(prompt)

    return QueryResult(
        answer=answer,
        sources=sources,
        num_contexts=len(contexts),
        retrieval_mode=retrieval_mode,
        rewritten_question=rewritten if rewritten != question else None,
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
# Documents
# -------------------------

@app.get("/documents", response_model=DocumentListResult)
def list_documents():
    return DocumentListResult(documents=[d["filename"] for d in store.list_documents()])


@app.get("/documents/metadata")
def list_documents_metadata():
    return {"documents": store.list_documents()}