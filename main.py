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

from data_loader import load_chunks, embed, EMBED_DIM
from vector_db import QdrantStorage
from reranker import rerank
from context_builder import generate_chunk_context, build_contextual_text
from custom_types import (
    ChatMessage,
    MetaFilter,
    IngestResult,
    DocumentListResult,
    QueryResult,
    MatchItem,
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

app = FastAPI(title="RAG Chatbot API", version="6.0.0")

store = QdrantStorage(dim=EMBED_DIM)

GEMINI_MODELS      = ["gemini-2.5-flash", "gemini-2.0-flash"]
MAX_HISTORY_TURNS  = int(os.getenv("MAX_HISTORY_TURNS", "6"))


# -------------------------
# Request model
# -------------------------

class QueryRequest(BaseModel):
    question:            str
    history:             list[ChatMessage] = []
    top_k:               int        = 5
    fetch_k:             int        = 30
    use_hybrid:          bool       = True
    use_window_expansion: bool      = True   # expand matched chunks with neighbors
    filters:             MetaFilter = MetaFilter()


# -------------------------
# LLM helper
# -------------------------

def _call_llm(prompt: str) -> str:
    for model_name in GEMINI_MODELS:
        try:
            resp = client.models.generate_content(model=model_name, contents=prompt)
            if hasattr(resp, "text") and resp.text:
                return resp.text.strip()
        except Exception as e:
            log.warning("%s failed: %s", model_name, e)
    raise HTTPException(status_code=503, detail="All LLM backends unavailable.")


# -------------------------
# Query rewriting
# -------------------------

def _rewrite_question(question: str, history: list[ChatMessage]) -> str:
    if not history:
        return question

    recent     = history[-(MAX_HISTORY_TURNS):]
    transcript = "\n".join(f"{m.role.upper()}: {m.content}" for m in recent)

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
        log.warning("Query rewriting failed, using original question.")
        return question


# -------------------------
# Window expansion
# -------------------------

def _expand_with_window(candidates: list[dict]) -> list[dict]:
    """
    Expand each retrieved chunk with its stored neighbors.

    Why this helps:
      A chunk that perfectly answers a retrieval query may lack the
      surrounding context the LLM needs to understand it — it might start
      mid-sentence or reference something from the previous paragraph.
      By prepending prev_chunk and appending next_chunk, the LLM receives
      the full semantic window around the matched text.

    The original `text` is preserved so the UI can still show what was
    actually matched.  `text_sent_to_llm` holds the expanded version.

    Deduplication: if two neighboring chunks both matched, their windows
    would overlap.  We track seen chunk IDs and skip expansion for chunks
    whose neighbors we've already included via another candidate.
    """
    for c in candidates:
        parts = [p for p in [c.get("prev_chunk", ""), c["text"], c.get("next_chunk", "")] if p]
        c["text_sent_to_llm"] = "\n\n".join(parts)

    return candidates


# -------------------------
# Health check
# -------------------------

@app.get("/")
def root():
    return {"status": "running", "version": "6.0.0 — context-aware retrieval"}


# -------------------------
# Ingest PDF
# -------------------------

@app.post("/ingest", response_model=IngestResult)
async def ingest_pdf(
    file:              UploadFile    = File(...),
    category:          Optional[str] = Form(None),
    author:            Optional[str] = Form(None),
    year:              Optional[int] = Form(None),
    tags:              Optional[str] = Form(None),
    use_contextual:    str           = Form("false"),  # "true" / "false"
):
    """
    Ingest a PDF with optional metadata and optional contextual enrichment.

    use_contextual (bool, default false)
    ─────────────────────────────────────
    When true, an LLM call is made for every chunk to generate a short
    context sentence describing where that chunk sits in the document.
    That context is prepended to the chunk before embedding, so vector
    search becomes context-aware rather than purely content-aware.

    This adds ~1 LLM call per chunk to ingest time — for a 50-chunk PDF
    expect roughly 50 extra fast LLM calls.  The quality improvement is
    significant for documents where chunks are ambiguous without context
    (tables, numbered lists, cross-references, etc.).

    Chunks are always stored with prev_chunk / next_chunk neighbors
    (used for window expansion at query time), regardless of this flag.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    do_contextual = use_contextual.lower() in ("true", "1", "yes")

    tag_list = (
        [t.strip() for t in tags.split(",") if t.strip()]
        if tags else []
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        pdf_path = tmp.name

    try:
        loaded     = load_chunks(pdf_path)
        chunks     = loaded["chunks"]        # list of dicts with text, prev, next, page
        full_text  = loaded["full_text"]     # entire doc text for context generation

        if not chunks:
            raise HTTPException(status_code=400, detail="No text found in PDF.")

        source_id = file.filename

        # ── Contextual enrichment ────────────────────────────────────────────
        # For each chunk, optionally generate an LLM context prefix and
        # build the enriched text that will be embedded.
        texts_to_embed = []
        contextual_texts = []

        for i, chunk in enumerate(chunks):
            if do_contextual:
                log.info(
                    "Generating context for chunk %d/%d from '%s'",
                    i + 1, len(chunks), source_id,
                )
                ctx = generate_chunk_context(
                    full_doc_text=full_text,
                    chunk_text=chunk["text"],
                    source=source_id,
                    llm_fn=_call_llm,
                )
                enriched = build_contextual_text(ctx, chunk["text"])
            else:
                ctx      = ""
                enriched = chunk["text"]

            contextual_texts.append(ctx)
            texts_to_embed.append(enriched)

        # ── Embed ────────────────────────────────────────────────────────────
        embeddings = embed(texts_to_embed)

        # ── Build payloads ───────────────────────────────────────────────────
        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]

        payloads = [
            {
                # Core
                "source":           source_id,
                "text":             chunks[i]["text"],       # original, always shown
                "prev_chunk":       chunks[i]["prev_chunk"],
                "next_chunk":       chunks[i]["next_chunk"],
                "chunk_index":      chunks[i]["chunk_index"],
                "page":             chunks[i]["page"],
                "filename":         source_id,
                "document_type":    "pdf",
                "uploaded_at":      datetime.utcnow().isoformat(),
                # User metadata
                "category":         category,
                "author":           author,
                "year":             year,
                "tags":             tag_list,
                # Context-aware fields
                "contextual_text":  contextual_texts[i] or None,  # None if not enriched
                "was_enriched":     do_contextual and bool(contextual_texts[i]),
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
            "Ingested %d chunks from '%s' | contextual=%s category=%s author=%s year=%s tags=%s",
            len(chunks), source_id, do_contextual, category, author, year, tag_list,
        )

        return IngestResult(
            success=True,
            source=source_id,
            chunks=len(chunks),
            contextual_enriched=do_contextual,
        )

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

    # ── Stage 1: Query rewriting ─────────────────────────────────────────────
    rewritten = _rewrite_question(question, req.history)
    if rewritten != question:
        log.info("Rewritten: '%s' → '%s'", question, rewritten)

    # ── Stage 2: Hybrid retrieval ─────────────────────────────────────────────
    q_emb    = embed([rewritten])
    q_dense  = q_emb["dense"][0]
    q_sparse = q_emb["sparse"][0]

    if req.use_hybrid:
        candidates     = store.search_hybrid_candidates(
            dense_vector=q_dense,
            sparse_vector=q_sparse,
            fetch_k=req.fetch_k,
            meta=req.filters,
        )
        retrieval_mode = "hybrid"
    else:
        candidates     = store.search_candidates(
            query_vector=q_dense,
            fetch_k=req.fetch_k,
            meta=req.filters,
        )
        retrieval_mode = "dense"

    # ── Stage 3: Rerank ───────────────────────────────────────────────────────
    # Rerank on the original chunk text (not the expanded window) so the
    # cross-encoder stays focused on the matched content, not the neighbors.
    reranked = rerank(rewritten, candidates, top_k=req.top_k)

    # ── Stage 4: Window expansion ─────────────────────────────────────────────
    # After reranking, expand each chunk with its stored neighbors.
    # The LLM reads the expanded text; the UI shows the original match.
    if req.use_window_expansion:
        reranked = _expand_with_window(reranked)

    window_expanded = req.use_window_expansion and any(
        c.get("prev_chunk") or c.get("next_chunk") for c in reranked
    )

    # ── Stage 5: Generate ─────────────────────────────────────────────────────
    # Use expanded text for the LLM if available, else raw chunk text.
    llm_texts = [
        c.get("text_sent_to_llm") or c["text"]
        for c in reranked
    ]

    context_block = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(llm_texts))

    sources = []
    for c in reranked:
        if c["source"] and c["source"] not in sources:
            sources.append(c["source"])

    recent_history = req.history[-(MAX_HISTORY_TURNS):]
    history_block  = (
        "\n".join(f"{m.role.upper()}: {m.content}" for m in recent_history)
        if recent_history else ""
    )

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
        num_contexts=len(reranked),
        retrieval_mode=retrieval_mode,
        rewritten_question=rewritten if rewritten != question else None,
        window_expanded=window_expanded,
        matches=[
            MatchItem(
                text=c["text"],
                source=c["source"],
                rrf_score=c.get("rrf_score"),
                vector_score=c.get("vector_score"),
                rerank_score=c.get("rerank_score"),
                text_sent_to_llm=c.get("text_sent_to_llm"),
                was_enriched=bool(c.get("contextual_text")),
            )
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