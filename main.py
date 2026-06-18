# main.py

import os
import uuid
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from google import genai

from data_loader import load_and_chunk_pdf, embed_texts, get_embed_dim
from vector_db import QdrantStorage
from reranker import rerank

from datetime import datetime

load_dotenv()

# -------------------------
# Configuration
# -------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBED_DIM = get_embed_dim()

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="RAG PDF API",
    version="1.0.0",
)

store = QdrantStorage(dim=EMBED_DIM)


# -------------------------
# Request Model
# -------------------------

from typing import Optional

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    fetch_k: int = 30
    filename: Optional[str] = None


# -------------------------
# Health Check
# -------------------------

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "RAG API is ready."
    }


# -------------------------
# Ingest PDF
# -------------------------

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:
        contents = await file.read()
        tmp.write(contents)
        pdf_path = tmp.name

    try:
        chunks = load_and_chunk_pdf(pdf_path)

        if len(chunks) == 0:
            raise HTTPException(
                status_code=400,
                detail="No text found in PDF."
            )

        vectors = embed_texts(chunks)

        source_id = file.filename

        ids = [
            str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{source_id}:{i}"
                )
            )
            for i in range(len(chunks))
        ]

        payloads = [
            {
                "source": source_id,
                "text": chunks[i],
                "filename": source_id,
                "uploaded_at": datetime.utcnow().isoformat(),
                "document_type": "pdf",
                "chunk_id": i,
            }
            for i in range(len(chunks))
        ]

        store.upsert(
            ids=ids,
            vectors=vectors,
            payloads=payloads,
        )

        return {
            "success": True,
            "source": source_id,
            "chunks": len(chunks),
        }

    finally:
        Path(pdf_path).unlink(missing_ok=True)


# -------------------------
# Query
# -------------------------

@app.post("/query")
def query(req: QueryRequest):
    question = req.question.strip()

    if question == "":
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    question_vector = embed_texts([question])[0]

    # Stage 1: cheap, wide dense retrieval (e.g. top 30 candidates).
    # Vector search is fast but approximate — it compares the query's
    # embedding to each chunk's embedding independently, so it sometimes
    # ranks loosely-related chunks above more relevant ones.
    candidates = store.search_candidates(
        query_vector=question_vector,
        fetch_k=req.fetch_k,
        filename=req.filename,
    )

    # Stage 2: precise, narrow reranking down to top_k.
    # The cross-encoder reads (question, chunk) together, so it catches
    # relevance that the vector search alone misses.
    reranked = rerank(question, candidates, top_k=req.top_k)

    contexts = [c["text"] for c in reranked]

    sources = []
    for c in reranked:
        if c["source"] and c["source"] not in sources:
            sources.append(c["source"])

    context_block = "\n\n".join(
        f"- {c}" for c in contexts
    )

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.
If the answer is not contained in the context,
say that you don't know.

Context:

{context_block}

Question:

{question}
"""
    print("=" * 80)
    print("QUESTION:")
    print(question)

    print("\nNUM CANDIDATES:", len(candidates))
    print("NUM RERANKED:", len(reranked))

    print("\nSOURCES:")
    print(sources)

    print("\nPROMPT LENGTH:")
    print(len(prompt))
    print("=" * 80)

    MODELS = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]

    response = None

    for model_name in MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            break
        except Exception as e:
            print(f"{model_name} failed: {e}")

    if response is None:
        raise HTTPException(
            status_code=503,
            detail="All LLM backends unavailable."
        )

    answer = ""

    if hasattr(response, "text") and response.text:
        answer = response.text.strip()

    return {
        "answer": answer,
        "sources": sources,
        "num_contexts": len(contexts),
        "matches": [
            {
                "text": c["text"],
                "source": c["source"],
                "vector_score": c.get("vector_score"),
                "rerank_score": c.get("rerank_score"),
            }
            for c in reranked
        ],
    }