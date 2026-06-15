# main.py

import os
import uuid
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from google import genai

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage

load_dotenv()

# -------------------------
# Configuration
# -------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="RAG PDF API",
    version="1.0.0",
)

store = QdrantStorage()


# -------------------------
# Request Model
# -------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


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

    result = store.search(
        question_vector,
        top_k=req.top_k,
    )

    contexts = result["contexts"]
    sources = result["sources"]

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

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    answer = ""

    if hasattr(response, "text") and response.text:
        answer = response.text.strip()

    return {
        "answer": answer,
        "sources": sources,
        "num_contexts": len(contexts),
    }