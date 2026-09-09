# Agent Instructions for RAG Chatbot

## Purpose
This repository is a small RAG chatbot project with a FastAPI backend, a Streamlit frontend, and a Qdrant vector store. It is designed for PDF ingestion, hybrid dense+sparse retrieval, reranking, and query answering.

## Key files
- `main.py` — FastAPI app exposing `/ingest` and `/query` endpoints.
- `streamlit_app.py` — Streamlit UI client that calls the backend API.
- `data_loader.py` — PDF ingestion, text chunking, and dense+sparse embedding with `FlagEmbedding` and `BAAI/bge-m3`.
- `vector_db.py` — Qdrant wrapper for storing dense + sparse vectors and metadata filters.
- `reranker.py` — cross-encoder reranker that re-scores retrieved candidates.
- `custom_types.py` — shared Pydantic models for API requests and responses.

## Important behavior
- Ingestion only accepts PDF files.
- Metadata fields (`category`, `author`, `year`, `tags`) are stored per chunk and can be filtered at query time.
- The backend uses hybrid retrieval by default: dense + sparse vectors, then reranks top candidates.
- Qdrant collection expects a dense vector size of `1024`. Changing this requires deleting the existing collection and re-ingesting.

## Environment variables
- `GEMINI_API_KEY` — required by `main.py` for the Google Gemini client.
- `QDRANT_URL` — optional Qdrant endpoint override, default is `http://localhost:6333`.
- `LOCAL_EMBED_MODEL` — optional local embedding model name, default is `BAAI/bge-m3`.
- `API_BASE` — optional Streamlit backend URL, default is `http://127.0.0.1:8000`.

## Common commands
- Run the API server: `uv run uvicorn main:app`
- Run the Streamlit UI: `uv run streamlit run streamlit_app.py`

## Notes for code changes
- Avoid altering the Qdrant vector schema unless the collection is recreated.
- Keep API models in `custom_types.py` so both FastAPI and clients remain consistent.
- Use `data_loader.py` for any new ingestion or embedding logic.

## Suggested next customization
Create a dedicated prompt or skill file for how to update retrieval behavior, metadata filters, and Qdrant schema safely in this repository.
