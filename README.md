                    Document Upload
                           │
                           ▼
                  Document Processing
        ┌──────────────────────────────────┐
        │                                  │
        │ OCR (if needed)                  │
        │ Cleaning                         │
        │ Metadata Extraction              │
        │ Table Detection                  │
        │ Chunking                         │
        └──────────────────────────────────┘
                           │
                           ▼
                    Embedding Service
                           │
                           ▼
                    Vector Database
                           │
                           │
──────────────────────────QUERY──────────────────────────

User Question
       │
       ▼
Query Processing
       │
       ▼
Hybrid Retrieval
(Dense + BM25)
       │
       ▼
Top 30 Documents
       │
       ▼
Reranker
       │
       ▼
Top 5 Chunks
       │
       ▼
Prompt Builder
       │
       ▼
LLM
       │
       ▼
Answer Generator
       │
       ▼
Citation Generator
       │
       ▼
Confidence Scorer
       │
       ▼
Final Response

## Folder Structure
rag-system/

│

├── app/
│      main.py
│
├── api/
│      routes.py
│
├── ingestion/
│      loader.py
│      parser.py
│      cleaner.py
│      chunker.py
│
├── embedding/
│      embedder.py
│
├── retrieval/
│      vector_search.py
│      bm25.py
│      hybrid.py
│      reranker.py
│
├── prompting/
│      builder.py
│
├── generation/
│      llm.py
│
├── evaluation/
│      metrics.py
│
├── storage/
│      qdrant.py
│
├── configs/
│
├── tests/
│
└── frontend/

uv run streamlit run streamlit_app.py

uv run uvicorn main:app

npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery