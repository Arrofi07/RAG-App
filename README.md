# Study-in-Germany AI Advisor  v8.0.0

> Agentic RAG chatbot with live web search, hybrid university recommendations,
> and persistent student profiles.  Built for MacBook M2 / 8 GB.

---

## What's new in v8

| Feature | Module | Description |
|---|---|---|
| **Agentic Planner** | `planner.py` | Classifies every question and routes it to the right tools before retrieval |
| **Web Search Tool** | `web_search_tool.py` | Fetches real-time deadlines, fees, and policy updates via Gemini Grounding |
| **University Recommender** | `university_recommender.py` | Hybrid SQL hard-filter + bge-m3 vector soft-match for personalised university suggestions |
| **University Finder UI** | `streamlit_app.py` | Dedicated page with match-score cards, deadlines, and research areas |
| **Intent Badge** | `streamlit_app.py` | Each answer shows which plan was used (RAG / Web / Finder) |

---

## Architecture

```
Student Question
       │
       ▼
 Query Rewriter          ← removes ambiguity using conversation history
       │
       ▼
 Intent Classifier       ← planner.py: rag_only / web_search / university_recommender
       │
  ┌────┴───────────────────────────────┐
  │                                    │
  ▼                                    ▼
Web Search Tool               University Recommender
(web_search_tool.py)          (university_recommender.py)
Gemini Grounding API          Stage 1: SQL hard-filter (city/language/budget/degree)
→ live deadlines, fees        Stage 2: bge-m3 vector soft-match (field similarity)
                              Stage 3: Score fusion + ranking
  │                                    │
  └──────────────┬─────────────────────┘
                 │
                 ▼
         RAG Retrieval            (skipped for recommender-only intent)
         (Qdrant hybrid)
         Dense + Sparse → RRF → Reranker → Window Expansion
                 │
                 ▼
        Augmented Context         ← planner.build_augmented_context()
        [RAG chunks] + ⟨WEB⟩ + University list
                 │
                 ▼
           Gemini LLM              ← gemini-2.5-flash-lite (cascade to larger models)
                 │
                 ▼
            Final Answer
```

---

## Memory budget (M2 / 8 GB)

| Component | RAM |
|---|---|
| `BAAI/bge-m3` (embedder) | ~2.3 GB |
| `BAAI/bge-reranker-v2-m3` | ~2.3 GB |
| Qdrant (in-process) | ~50 MB |
| FastAPI + Python | ~150 MB |
| OS overhead | ~1 GB |
| **Total** | **~5.8 GB** |

The recommender and web search tool add **zero steady-state RAM** — they
are pure Python + network calls.  The bge-m3 model used for recommendation
embeddings is the same instance already loaded for RAG.

---

## Setup

### 1. Prerequisites

```bash
# Qdrant (vector store) — run in a separate terminal
docker run -p 6333:6333 qdrant/qdrant

# Or install qdrant locally on macOS:
# brew install qdrant
```

### 2. Environment

```bash
cp .env.example .env
# Edit .env:
#   GEMINI_API_KEY=your_key_here
#   QDRANT_URL=http://localhost:6333   # default
```

`.env.example`:
```
GEMINI_API_KEY=
QDRANT_URL=http://localhost:6333
LOCAL_EMBED_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
GEMINI_MODELS=gemini-2.5-flash-lite,gemini-2.5-flash
LLM_MAX_RETRIES=2
```

### 3. Install dependencies

```bash
uv sync
# or: pip install -e .
```

First run downloads models (~4.6 GB one-time):
- `BAAI/bge-m3` → `~/.cache/huggingface/`
- `BAAI/bge-reranker-v2-m3` → same cache

### 4. Run

```bash
# Terminal 1: Backend API
uv run uvicorn main:app --reload

# Terminal 2: Frontend UI
uv run streamlit run streamlit_app.py
```

The backend startup sequence on first run:
1. Load bge-m3 embedding model (~30 s on M2)
2. Load bge-reranker model (~30 s on M2)
3. Seed university database (~5 s, one-time)
4. Server ready at http://localhost:8000

---

## New API endpoints (v8)

| Method | Path | Description |
|---|---|---|
| `POST` | `/recommend` | Direct university recommendation from profile |
| `POST` | `/seed-universities` | Re-seed the university DB (admin) |

### `POST /recommend`
```json
{
  "profile": {
    "field_of_study": "Computer Science",
    "target_degree": "Master's",
    "german_level": "None (complete beginner)",
    "budget_monthly_eur": 900,
    "target_cities": "Berlin, Munich"
  },
  "question": "strong AI research, English-taught",
  "top_k": 5
}
```

### `POST /query` (v8 additions)
The query request now accepts `force_intent` to bypass the planner:
```json
{
  "question": "What is the blocked account amount for 2026/27?",
  "force_intent": "web_search"
}
```

Query response now includes:
```json
{
  "intent": "web_search",
  "web_sources": [
    {"title": "DAAD — Proof of Financing", "url": "https://www.daad.de/..."}
  ]
}
```

---

## Adding more universities

Edit `SEED_UNIVERSITIES` in `university_recommender.py` and add new dicts
following the same structure.  Then call:

```bash
curl -X POST http://localhost:8000/seed-universities
```

Or load from a CSV by replacing `ensure_seeded()` with a CSV reader.

---

## Files

```
.
├── main.py                    # FastAPI app (v8 — orchestrates all modules)
├── planner.py                 # ★ NEW — agentic intent classifier + plan executor
├── web_search_tool.py         # ★ NEW — live web search via Gemini Grounding
├── university_recommender.py  # ★ NEW — hybrid SQL + vector university finder
├── streamlit_app.py           # UI (v8 — intent badge, finder page, web sources)
├── custom_types.py            # Pydantic models (v8 — UniversityMatch, WebSource)
├── advisor_config.py          # System prompt + profile field definitions
├── data_loader.py             # PDF chunking + bge-m3 embedding
├── vector_db.py               # Qdrant hybrid retrieval wrapper
├── reranker.py                # Cross-encoder reranker (bge-reranker-v2-m3)
├── context_builder.py         # Contextual chunk enrichment (Anthropic technique)
├── storage.py                 # SQLite user/conversation persistence
├── pyproject.toml             # Dependencies
└── data/
    └── chatbot.db             # SQLite: users + conversations + universities
```