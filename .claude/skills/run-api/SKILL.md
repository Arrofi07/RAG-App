---
description: Launch the FastAPI backend for the Study-in-Germany RAG chatbot locally (starts Docker/Qdrant if needed, waits for bge-m3/reranker model load, confirms the server is reachable). Use for "run the app", "start the server", "smoke test the API".
---

# Running the RAG chatbot API locally

`scripts/dev_up.sh` and `scripts/dev_down.sh` wrap the verified sequence
for getting `main:app` serving on `localhost:8000` (macOS, models already
cached under `~/.cache/huggingface`). Prefer them over manual steps —
they're idempotent (skip anything already running) and this file exists
mainly to explain *why* each step is there.

```bash
cd "/Users/apple/Documents/Projects/RAG Chatbot"
./scripts/dev_up.sh       # Qdrant + API server, waits for readiness
./scripts/smoke_test.sh   # full auth/admin-gating check, 7 assertions
./scripts/dev_down.sh     # stop server, remove the dev Qdrant container
```

No Docker Compose needed for this — Compose (`docker-compose.yml`) is
for the full multi-container demo deployment (Qdrant + Ollama + api +
ui) once that's what you're testing instead.

## Why each step exists (for when something goes wrong)

**Qdrant must be reachable before the app can even start**, not just
before retrieval works — `QdrantStorage.__init__` connects at import
time in `main.py`, so a missing Qdrant fails the whole server, not just
degrades it. `dev_up.sh` checks `http://localhost:6333/healthz`, and if
Docker Desktop itself isn't running yet it launches it and waits for the
daemon before starting the `rag_qdrant_dev` container (named distinctly
from docker-compose's `rag_qdrant` so the two don't collide).

**Model loading takes ~30-90s**, sometimes longer if HuggingFace Hub
metadata checks are slow even with a warm local cache. `dev_up.sh` polls
`http://localhost:8000/` rather than guessing a sleep duration. If it's
taking much longer than that, tail `/tmp/rag_api_server.log` — a stuck
HF Hub lookup or a missing `.env` var are the usual causes.

**The smoke test needs a real running server**, not mocks — running it
cold (without `dev_up.sh` first) just hangs on the first `curl` call
with no useful error, since the script doesn't start anything itself.

## Notes

- `.env` in the repo root already has `GEMINI_API_KEY`, `JWT_SECRET_KEY`,
  etc. populated for local dev — no setup needed there.
- The university DB seed is idempotent (`ensure_seeded` only inserts if
  empty), so re-running `/seed-universities` repeatedly is safe.
- `dev_down.sh` leaves Docker Desktop itself running — quitting it isn't
  necessary and may interrupt other work.
