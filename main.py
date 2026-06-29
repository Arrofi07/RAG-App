# main.py  (v8.0.0 — Agentic Planner + Web Search + University Recommender)
#
# WHAT CHANGED FROM v7 → v8
# ──────────────────────────
# 1. AGENTIC PLANNER (planner.py)
#    The query no longer goes straight to retrieval.  The planner first
#    classifies the question intent (rag_only / web_search /
#    university_recommender / recommend_and_search) and assembles the
#    right set of tools.
#
# 2. WEB SEARCH TOOL (web_search_tool.py)
#    Time-sensitive questions (deadlines, fees, blocked-account amounts)
#    trigger a live Google Search via Gemini Grounding before the LLM
#    writes its answer.
#
# 3. UNIVERSITY RECOMMENDER (university_recommender.py)
#    Questions like "which university suits me?" trigger the hybrid
#    recommender: SQL hard-filters (city, language, degree, budget) +
#    vector soft-match (field similarity).
#
# 4. NEW ENDPOINTS
#    POST /recommend     — call the recommender directly (for testing/UI)
#    POST /seed-universities  — re-seed the university DB (admin utility)
#
# MEMORY BUDGET (M2 / 8 GB)
# ──────────────────────────
# Model            RAM
# bge-m3           ~2.3 GB   (embedder, loaded once at startup)
# bge-reranker     ~2.3 GB   (reranker, loaded once at startup)
# Qdrant client    ~50 MB    (in-process, no separate server needed)
# FastAPI + rest   ~150 MB
# OS + Python      ~1 GB
# ─────────────────────────
# Total            ~5.8 GB   (leaves ~2.2 GB free for OS caching)
#
# The recommender and web search tool add ~0 MB steady-state RAM —
# they are pure logic + network calls.

import os
import uuid
import logging
import tempfile
from typing import Optional, Any
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from google import genai

# ── Local modules (existing) ───────────────────────────────────────────────────
from data_loader import load_chunks, embed, EMBED_DIM, _get_embed_model
from vector_db import QdrantStorage
from reranker import rerank, warmup as warmup_reranker
from context_builder import build_contextual_text
from storage import Storage
from advisor_config import build_system_prompt
from custom_types import (
    ChatMessage, MetaFilter, IngestResult, DocumentListResult,
    QueryResult, MatchItem, ConversationMeta, UserInfo, UserProfile,
)

# ── New modules (v8) ───────────────────────────────────────────────────────────
from planner import classify_intent, execute_plan, build_augmented_context
from university_recommender import UniversityRecommender

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Boot — initialise singletons
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set.")

# Gemini client for LLM generation and web search grounding
client = genai.Client(api_key=GEMINI_API_KEY)

app   = FastAPI(title="Study-in-Germany AI Advisor API", version="8.0.0")
store = QdrantStorage(dim=EMBED_DIM)   # Qdrant vector store (existing)
db    = Storage()                       # SQLite user/conversation store (existing)

# University recommender — new in v8
# Initialised here so it shares the same DB file as Storage
recommender = UniversityRecommender(db_path=Path("data/chatbot.db"))

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))

# Model cascade — fastest/cheapest first, fall back if rate-limited
GEMINI_MODELS = (
    os.getenv(
        "GEMINI_MODELS",
        "gemini-2.5-flash-lite,gemini-2.5-flash,gemini-3.5-flash,gemini-3.1-flash-lite",
    ).split(",")
)

LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_503_BACKOFF = float(os.getenv("LLM_503_BACKOFF", "5"))


# ─────────────────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:             str
    history:              list[ChatMessage] = []
    user_id:              Optional[str]     = None
    conversation_id:      Optional[str]     = None
    top_k:                int               = 5
    fetch_k:              int               = 30
    use_hybrid:           bool              = True
    use_window_expansion: bool              = True
    filters:              MetaFilter        = MetaFilter()
    # New v8: allow the frontend to skip the planner and force a mode
    force_intent:         Optional[str]     = None  # "rag_only", "web_search", etc.


class RecommendRequest(BaseModel):
    """Direct call to the university recommender (for testing or the UI)."""
    profile:   dict[str, Any]
    question:  str            = ""
    top_k:     int            = 5


class CreateUserRequest(BaseModel):
    name:    str
    profile: dict[str, Any] = {}


class UpdateProfileRequest(BaseModel):
    profile: dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# LLM helper (unchanged from v7)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_retry_delay(exc: Exception) -> float:
    """Extract server-suggested retry delay from a Gemini 429 error string."""
    import re
    try:
        m = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", str(exc))
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return 30.0


def _error_code(exc: Exception) -> int | None:
    """Extract HTTP status code from a google-genai exception."""
    import re
    try:
        m = re.match(r"^(\d{3})\s", str(exc))
        if m:
            return int(m.group(1))
        code = getattr(exc, "code", None)
        if isinstance(code, int):
            return code
    except Exception:
        pass
    return None


def _call_llm(prompt: str, unlimited_retries: bool = False) -> str:
    """
    Call the LLM with automatic retry and model cascade.

    Args:
        prompt           : the prompt to send
        unlimited_retries: if True, retry 429/503 across ALL models in a round-
                           robin loop until success (used during ingestion).
                           If False, give up after LLM_MAX_RETRIES per model.

    Retry policy (per-model):
        429 → sleep max(retryDelay, MIN_429_DELAY) seconds, then retry.
              After LLM_429_PER_MODEL_MAX consecutive 429s on one model,
              move to the next model even in unlimited mode.
        503 → sleep LLM_503_BACKOFF seconds, retry up to LLM_MAX_RETRIES.
        Other → move to next model immediately (non-transient error).

    In unlimited mode the outer loop restarts from the first model once all
    models have been tried, so we never give up on transient quota errors.

    TWO BUGS FIXED vs PREVIOUS VERSION
    ────────────────────────────────────
    Bug 1: Gemini sometimes returns retryDelay=0, causing instant re-fire.
           Fix: enforce a minimum delay of MIN_429_DELAY seconds.
    Bug 2: unlimited_retries looped forever on the SAME model, never cascading.
           Fix: after LLM_429_PER_MODEL_MAX 429s on one model, move to the next.
    """
    import time

    # Minimum seconds to wait after a 429, even if Gemini says 0.
    # Free-tier flash-lite = 15 RPM → one call per 4 s minimum.
    # Using 5 s gives a small safety buffer.
    MIN_429_DELAY = float(os.getenv("LLM_MIN_429_DELAY", "5"))

    # How many consecutive 429s on one model before we cascade to the next,
    # even in unlimited mode. Prevents getting stuck on a fully-exhausted quota.
    LLM_429_PER_MODEL_MAX = int(os.getenv("LLM_429_PER_MODEL_MAX", "3"))

    last_exc: Exception | None = None

    def _try_models_once() -> str | None:
        """
        Attempt each model once (with its own retry budget).
        Returns the answer string on success, or None if all models fail.
        Sets last_exc as a side-effect via nonlocal.
        """
        nonlocal last_exc

        for model_name in GEMINI_MODELS:
            consecutive_429s = 0  # reset counter for each model

            for attempt in range(1, LLM_MAX_RETRIES + 2):
                try:
                    resp = client.models.generate_content(model=model_name, contents=prompt)
                    if hasattr(resp, "text") and resp.text:
                        if attempt > 1:
                            log.info("%s succeeded on attempt %d", model_name, attempt)
                        return resp.text.strip()
                    # Empty response — treat as transient and retry once
                    log.warning("%s: empty response (attempt %d)", model_name, attempt)

                except Exception as exc:
                    last_exc  = exc
                    http_code = _error_code(exc)

                    if http_code == 429:
                        consecutive_429s += 1
                        # Raw delay from Gemini header — enforce minimum
                        raw_delay = _parse_retry_delay(exc)
                        delay     = max(raw_delay, MIN_429_DELAY)
                        if delay != raw_delay:
                            log.debug(
                                "retryDelay %.1fs is below minimum; using %.1fs",
                                raw_delay, delay,
                            )

                        # Move to next model if we've hit too many 429s here
                        if consecutive_429s >= LLM_429_PER_MODEL_MAX:
                            log.warning(
                                "%s — %d consecutive 429s, switching model. (last delay %.1fs)",
                                model_name, consecutive_429s, delay,
                            )
                            # Still sleep the delay before trying the next model
                            # so we don't immediately hammer a fresh quota.
                            time.sleep(delay)
                            break  # → next model in outer for-loop

                        log.warning(
                            "%s — 429 (attempt %d/%d). Waiting %.1fs…",
                            model_name, attempt,
                            LLM_MAX_RETRIES + 1, delay,
                        )
                        time.sleep(delay)
                        continue  # retry same model

                    elif http_code == 503:
                        if attempt <= LLM_MAX_RETRIES:
                            log.warning(
                                "%s — 503 overloaded (attempt %d/%d). Waiting %.1fs…",
                                model_name, attempt, LLM_MAX_RETRIES + 1, LLM_503_BACKOFF,
                            )
                            time.sleep(LLM_503_BACKOFF)
                            continue
                        log.warning("%s — 503 exhausted. Next model.", model_name)
                        break

                    else:
                        # Non-transient (bad request, auth error, etc.)
                        log.warning("%s — error %s: %s", model_name, http_code, exc)
                        break  # → next model

        return None  # all models failed this round

    if unlimited_retries:
        # Keep cycling through all models until one succeeds.
        # This is used during ingestion so a PDF is never aborted by quota.
        round_num = 0
        while True:
            round_num += 1
            result = _try_models_once()
            if result is not None:
                return result
            # All models rate-limited — wait before starting another round
            pause = MIN_429_DELAY * 2
            log.warning(
                "All models rate-limited (round %d). Waiting %.1fs before retry…",
                round_num, pause,
            )
            time.sleep(pause)
    else:
        result = _try_models_once()
        if result is not None:
            return result

    raise HTTPException(
        status_code=503,
        detail=f"All LLM backends unavailable. Last error: {last_exc}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Query rewriting (unchanged from v7)
# ─────────────────────────────────────────────────────────────────────────────

def _rewrite_question(question: str, history: list[ChatMessage]) -> str:
    """Rewrite an ambiguous question into a self-contained query using conversation history."""
    if not history:
        return question

    recent     = history[-(MAX_HISTORY_TURNS):]
    transcript = "\n".join(f"{m.role.upper()}: {m.content}" for m in recent)

    prompt = f"""You are a query rewriter for a RAG system.
Rewrite the user's latest message into a single, fully self-contained question.
Rules:
- Output ONLY the rewritten question. No preamble.
- If already self-contained, return it unchanged.
- Preserve the original language.

Conversation history:
{transcript}

Latest message: {question}

Rewritten question:"""

    try:
        return _call_llm(prompt)
    except Exception:
        log.warning("Query rewriting failed, using original.")
        return question


# ─────────────────────────────────────────────────────────────────────────────
# Window expansion (unchanged from v7)
# ─────────────────────────────────────────────────────────────────────────────

def _expand_with_window(candidates: list[dict]) -> list[dict]:
    """Prepend/append neighbouring chunks to each retrieved chunk."""
    for c in candidates:
        parts = [p for p in [c.get("prev_chunk", ""), c["text"], c.get("next_chunk", "")] if p]
        c["text_sent_to_llm"] = "\n\n".join(parts)
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
def _startup() -> None:
    """
    Pre-load ML models and seed the university database on server start.

    We do this at startup rather than on first request so that:
      1. Cold-start latency is paid once (when you run the server), not
         when the first student asks a question.
      2. The university DB is always ready — no first-query lag.
    """
    # Pre-load the embedding model (~2.3 GB into RAM)
    log.info("Pre-loading embedding model…")
    _get_embed_model()

    # Pre-load the reranker model (~2.3 GB into RAM)
    log.info("Pre-loading reranker…")
    warmup_reranker()

    # Seed the university database (fast if already seeded)
    # We pass embed_fn so embeddings are computed on first seed.
    # On M2 / 8 GB: seeding ~30 universities takes ~5 s (embedding is batched).
    log.info("Ensuring university database is seeded…")
    recommender.ensure_seeded(embed_fn=embed)

    log.info("✅ All models and data ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status":  "running",
        "version": "8.0.0 — agentic planner + web search + university recommender",
    }


# ─────────────────────────────────────────────────────────────────────────────
# User / profile endpoints (unchanged from v7)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/users", response_model=UserInfo)
def create_user(req: CreateUserRequest):
    uid  = db.create_user(name=req.name, profile=req.profile)
    user = db.get_user(uid)
    return UserInfo(**{**user, "profile": user["profile"]})


@app.get("/users", response_model=list[UserInfo])
def list_users():
    users = db.list_users()
    result = []
    for u in users:
        full = db.get_user(u["id"])
        if full:
            result.append(UserInfo(**full))
    return result


@app.get("/users/{user_id}", response_model=UserInfo)
def get_user(user_id: str):
    user = db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return UserInfo(**user)


@app.put("/users/{user_id}/profile")
def update_profile(user_id: str, req: UpdateProfileRequest):
    if not db.get_user(user_id):
        raise HTTPException(status_code=404, detail="User not found.")
    db.update_profile(user_id, req.profile)
    return {"success": True}


# ─────────────────────────────────────────────────────────────────────────────
# Conversation endpoints (unchanged from v7)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/users/{user_id}/conversations")
def create_conversation(user_id: str, title: str = "New conversation"):
    if not db.get_user(user_id):
        raise HTTPException(status_code=404, detail="User not found.")
    cid = db.create_conversation(user_id, title)
    return {"conversation_id": cid}


@app.get("/users/{user_id}/conversations", response_model=list[ConversationMeta])
def list_conversations(user_id: str):
    return db.list_conversations(user_id)


@app.get("/conversations/{conversation_id}/messages")
def get_messages(conversation_id: str):
    return {"messages": db.get_messages(conversation_id)}


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    db.delete_conversation(conversation_id)
    return {"success": True}


# ─────────────────────────────────────────────────────────────────────────────
# Ingest PDF (unchanged from v7)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResult)
async def ingest_pdf(
    file:           UploadFile    = File(...),
    category:       Optional[str] = Form(None),
    author:         Optional[str] = Form(None),
    year:           Optional[int] = Form(None),
    tags:           Optional[str] = Form(None),
    use_contextual: str           = Form("false"),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    do_contextual = use_contextual.lower() in ("true", "1", "yes")
    tag_list      = ([t.strip() for t in tags.split(",") if t.strip()] if tags else [])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        pdf_path = tmp.name

    try:
        loaded    = load_chunks(pdf_path)
        chunks    = loaded["chunks"]
        full_text = loaded["full_text"]
        source_id = file.filename

        if not chunks:
            raise HTTPException(status_code=400, detail="No text found in PDF.")

        texts_to_embed   = []
        contextual_texts = []

        if do_contextual:
            # ── Contextual enrichment with rate-limit protection ─────────────
            #
            # WHY THIS IS NEEDED
            # ──────────────────
            # Gemini free tier limits are tight:
            #   gemini-2.5-flash-lite : 15 RPM (= 1 call every 4 seconds)
            #   gemini-2.5-flash      : 10 RPM (= 1 call every 6 seconds)
            # Firing one LLM call per chunk with no delay saturates the quota
            # within seconds on any PDF with more than ~3 pages.
            #
            # THREE-LAYER FIX
            # ───────────────
            # 1. BATCH: send up to CONTEXT_BATCH_SIZE chunks in a SINGLE LLM
            #    call, asking the model to return one context per chunk as a
            #    numbered list.  This cuts total calls by ~5-10×.
            # 2. INTER-BATCH DELAY: sleep CONTEXT_INTER_BATCH_DELAY seconds
            #    between batches to stay under the RPM ceiling.
            # 3. UNLIMITED RETRIES: pass unlimited_retries=True to _call_llm
            #    so a 429 during ingestion is always honoured and retried (with
            #    the server-suggested delay) rather than aborting mid-PDF.
            #
            # TUNING (edit via environment variables)
            # ─────────────────────────────────────────
            # CONTEXT_BATCH_SIZE         default 5  — chunks per LLM call
            #   Larger → fewer calls but longer prompts. Stay ≤ 8 for reliability.
            # CONTEXT_INTER_BATCH_DELAY  default 5  — seconds between batches
            #   Set to 6+ if you still see 429s with the default model.
            #   Set to 0 if you have a paid Gemini plan with higher RPM limits.

            CONTEXT_BATCH_SIZE        = int(os.getenv("CONTEXT_BATCH_SIZE", "5"))
            CONTEXT_INTER_BATCH_DELAY = float(os.getenv("CONTEXT_INTER_BATCH_DELAY", "5"))

            # Truncate document preview (same as context_builder.py uses)
            DOC_PREVIEW_CHARS = 3000
            doc_preview = full_text[:DOC_PREVIEW_CHARS]

            # Pre-fill with empty strings so the list is always len(chunks)
            contextual_texts = [""] * len(chunks)

            # Process chunks in batches
            for batch_start in range(0, len(chunks), CONTEXT_BATCH_SIZE):
                batch_end   = min(batch_start + CONTEXT_BATCH_SIZE, len(chunks))
                batch       = chunks[batch_start:batch_end]
                batch_size  = len(batch)

                log.info(
                    "Contextual enrichment: chunks %d–%d / %d  (batch of %d)",
                    batch_start + 1, batch_end, len(chunks), batch_size,
                )

                # Build a single prompt that asks for ALL chunks in this batch.
                # The model returns a numbered list "1. <context>\n2. <context>…"
                chunk_sections = "\n\n".join(
                    f"CHUNK {j + 1}:\n{batch[j]['text']}"
                    for j in range(batch_size)
                )

                batch_prompt = f"""You are indexing document chunks for a retrieval system.

Document name: {source_id}

Document preview:
<document>
{doc_preview}
</document>

Below are {batch_size} chunk(s) from this document.
For EACH chunk write ONE short sentence (max 20 words) that situates the chunk
within the document.  Output ONLY a numbered list in this exact format:

1. <context for chunk 1>
2. <context for chunk 2>
...

Rules:
- One line per chunk. No preamble or explanation.
- Do NOT repeat the chunk text — just contextualise it.
- Be factual and concise.

{chunk_sections}

Numbered context list:"""

                try:
                    # unlimited_retries=True means a 429 during ingestion will
                    # always be honoured and retried, never aborting the PDF.
                    raw = _call_llm(batch_prompt, unlimited_retries=True)

                    # Parse the numbered list response.
                    # Lines look like "1. This chunk covers admission requirements…"
                    # We strip the leading "N. " prefix and store the rest.
                    import re as _re
                    parsed = {}
                    for line in raw.splitlines():
                        line = line.strip()
                        m = _re.match(r"^(\d+)\.\s+(.+)", line)
                        if m:
                            idx = int(m.group(1)) - 1   # 0-based
                            if 0 <= idx < batch_size:
                                parsed[idx] = m.group(2).strip()

                    # Store results; fall back to "" for any chunk the LLM missed
                    for j in range(batch_size):
                        contextual_texts[batch_start + j] = parsed.get(j, "")

                except Exception as exc:
                    # If the whole batch fails, log and continue without context
                    # for this batch — we never abort ingestion over enrichment.
                    log.warning(
                        "Batch context generation failed for chunks %d–%d: %s — skipping.",
                        batch_start + 1, batch_end, exc,
                    )

                # Polite inter-batch pause to stay under RPM ceiling.
                # Skip delay after the last batch (no next call needed).
                if batch_end < len(chunks):
                    log.info(
                        "Rate-limit pause: sleeping %.1fs before next batch…",
                        CONTEXT_INTER_BATCH_DELAY,
                    )
                    import time as _time
                    _time.sleep(CONTEXT_INTER_BATCH_DELAY)

            # Build embed texts: prepend context to chunk text where available
            for i, chunk in enumerate(chunks):
                ctx      = contextual_texts[i]
                enriched = build_contextual_text(ctx, chunk["text"])
                texts_to_embed.append(enriched)

        else:
            # No contextual enrichment — embed raw chunks directly
            contextual_texts = [""] * len(chunks)
            for chunk in chunks:
                texts_to_embed.append(chunk["text"])

        embeddings = embed(texts_to_embed)

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]

        payloads = [
            {
                "source":          source_id,
                "text":            chunks[i]["text"],
                "prev_chunk":      chunks[i]["prev_chunk"],
                "next_chunk":      chunks[i]["next_chunk"],
                "chunk_index":     chunks[i]["chunk_index"],
                "page":            chunks[i]["page"],
                "filename":        source_id,
                "document_type":   "pdf",
                "uploaded_at":     datetime.utcnow().isoformat(),
                "category":        category,
                "author":          author,
                "year":            year,
                "tags":            tag_list,
                "contextual_text": contextual_texts[i] or None,
                "was_enriched":    do_contextual and bool(contextual_texts[i]),
            }
            for i in range(len(chunks))
        ]

        store.upsert(ids=ids, dense_vectors=embeddings["dense"],
                     sparse_vectors=embeddings["sparse"], payloads=payloads)

        log.info("Ingested %d chunks from '%s' | contextual=%s",
                 len(chunks), source_id, do_contextual)

        return IngestResult(success=True, source=source_id, chunks=len(chunks),
                            contextual_enriched=do_contextual)
    finally:
        Path(pdf_path).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Query — v8: now runs through the agentic planner
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResult)
def query(req: QueryRequest):
    """
    Main query endpoint.  Now orchestrated by the agentic planner.

    Flow:
    1. Rewrite question       (same as v7)
    2. Classify intent        (NEW v8 — planner decides which tools to use)
    3. Execute plan           (NEW v8 — may trigger web search or recommender)
    4. RAG retrieval + rerank (same as v7, skipped for recommender-only intent)
    5. Build augmented context (NEW v8 — merge RAG + web + recommendations)
    6. Generate answer        (same as v7, with richer context block)
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Load user profile for personalised system prompt and planner context
    profile: dict = {}
    if req.user_id:
        user = db.get_user(req.user_id)
        if user:
            profile = user.get("profile", {})

    # Persist incoming user message
    if req.conversation_id:
        db.add_message(req.conversation_id, "user", question)

    # ── Build history ──────────────────────────────────────────────────────────
    if req.conversation_id:
        db_msgs = db.get_messages(req.conversation_id)
        history = [ChatMessage(role=m["role"], content=m["content"]) for m in db_msgs[:-1]]
    else:
        history = req.history

    # ── Stage 1: Query rewriting ───────────────────────────────────────────────
    rewritten = _rewrite_question(question, history)
    if rewritten != question:
        log.info("Rewritten: '%s' → '%s'", question, rewritten)

    # ── Stage 2: Intent classification (NEW v8) ────────────────────────────────
    # The force_intent field lets the frontend bypass classification for
    # specific UI actions (e.g. a "Find Universities" button always forces
    # the recommender intent regardless of the question text).
    intent = req.force_intent or classify_intent(rewritten, _call_llm)
    log.info("Intent: %s", intent)

    # ── Stage 3: Execute plan (NEW v8) ────────────────────────────────────────
    # The recommender is wired in here via a lambda so university_recommender.py
    # never imports main.py (avoids circular imports).
    def _recommend(profile, question):
        return recommender.recommend(
            profile=profile,
            question=question,
            embed_fn=embed,   # reuse the already-loaded bge-m3 model
            top_k=5,
        )

    plan_result = execute_plan(
        question=rewritten,
        profile=profile,
        intent=intent,
        call_llm=_call_llm,
        recommender_fn=_recommend,
    )

    # ── Stage 4: RAG retrieval + rerank ───────────────────────────────────────
    # Skip RAG for pure recommender intent — the recommendations ARE the context.
    rag_chunks     = []
    retrieval_mode = "none"
    window_expanded = False

    if plan_result.intent in ("rag_only", "web_search"):
        q_emb    = embed([rewritten])
        q_dense  = q_emb["dense"][0]
        q_sparse = q_emb["sparse"][0]

        if req.use_hybrid:
            candidates     = store.search_hybrid_candidates(q_dense, q_sparse,
                                                            fetch_k=req.fetch_k,
                                                            meta=req.filters)
            retrieval_mode = "hybrid"
        else:
            candidates     = store.search_candidates(q_dense, fetch_k=req.fetch_k,
                                                     meta=req.filters)
            retrieval_mode = "dense"

        reranked = rerank(rewritten, candidates, top_k=req.top_k)

        if req.use_window_expansion:
            reranked = _expand_with_window(reranked)
            window_expanded = any(c.get("prev_chunk") or c.get("next_chunk")
                                  for c in reranked)

        rag_chunks = reranked

    # ── Stage 5: Build augmented context (NEW v8) ─────────────────────────────
    # Merges RAG chunks + live web data + recommendations into one context block.
    context_block = build_augmented_context(rag_chunks, plan_result)

    # ── Stage 6: Generate answer ───────────────────────────────────────────────
    sources = []
    for c in rag_chunks:
        if c["source"] and c["source"] not in sources:
            sources.append(c["source"])

    recent_history = history[-(MAX_HISTORY_TURNS):]
    history_block  = (
        "\n".join(f"{m.role.upper()}: {m.content}" for m in recent_history)
        if recent_history else ""
    )

    system_prompt = build_system_prompt({**profile, "name": profile.get("name", "")})

    # The prompt now includes a "tool usage guide" so the LLM knows how to
    # cite web results (⟨WEB⟩) vs RAG chunks ([1]) vs recommendations.
    prompt = f"""{system_prompt}

━━━ CONTEXT (assembled by the agentic planner) ━━━

{context_block}

━━━ INSTRUCTIONS ━━━
- Answer using the context above AND your advisor knowledge.
- For RAG chunks, cite as [1], [2], etc.
- For live web data (marked ⟨WEB⟩), say "According to current web sources…"
- For university recommendations, present them as a numbered list with key facts.
- If the context is insufficient, say so and direct the student to the official source.
- Be warm, specific, and actionable.
{("━━━ CONVERSATION SO FAR ━━━" + chr(10) + history_block) if history_block else ""}

User: {question}

Advisor:"""

    answer = _call_llm(prompt)

    # Persist assistant answer
    if req.conversation_id:
        db.add_message(req.conversation_id, "assistant", answer)

        # Auto-title from first user message
        conv = db.get_conversation(req.conversation_id)
        if conv and conv["title"] == "New conversation":
            title = question[:60] + ("…" if len(question) > 60 else "")
            db.update_conversation_title(req.conversation_id, title)

    # Build the response — include new v8 fields in the match items
    return QueryResult(
        answer=answer,
        sources=sources,
        num_contexts=len(rag_chunks),
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
            for c in rag_chunks
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# University Recommender endpoint (NEW v8)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/recommend")
def recommend(req: RecommendRequest):
    """
    Direct university recommendation endpoint.

    Use this from the UI for a dedicated "Find Universities" page, or
    from the query endpoint when intent = 'university_recommender'.

    Request body:
        profile  : student profile dict (same format as /users/{id}/profile)
        question : optional free-text query to improve vector matching
        top_k    : number of results (default 5)

    Returns:
        {
            "universities": [...],   # ranked list of matches
            "total_filtered": int,   # how many passed SQL hard-filter
        }
    """
    result = recommender.recommend(
        profile=req.profile,
        question=req.question,
        embed_fn=embed,   # reuse the already-loaded bge-m3
        top_k=req.top_k,
    )
    return result


@app.post("/seed-universities")
def seed_universities():
    """
    (Admin endpoint) Re-seed the university database and recompute embeddings.
    Safe to call multiple times — it only inserts if the table is empty.
    To force a full re-seed, delete the universities table first via SQLite.
    """
    recommender.ensure_seeded(embed_fn=embed)
    n = recommender.compute_missing_embeddings(embed_fn=embed)
    return {"success": True, "newly_embedded": n}


# ─────────────────────────────────────────────────────────────────────────────
# Documents endpoint (unchanged from v7)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/documents", response_model=DocumentListResult)
def list_documents():
    return DocumentListResult(documents=[d["filename"] for d in store.list_documents()])


@app.get("/documents/metadata")
def list_documents_metadata():
    return {"documents": store.list_documents()}