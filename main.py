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

# main.py  (v9.0.0 — Multi-provider LLM routing)
#
# WHAT CHANGED FROM v8 → v9
# ──────────────────────────
# The single Gemini `_call_llm()` function is replaced by a provider registry
# (llm_providers.py) that routes each task to the right model:
#
#   role="enrichment" → local Ollama (Qwen3:1.7b) — NO Gemini quota used
#   role="planning"   → local Ollama (Qwen3:1.7b) — intent, rewrite, search queries
#   role="answer"     → Gemini cascade             — final student-facing answer
#
# All three roles are swappable via .env without touching this file.
# See llm_providers.py for the full configuration reference.

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

# ── Local modules ──────────────────────────────────────────────────────────────
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
from planner import classify_intent, execute_plan, build_augmented_context
from university_recommender import UniversityRecommender

# ── Provider registry (v9) ─────────────────────────────────────────────────────
# build_registry() reads .env and constructs one LLMProvider per role.
# call_llm(prompt, role=...) is the ONLY LLM call site used everywhere below.
from llm_providers import build_registry, call_llm, check_providers, get_provider

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Boot — initialise singletons
# ─────────────────────────────────────────────────────────────────────────────

app   = FastAPI(title="Study-in-Germany AI Advisor API", version="9.0.0")
store = QdrantStorage(dim=EMBED_DIM)
db    = Storage()
recommender = UniversityRecommender(db_path=Path("data/chatbot.db"))

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))


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
# LLM helper — v9: thin wrapper around the provider registry
# ─────────────────────────────────────────────────────────────────────────────
#
# All the retry/cascade/429-handling logic now lives in llm_providers.py
# (inside GeminiProvider, OllamaProvider, OpenAICompatibleProvider).
# This wrapper just picks the right ROLE for each call site:
#
#   _llm_answer(prompt)      → role="answer"     (Gemini, student-facing)
#   _llm_planning(prompt)    → role="planning"    (local Qwen, fast/free)
#   _llm_enrichment(prompt)  → role="enrichment"  (local Qwen, ingestion)

def _llm_answer(prompt: str) -> str:
    """Generate the final, student-facing answer. Uses the 'answer' provider (default: Gemini)."""
    try:
        return call_llm(prompt, role="answer")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Answer generation failed: {exc}")


def _llm_planning(prompt: str) -> str:
    """
    Lightweight LLM calls for query rewriting, intent classification, and
    search query construction. Uses the 'planning' provider (default: local
    Ollama/Qwen3:1.7b) so these frequent small calls never touch Gemini quota.
    """
    return call_llm(prompt, role="planning")


def _llm_enrichment(prompt: str, unlimited_retries: bool = False) -> str:
    """
    Contextual chunk enrichment during PDF ingestion. Uses the 'enrichment'
    provider (default: local Ollama/Qwen3:1.7b) — completely offline, so
    ingesting large PDFs never hits a cloud rate limit.

    unlimited_retries is passed through only if the underlying provider
    supports it (currently GeminiProvider); OllamaProvider ignores it since
    a local model has no quota to exhaust — failures there are usually
    "Ollama isn't running", which retrying won't fix.
    """
    provider = get_provider("enrichment")
    # Only GeminiProvider.call() accepts unlimited_retries; other providers
    # use the base class signature (prompt only). We detect support via hasattr.
    import inspect
    sig = inspect.signature(provider.call)
    if "unlimited_retries" in sig.parameters:
        return provider.call(prompt, unlimited_retries=unlimited_retries)
    return provider.call(prompt)


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
        return _llm_planning(prompt)
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
    Build the LLM provider registry, pre-load ML models, and seed the
    university database on server start.

    Order matters:
      1. build_registry()  — read .env, construct each LLMProvider (cheap, no I/O)
      2. check_providers() — ping Ollama if used, log warnings (non-fatal)
      3. Load embedding + reranker models (~2.3 GB each, ~30-60s on M2)
      4. Seed university DB (uses the embedding model + enrichment LLM)
    """
    # Build the provider registry FIRST — everything below may call_llm()
    log.info("Building LLM provider registry…")
    build_registry()

    # Health-check local providers (e.g. Ollama). Logs warnings, doesn't crash
    # startup — if Ollama isn't running yet, you'll see a clear warning telling
    # you to run `ollama serve && ollama pull qwen3:1.7b`.
    check_providers()

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
            # ── Batch enrichment config ───────────────────────────────────────
            #
            # INTER-BATCH DELAY was originally added to stay under Gemini's
            # RPM ceiling (15 req/min on free tier = 4s minimum between calls).
            # But the enrichment role now defaults to LOCAL Ollama/Qwen3:1.7b,
            # which has NO rate limit — adding a 5s pause between every batch
            # is pure wasted time.
            #
            # NEW BEHAVIOUR: auto-detect the enrichment provider type and set
            # sensible defaults accordingly:
            #
            #   Local (Ollama):  batch_size=20, delay=0s
            #     → 133 chunks ÷ 20 = 7 batches × ~3s each ≈ ~21s total
            #
            #   Cloud (Gemini):  batch_size=5,  delay=5s  (original safe values)
            #     → 133 chunks ÷ 5 = 27 batches × ~8s each ≈ ~216s total
            #
            # You can still override both via .env:
            #   CONTEXT_BATCH_SIZE=10          — chunks per LLM call
            #   CONTEXT_INTER_BATCH_DELAY=2    — seconds between batches (0 = off)

            from llm_providers import get_provider, OllamaProvider as _OllamaProvider

            _enrichment_provider = get_provider("enrichment")
            _is_local = isinstance(_enrichment_provider, _OllamaProvider)

            # Per-env overrides take precedence; otherwise use smart defaults
            _default_batch  = "20" if _is_local else "5"
            _default_delay  = "0"  if _is_local else "5"

            CONTEXT_BATCH_SIZE        = int(os.getenv("CONTEXT_BATCH_SIZE", _default_batch))
            CONTEXT_INTER_BATCH_DELAY = float(os.getenv("CONTEXT_INTER_BATCH_DELAY", _default_delay))

            # Truncate document preview (same as context_builder.py uses)
            DOC_PREVIEW_CHARS = 3000
            doc_preview = full_text[:DOC_PREVIEW_CHARS]

            # Pre-fill with empty strings so the list is always len(chunks)
            contextual_texts = [""] * len(chunks)

            # ── Time estimate upfront ─────────────────────────────────────────
            total_batches = max(1, -(-len(chunks) // CONTEXT_BATCH_SIZE))  # ceil
            provider_name = _enrichment_provider.name
            log.info(
                "Contextual enrichment plan: %d chunks → %d batches of %d "
                "(delay=%.0fs between batches) via %s",
                len(chunks), total_batches, CONTEXT_BATCH_SIZE,
                CONTEXT_INTER_BATCH_DELAY, provider_name,
            )

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
                    # role="enrichment" → local Ollama/Qwen3:1.7b by default.
                    # unlimited_retries only matters if you've configured Gemini
                    # as the enrichment provider; Ollama ignores the flag since
                    # local inference has no quota to exhaust.
                    raw = _llm_enrichment(batch_prompt, unlimited_retries=True)

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

                # Inter-batch pause — only needed for cloud providers with RPM
                # limits. Local Ollama has no rate limit so delay defaults to 0.
                if CONTEXT_INTER_BATCH_DELAY > 0 and batch_end < len(chunks):
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

    # ── Stage 2: Intent classification (v9: routed to 'planning' provider) ─────
    # The force_intent field lets the frontend bypass classification for
    # specific UI actions (e.g. a "Find Universities" button always forces
    # the recommender intent regardless of the question text).
    # classify_intent() takes a call_llm callable — we pass _llm_planning so
    # this fast, frequent classification call uses local Qwen, not Gemini.
    intent = req.force_intent or classify_intent(rewritten, _llm_planning)
    log.info("Intent: %s", intent)

    # ── Stage 3: Execute plan ───────────────────────────────────────────────────
    # The recommender is wired in here via a lambda so university_recommender.py
    # never imports main.py (avoids circular imports).
    # planner.execute_plan also uses call_llm internally for search-query
    # building — we pass _llm_planning so that stays on the local model too.
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
        call_llm=_llm_planning,
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

    # Final, student-facing answer — uses role="answer" (Gemini cascade by
    # default). This is the ONLY call in the whole query flow that touches
    # Gemini quota, since rewriting/classification/planning all use the local
    # 'planning' provider above.
    answer = _llm_answer(prompt)

    # Persist assistant answer
    if req.conversation_id:
        db.add_message(req.conversation_id, "assistant", answer)

        # Auto-title from first user message
        conv = db.get_conversation(req.conversation_id)
        if conv and conv["title"] == "New conversation":
            title = question[:60] + ("…" if len(question) > 60 else "")
            db.update_conversation_title(req.conversation_id, title)

    # Build the response — include planner intent + web sources (v8/v9 fields)
    web_sources_out = []
    if plan_result.used_web_search:
        for raw in plan_result.web_results:
            for src in raw.get("sources", []):
                web_sources_out.append({"title": src.get("title", ""), "url": src.get("url", "")})

    return QueryResult(
        answer=answer,
        sources=sources,
        num_contexts=len(rag_chunks),
        retrieval_mode=retrieval_mode,
        rewritten_question=rewritten if rewritten != question else None,
        window_expanded=window_expanded,
        intent=plan_result.intent,
        web_sources=web_sources_out or None,
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