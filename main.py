# main.py

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

from data_loader import load_chunks, embed, EMBED_DIM
from vector_db import QdrantStorage
from reranker import rerank
from context_builder import generate_chunk_context, build_contextual_text
from storage import Storage
from advisor_config import build_system_prompt
from custom_types import (
    ChatMessage,
    MetaFilter,
    IngestResult,
    DocumentListResult,
    QueryResult,
    MatchItem,
    ConversationMeta,
    UserInfo,
    UserProfile,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Boot
# ─────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set.")

client  = genai.Client(api_key=GEMINI_API_KEY)
app     = FastAPI(title="RAG Chatbot API", version="7.0.0")
store   = QdrantStorage(dim=EMBED_DIM)
db      = Storage()

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))

# Model priority order, tuned to June 2026 free-tier limits:
#   gemini-2.5-flash-lite  — 15 RPM, ~1 000 RPD  → try first (most generous daily)
#   gemini-2.5-flash       — 10 RPM, ~250  RPD  → second choice
#   gemini-3-flash         — 10 RPM, ~1 500 RPD  → last resort (newest, may vary by account)
# gemini-2.0-flash is intentionally removed — its free-tier quota is 0.
GEMINI_MODELS = (
    os.getenv("GEMINI_MODELS", "gemini-2.5-flash-lite,gemini-2.5-flash,gemini-3-flash")
    .split(",")
)

# How many times to retry a 429 on the same model before moving to the next.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))


# ─────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:             str
    history:              list[ChatMessage] = []
    user_id:              Optional[str]     = None   # load profile if supplied
    conversation_id:      Optional[str]     = None   # persist messages if supplied
    top_k:                int               = 5
    fetch_k:              int               = 30
    use_hybrid:           bool              = True
    use_window_expansion: bool              = True
    filters:              MetaFilter        = MetaFilter()


class CreateUserRequest(BaseModel):
    name:    str
    profile: dict[str, Any] = {}


class UpdateProfileRequest(BaseModel):
    profile: dict[str, Any]


# ─────────────────────────────────────────────
# LLM helper
# ─────────────────────────────────────────────

def _parse_retry_delay(exc: Exception) -> float:
    """
    Extract the server-suggested retry delay from a Gemini 429 error.

    Gemini embeds a RetryInfo block in the error details:
        {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '22s'}

    We parse that string and return seconds as a float.
    Falls back to a conservative 30 s if parsing fails.
    """
    try:
        details = exc.args[0].get("error", {}).get("details", []) if exc.args else []
        for d in details:
            if "retryDelay" in d:
                raw = d["retryDelay"]            # e.g. "22.685926265s" or "22s"
                return float(raw.rstrip("s"))
    except Exception:
        pass
    return 30.0


def _is_rate_limit(exc: Exception) -> bool:
    try:
        code = getattr(exc, "code", None) or (exc.args[0] if exc.args else {})
        if isinstance(code, int):
            return code == 429
        if isinstance(code, dict):
            return code.get("error", {}).get("code") == 429
        return "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
    except Exception:
        return False


def _call_llm(prompt: str) -> str:
    """
    Call the LLM with automatic retry on 429 errors.

    Strategy per model:
      1. Try the model.
      2. On 429: read the server-supplied retryDelay, wait, then retry
         (up to LLM_MAX_RETRIES times per model).
      3. On any other error, or after exhausting retries: move to the next model.
    Only raises HTTPException 503 after ALL models are exhausted.
    """
    last_exc: Exception | None = None

    for model_name in GEMINI_MODELS:
        for attempt in range(1, LLM_MAX_RETRIES + 2):   # +2: first try + N retries
            try:
                resp = client.models.generate_content(model=model_name, contents=prompt)
                if hasattr(resp, "text") and resp.text:
                    if attempt > 1:
                        log.info("%s succeeded after %d attempt(s)", model_name, attempt)
                    return resp.text.strip()

            except Exception as exc:
                last_exc = exc

                if _is_rate_limit(exc):
                    if attempt <= LLM_MAX_RETRIES:
                        delay = _parse_retry_delay(exc)
                        log.warning(
                            "%s — 429 rate limit (attempt %d/%d). "
                            "Waiting %.1f s before retry…",
                            model_name, attempt, LLM_MAX_RETRIES + 1, delay,
                        )
                        import time; time.sleep(delay)
                        continue    # retry same model
                    else:
                        log.warning(
                            "%s — 429 exhausted after %d attempts. Trying next model.",
                            model_name, attempt,
                        )
                        break       # move to next model
                else:
                    log.warning("%s — non-rate-limit error: %s", model_name, exc)
                    break           # don't retry non-429 errors; move to next model

    raise HTTPException(
        status_code=503,
        detail=(
            f"All LLM backends unavailable after retries. "
            f"Last error: {last_exc}"
        ),
    )


# ─────────────────────────────────────────────
# Query rewriting
# ─────────────────────────────────────────────

def _rewrite_question(question: str, history: list[ChatMessage]) -> str:
    if not history:
        return question

    recent     = history[-(MAX_HISTORY_TURNS):]
    transcript = "\n".join(f"{m.role.upper()}: {m.content}" for m in recent)

    prompt = f"""You are a query rewriter for a RAG system.

Rewrite the user's latest message into a single, fully self-contained question
that can be understood without the conversation history.

Rules:
- Output ONLY the rewritten question. No preamble or explanation.
- If already self-contained, return it unchanged.
- Preserve the original language of the user.

Conversation history:
{transcript}

Latest message: {question}

Rewritten question:"""

    try:
        return _call_llm(prompt)
    except Exception:
        log.warning("Query rewriting failed, using original question.")
        return question


# ─────────────────────────────────────────────
# Window expansion
# ─────────────────────────────────────────────

def _expand_with_window(candidates: list[dict]) -> list[dict]:
    for c in candidates:
        parts = [p for p in [c.get("prev_chunk", ""), c["text"], c.get("next_chunk", "")] if p]
        c["text_sent_to_llm"] = "\n\n".join(parts)
    return candidates


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "running", "version": "7.0.0 — persistent memory + advisor profile"}


# ─────────────────────────────────────────────
# User / profile endpoints
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Conversation endpoints
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Ingest PDF
# ─────────────────────────────────────────────

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

        for i, chunk in enumerate(chunks):
            if do_contextual:
                log.info("Generating context chunk %d/%d — '%s'", i + 1, len(chunks), source_id)
                ctx      = generate_chunk_context(full_text, chunk["text"], source_id, _call_llm)
                enriched = build_contextual_text(ctx, chunk["text"])
            else:
                ctx      = ""
                enriched = chunk["text"]

            contextual_texts.append(ctx)
            texts_to_embed.append(enriched)

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

        log.info("Ingested %d chunks from '%s' | contextual=%s", len(chunks), source_id, do_contextual)

        return IngestResult(success=True, source=source_id, chunks=len(chunks),
                            contextual_enriched=do_contextual)
    finally:
        Path(pdf_path).unlink(missing_ok=True)


# ─────────────────────────────────────────────
# Query
# ─────────────────────────────────────────────

@app.post("/query", response_model=QueryResult)
def query(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Load user profile for personalised system prompt
    profile: dict = {}
    if req.user_id:
        user = db.get_user(req.user_id)
        if user:
            profile = user.get("profile", {})

    # Persist incoming user message
    if req.conversation_id:
        db.add_message(req.conversation_id, "user", question)

    # ── Build history from DB if conversation_id supplied ────────────────────
    # DB history is the authoritative source; the client-sent history is a
    # fallback for stateless callers (e.g. direct API use, testing).
    if req.conversation_id:
        db_msgs   = db.get_messages(req.conversation_id)
        # Exclude the message we just inserted (last one) — it's the current question
        history   = [ChatMessage(role=m["role"], content=m["content"]) for m in db_msgs[:-1]]
    else:
        history = req.history

    # ── Stage 1: Query rewriting ─────────────────────────────────────────────
    rewritten = _rewrite_question(question, history)
    if rewritten != question:
        log.info("Rewritten: '%s' → '%s'", question, rewritten)

    # ── Stage 2: Retrieval ───────────────────────────────────────────────────
    q_emb    = embed([rewritten])
    q_dense  = q_emb["dense"][0]
    q_sparse = q_emb["sparse"][0]

    if req.use_hybrid:
        candidates     = store.search_hybrid_candidates(q_dense, q_sparse,
                                                        fetch_k=req.fetch_k, meta=req.filters)
        retrieval_mode = "hybrid"
    else:
        candidates     = store.search_candidates(q_dense, fetch_k=req.fetch_k, meta=req.filters)
        retrieval_mode = "dense"

    # ── Stage 3: Rerank ──────────────────────────────────────────────────────
    reranked = rerank(rewritten, candidates, top_k=req.top_k)

    # ── Stage 4: Window expansion ─────────────────────────────────────────────
    if req.use_window_expansion:
        reranked = _expand_with_window(reranked)

    window_expanded = req.use_window_expansion and any(
        c.get("prev_chunk") or c.get("next_chunk") for c in reranked
    )

    # ── Stage 5: Generate ─────────────────────────────────────────────────────
    llm_texts     = [c.get("text_sent_to_llm") or c["text"] for c in reranked]
    context_block = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(llm_texts))

    sources = []
    for c in reranked:
        if c["source"] and c["source"] not in sources:
            sources.append(c["source"])

    # Combine recent history for the prompt
    recent_history = history[-(MAX_HISTORY_TURNS):]
    history_block  = (
        "\n".join(f"{m.role.upper()}: {m.content}" for m in recent_history)
        if recent_history else ""
    )

    system_prompt = build_system_prompt({**profile, "name": profile.get("name", "")})

    prompt = f"""{system_prompt}

━━━ RETRIEVED DOCUMENT CONTEXT ━━━

{context_block}

━━━ INSTRUCTIONS ━━━
- Answer using ONLY the retrieved context above AND your advisor knowledge.
- If the context doesn't cover the question, rely on your Germany-studies expertise.
- Cite context numbers like [1], [2] when quoting specific documents.
- Be warm, specific, and actionable.
{"━━━ CONVERSATION SO FAR ━━━" + chr(10) + history_block if history_block else ""}

User: {question}

Advisor:"""

    answer = _call_llm(prompt)

    # Persist assistant answer
    if req.conversation_id:
        db.add_message(req.conversation_id, "assistant", answer)

        # Auto-title the conversation from the first user message
        conv = db.get_conversation(req.conversation_id)
        if conv and conv["title"] == "New conversation":
            title = question[:60] + ("…" if len(question) > 60 else "")
            db.update_conversation_title(req.conversation_id, title)

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


# ─────────────────────────────────────────────
# Documents
# ─────────────────────────────────────────────

@app.get("/documents", response_model=DocumentListResult)
def list_documents():
    return DocumentListResult(documents=[d["filename"] for d in store.list_documents()])


@app.get("/documents/metadata")
def list_documents_metadata():
    return {"documents": store.list_documents()}