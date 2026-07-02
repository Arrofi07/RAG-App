# eval_engine.py
#
# Evaluation framework for the Study-in-Germany AI Advisor RAG pipeline.
#
# TWO-TIER DESIGN (per your requirements)
# ────────────────────────────────────────
# Tier 1 — Retrieval metrics (fast, free, runs automatically):
#   Measures whether the right chunks come back from the pipeline.
#   No LLM cost. Runs in seconds per question.
#   Metrics: Hit Rate, MRR, Context Precision (RAGAS-inspired definitions).
#
# Tier 2 — Answer quality metrics (RAGAS, on-demand only):
#   Measures whether the final answer is faithful to the retrieved context
#   and actually answers the question. Costs Gemini quota (1 call per
#   question per metric). Triggered only when the user clicks a button.
#   Metrics: Faithfulness, Answer Relevancy.
#
# SYNTHETIC Q&A GENERATION
# ─────────────────────────
# Rather than asking you to hand-write test questions, we generate them
# automatically from the ingested chunks using the LLM. The process:
#   1. Sample N chunks from Qdrant (diverse, covering different documents)
#   2. For each chunk, ask the LLM to write one question that the chunk answers
#   3. Store (question, source_chunk_id, source_text) as the ground truth
#
# Limitation to be aware of: synthetic questions test whether the retrieval
# system finds what it already knows, not whether the knowledge base covers
# what students actually ask. A curated set of real questions would be more
# valuable — this is a starting point, not the gold standard.
#
# METRIC DEFINITIONS
# ───────────────────
# Hit Rate @ K:
#   For each question, run retrieval with top_k=K. Score = 1 if the ground-
#   truth chunk appears in the top K results, else 0. Average over all questions.
#   Range: 0–1. Higher is better.
#
# MRR (Mean Reciprocal Rank) @ K:
#   Like hit rate, but rewards finding the right chunk higher in the ranking.
#   Score per question = 1/rank if found in top K, else 0. Average over all.
#   Range: 0–1. Higher is better.
#
# Context Precision @ K (RAGAS-inspired):
#   Of the K chunks retrieved, what fraction are actually relevant to the
#   question? We use the LLM to judge relevance (binary: relevant or not).
#   This is the most expensive retrieval metric — it costs 1 LLM call per
#   retrieved chunk per question. Run it on-demand rather than automatically.
#
# Faithfulness (RAGAS, on-demand):
#   Does every claim in the generated answer appear in the retrieved context?
#   LLM breaks the answer into atomic claims, then checks each against context.
#   Score = claims supported / total claims. Range: 0–1.
#
# Answer Relevancy (RAGAS, on-demand):
#   Does the answer actually address the question asked?
#   LLM generates N candidate questions from the answer, then checks how
#   similar they are to the original question via embedding cosine similarity.
#   Range: 0–1. Higher is better.

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/chatbot.db")
DEFAULT_TOP_K   = 5     # K for Hit Rate and MRR
MIN_CHUNK_CHARS = 200   # Skip chunks too short to generate a meaningful question


# ──────────────────────────────────────────────────────────────────────────────
# Eval dataset storage (SQLite, same DB as everything else)
# ──────────────────────────────────────────────────────────────────────────────

def _init_eval_schema(db_path: Path) -> None:
    """Create eval tables if they don't exist. Non-destructive."""
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript("""
        -- eval_questions: the synthetic ground-truth dataset
        CREATE TABLE IF NOT EXISTS eval_questions (
            id              TEXT PRIMARY KEY,
            question        TEXT NOT NULL,
            source_chunk_id TEXT NOT NULL,   -- Qdrant point ID the question came from
            source_text     TEXT NOT NULL,   -- the actual chunk text (snapshot at generation time)
            source_doc      TEXT NOT NULL,   -- filename of the source document
            created_at      TEXT NOT NULL
        );

        -- eval_runs: one row per evaluation run (a batch of questions evaluated)
        CREATE TABLE IF NOT EXISTS eval_runs (
            id           TEXT PRIMARY KEY,
            run_type     TEXT NOT NULL,     -- "retrieval" or "answer_quality"
            num_questions INTEGER NOT NULL,
            top_k        INTEGER NOT NULL,
            hit_rate     REAL,
            mrr          REAL,
            faithfulness REAL,              -- NULL for retrieval-only runs
            answer_relevancy REAL,          -- NULL for retrieval-only runs
            notes        TEXT,
            ran_at       TEXT NOT NULL
        );

        -- eval_results: individual question results within a run
        CREATE TABLE IF NOT EXISTS eval_results (
            id              TEXT PRIMARY KEY,
            run_id          TEXT NOT NULL REFERENCES eval_runs(id),
            question_id     TEXT NOT NULL REFERENCES eval_questions(id),
            question        TEXT NOT NULL,
            hit             INTEGER,        -- 1=found, 0=not found (retrieval)
            rank            INTEGER,        -- position in results (1=top, NULL=not found)
            retrieved_ids   TEXT,           -- JSON list of retrieved chunk IDs
            answer          TEXT,           -- generated answer (answer_quality runs only)
            faithfulness    REAL,           -- per-question score (answer_quality runs only)
            answer_relevancy REAL,          -- per-question score (answer_quality runs only)
            created_at      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_eval_results_run
            ON eval_results(run_id);
    """)
    con.commit()
    con.close()


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic question generation
# ──────────────────────────────────────────────────────────────────────────────

_QUESTION_GEN_PROMPT = """You are building an evaluation dataset for a RAG system about studying in Germany.

Given the chunk of text below, write ONE clear, specific question that:
1. Can be answered directly from this chunk
2. A student planning to study in Germany might realistically ask
3. Is self-contained (doesn't require knowing which document it came from)

Output ONLY the question — no preamble, no numbering, no explanation.

Chunk:
{chunk_text}

Question:"""


def generate_eval_questions(
    qdrant_store,             # QdrantStorage instance
    call_llm:   Callable,     # LLM callable (role="planning" — local Qwen)
    db_path:    Path = DEFAULT_DB_PATH,
    n_questions: int = 20,    # how many questions to generate
    sample_seed: int = 42,    # for reproducible sampling
) -> list[dict]:
    """
    Sample chunks from Qdrant and generate one synthetic question per chunk.

    Stores the generated questions in eval_questions table and returns them.

    Args:
        qdrant_store: QdrantStorage instance (for scrolling chunks)
        call_llm:     LLM callable — use the planning provider (local, free)
        db_path:      SQLite DB path
        n_questions:  target number of questions (may be less if few chunks)
        sample_seed:  random seed for reproducible chunk sampling

    Returns:
        List of question dicts with keys: id, question, source_chunk_id,
        source_text, source_doc, created_at
    """
    import random
    _init_eval_schema(db_path)

    # ── Sample chunks from Qdrant ─────────────────────────────────────────────
    # We scroll the whole collection (no query vector needed) and sample
    # a diverse set. The scroll limit is set generously to get variety;
    # we'll sample from whatever comes back.
    log.info("Sampling chunks from Qdrant for eval question generation…")

    try:
        records, _ = qdrant_store.client.scroll(
            collection_name=qdrant_store.collection,
            limit=500,            # fetch up to 500, sample n_questions from them
            with_payload=True,
            with_vectors=False,   # no vectors needed for question generation
        )
    except Exception as exc:
        log.error("Failed to scroll Qdrant for eval generation: %s", exc)
        return []

    # Filter: skip chunks that are too short to make a good question
    usable = [
        r for r in records
        if len((r.payload or {}).get("text", "")) >= MIN_CHUNK_CHARS
    ]

    if not usable:
        log.warning("No usable chunks found (all below %d chars). Ingest some PDFs first.", MIN_CHUNK_CHARS)
        return []

    # Reproducible random sample
    rng = random.Random(sample_seed)
    sample = rng.sample(usable, min(n_questions, len(usable)))
    log.info("Sampled %d chunks from %d usable chunks", len(sample), len(usable))

    # ── Generate questions ────────────────────────────────────────────────────
    generated = []
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")

    for i, record in enumerate(sample):
        payload    = record.payload or {}
        chunk_text = payload.get("text", "")
        source_doc = payload.get("source") or payload.get("filename", "unknown")
        chunk_id   = str(record.id)

        log.info("Generating eval question %d/%d from '%s'…", i + 1, len(sample), source_doc)

        prompt = _QUESTION_GEN_PROMPT.format(chunk_text=chunk_text[:1500])

        try:
            question = call_llm(prompt).strip().strip('"').strip("'")
        except Exception as exc:
            log.warning("Question generation failed for chunk %s: %s — skipping.", chunk_id, exc)
            continue

        if not question or len(question) < 10:
            log.warning("LLM returned a too-short question for chunk %s — skipping.", chunk_id)
            continue

        qid = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        con.execute(
            """INSERT INTO eval_questions
               (id, question, source_chunk_id, source_text, source_doc, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (qid, question, chunk_id, chunk_text, source_doc, now),
        )
        con.commit()

        generated.append({
            "id":              qid,
            "question":        question,
            "source_chunk_id": chunk_id,
            "source_text":     chunk_text,
            "source_doc":      source_doc,
            "created_at":      now,
        })

    con.close()
    log.info("✅ Generated %d eval questions.", len(generated))
    return generated


def load_eval_questions(db_path: Path = DEFAULT_DB_PATH) -> list[dict]:
    """Load all stored eval questions from the DB."""
    _init_eval_schema(db_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM eval_questions ORDER BY created_at DESC"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def delete_eval_questions(db_path: Path = DEFAULT_DB_PATH) -> int:
    """Delete all eval questions (to regenerate from scratch). Returns count deleted."""
    con = sqlite3.connect(db_path)
    cur = con.execute("DELETE FROM eval_questions")
    count = cur.rowcount
    con.commit()
    con.close()
    return count


# ──────────────────────────────────────────────────────────────────────────────
# Tier 1 — Retrieval evaluation (fast, free)
# ──────────────────────────────────────────────────────────────────────────────

def run_retrieval_eval(
    questions:    list[dict],
    embed_fn:     Callable,       # embed(texts) → {"dense": ..., "sparse": ...}
    qdrant_store,                 # QdrantStorage instance
    rerank_fn:    Optional[Callable] = None,   # rerank(query, candidates) → list
    top_k:        int = DEFAULT_TOP_K,
    db_path:      Path = DEFAULT_DB_PATH,
    notes:        str = "",
) -> dict:
    """
    Run retrieval evaluation over a set of questions and store results.

    For each question:
      1. Embed the question
      2. Run hybrid retrieval (dense + sparse + RRF)
      3. Optionally rerank
      4. Check if the ground-truth chunk_id appears in the top_k results
      5. Record the rank if found (for MRR)

    Returns a summary dict with hit_rate, mrr, and per-question details.

    NOTE: This requires that the ground-truth chunk (source_chunk_id) is still
    in Qdrant. If you delete and re-ingest documents, regenerate the questions.
    """
    _init_eval_schema(db_path)

    run_id  = str(uuid.uuid4())
    now     = datetime.utcnow().isoformat()
    hits    = []
    ranks   = []
    details = []

    log.info("Starting retrieval eval: %d questions, top_k=%d", len(questions), top_k)

    for i, q in enumerate(questions):
        question        = q["question"]
        target_chunk_id = q["source_chunk_id"]

        # Embed the question for retrieval
        try:
            q_emb    = embed_fn([question])
            q_dense  = q_emb["dense"][0]
            q_sparse = q_emb["sparse"][0]
        except Exception as exc:
            log.warning("Embedding failed for question %d: %s — skipping.", i, exc)
            continue

        # Run hybrid retrieval
        try:
            candidates = qdrant_store.search_hybrid_candidates(
                dense_vector=q_dense,
                sparse_vector=q_sparse,
                fetch_k=top_k * 3,   # fetch extra so reranker has room to work
            )
        except Exception as exc:
            log.warning("Retrieval failed for question %d: %s — skipping.", i, exc)
            continue

        # Optionally rerank
        if rerank_fn is not None:
            try:
                candidates = rerank_fn(question, candidates, top_k=top_k)
            except Exception as exc:
                log.warning("Reranking failed for question %d: %s — using pre-rerank order.", i, exc)
                candidates = candidates[:top_k]
        else:
            candidates = candidates[:top_k]

        # Check if ground-truth chunk appears in results
        retrieved_ids = [c["id"] for c in candidates]
        hit           = target_chunk_id in retrieved_ids
        rank          = (retrieved_ids.index(target_chunk_id) + 1) if hit else None

        hits.append(1 if hit else 0)
        ranks.append(1.0 / rank if rank else 0.0)

        log.info(
            "  Q%d: %s → hit=%s rank=%s",
            i + 1, question[:60], hit, rank,
        )

        details.append({
            "question_id":    q["id"],
            "question":       question,
            "hit":            int(hit),
            "rank":           rank,
            "retrieved_ids":  retrieved_ids,
        })

    # ── Compute summary metrics ───────────────────────────────────────────────
    n         = len(hits)
    hit_rate  = sum(hits) / n if n > 0 else 0.0
    mrr       = sum(ranks) / n if n > 0 else 0.0

    log.info("Retrieval eval complete: hit_rate=%.3f  mrr=%.3f  (n=%d)", hit_rate, mrr, n)

    # ── Store run and results ─────────────────────────────────────────────────
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(
        """INSERT INTO eval_runs
           (id, run_type, num_questions, top_k, hit_rate, mrr, ran_at, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, "retrieval", n, top_k, hit_rate, mrr, now, notes),
    )

    for d in details:
        con.execute(
            """INSERT INTO eval_results
               (id, run_id, question_id, question, hit, rank, retrieved_ids, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), run_id, d["question_id"], d["question"],
             d["hit"], d["rank"], json.dumps(d["retrieved_ids"]), now),
        )

    con.commit()
    con.close()

    return {
        "run_id":    run_id,
        "run_type":  "retrieval",
        "hit_rate":  hit_rate,
        "mrr":       mrr,
        "n":         n,
        "top_k":     top_k,
        "details":   details,
        "ran_at":    now,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Tier 2 — Answer quality evaluation (RAGAS, on-demand)
# ──────────────────────────────────────────────────────────────────────────────

_FAITHFULNESS_CLAIM_PROMPT = """Break the following answer into individual factual claims.
Output each claim on a new line, numbered. Output ONLY the claims, nothing else.

Answer:
{answer}

Claims:"""

_FAITHFULNESS_CHECK_PROMPT = """Does the following claim appear in or follow directly from the context?
Answer with exactly "YES" or "NO".

Context:
{context}

Claim: {claim}

Answer:"""

_RELEVANCY_QUESTION_GEN_PROMPT = """Given the following answer, generate {n} questions that this answer could be responding to.
Output one question per line, no numbering, no preamble.

Answer:
{answer}

Questions:"""


def run_answer_quality_eval(
    questions:    list[dict],
    embed_fn:     Callable,
    qdrant_store,
    answer_fn:    Callable,      # callable(question) → answer string (full pipeline)
    call_llm:     Callable,      # for RAGAS LLM calls (use answer role — Gemini)
    rerank_fn:    Optional[Callable] = None,
    top_k:        int = DEFAULT_TOP_K,
    db_path:      Path = DEFAULT_DB_PATH,
    n_questions_for_relevancy: int = 3,   # questions generated per answer for relevancy
    notes:        str = "",
) -> dict:
    """
    On-demand RAGAS-style answer quality evaluation.

    For each question:
      1. Run the full RAG pipeline to get an answer + retrieved context
      2. Faithfulness: decompose answer into claims, check each against context
      3. Answer Relevancy: generate N questions from the answer, check similarity
         to the original question using embedding cosine similarity

    This is expensive: roughly 5–10 Gemini calls per question.
    Only run on-demand, not as part of automated regression testing.
    """
    import numpy as np
    _init_eval_schema(db_path)

    run_id        = str(uuid.uuid4())
    now           = datetime.utcnow().isoformat()
    faithfulness_scores  = []
    relevancy_scores     = []
    details       = []

    log.info("Starting answer quality eval: %d questions", len(questions))

    for i, q in enumerate(questions):
        question        = q["question"]
        target_chunk_id = q["source_chunk_id"]

        log.info("  Answer quality Q%d/%d: %s", i + 1, len(questions), question[:60])

        # ── Get answer + context from the pipeline ────────────────────────────
        try:
            answer, context = answer_fn(question)
        except Exception as exc:
            log.warning("Answer generation failed for Q%d: %s — skipping.", i, exc)
            continue

        if not answer or not context:
            log.warning("Empty answer or context for Q%d — skipping.", i)
            continue

        # ── Faithfulness score ────────────────────────────────────────────────
        # Step 1: Extract atomic claims from the answer
        faithfulness = None
        try:
            claims_raw = call_llm(
                _FAITHFULNESS_CLAIM_PROMPT.format(answer=answer)
            )
            # Parse numbered list: "1. claim\n2. claim\n..."
            import re
            claims = [
                re.sub(r"^\d+\.\s*", "", line).strip()
                for line in claims_raw.splitlines()
                if line.strip() and re.match(r"^\d+\.", line.strip())
            ]

            if claims:
                # Step 2: Check each claim against the context
                supported = 0
                context_str = context if isinstance(context, str) else "\n\n".join(context)

                for claim in claims:
                    verdict = call_llm(
                        _FAITHFULNESS_CHECK_PROMPT.format(
                            context=context_str[:3000],
                            claim=claim,
                        )
                    ).strip().upper()
                    if verdict.startswith("YES"):
                        supported += 1

                faithfulness = supported / len(claims)
                faithfulness_scores.append(faithfulness)
                log.info("    Faithfulness: %.2f (%d/%d claims supported)",
                         faithfulness, supported, len(claims))
            else:
                log.warning("    No claims extracted from answer — faithfulness skipped.")

        except Exception as exc:
            log.warning("    Faithfulness eval failed for Q%d: %s", i, exc)

        # ── Answer relevancy score ────────────────────────────────────────────
        relevancy = None
        try:
            # Generate n candidate questions from the answer
            gen_questions_raw = call_llm(
                _RELEVANCY_QUESTION_GEN_PROMPT.format(
                    answer=answer[:2000],
                    n=n_questions_for_relevancy,
                )
            )
            gen_questions = [
                line.strip()
                for line in gen_questions_raw.splitlines()
                if line.strip() and len(line.strip()) > 5
            ][:n_questions_for_relevancy]

            if gen_questions:
                # Embed the original question and the generated questions
                all_texts  = [question] + gen_questions
                embeddings = embed_fn(all_texts)["dense"]
                orig_vec   = embeddings[0]
                gen_vecs   = embeddings[1:]

                # Cosine similarity between original and each generated question
                orig_norm = np.linalg.norm(orig_vec)
                sims = []
                for gv in gen_vecs:
                    gv_norm = np.linalg.norm(gv)
                    if orig_norm > 0 and gv_norm > 0:
                        sim = float(np.dot(orig_vec, gv) / (orig_norm * gv_norm))
                        sims.append(sim)

                if sims:
                    relevancy = float(np.mean(sims))
                    relevancy_scores.append(relevancy)
                    log.info("    Answer relevancy: %.2f (avg of %d sims)", relevancy, len(sims))

        except Exception as exc:
            log.warning("    Answer relevancy eval failed for Q%d: %s", i, exc)

        details.append({
            "question_id":    q["id"],
            "question":       question,
            "answer":         answer,
            "faithfulness":   faithfulness,
            "answer_relevancy": relevancy,
        })

    # ── Compute summary metrics ───────────────────────────────────────────────
    n            = len(details)
    avg_faith    = float(sum(faithfulness_scores) / len(faithfulness_scores)) if faithfulness_scores else None
    avg_relevancy = float(sum(relevancy_scores) / len(relevancy_scores)) if relevancy_scores else None

    log.info(
        "Answer quality eval complete: faithfulness=%.3f  relevancy=%.3f  (n=%d)",
        avg_faith or 0, avg_relevancy or 0, n,
    )

    # ── Store run and results ─────────────────────────────────────────────────
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(
        """INSERT INTO eval_runs
           (id, run_type, num_questions, top_k, faithfulness, answer_relevancy, ran_at, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, "answer_quality", n, top_k, avg_faith, avg_relevancy, now, notes),
    )

    for d in details:
        con.execute(
            """INSERT INTO eval_results
               (id, run_id, question_id, question, answer, faithfulness,
                answer_relevancy, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), run_id, d["question_id"], d["question"],
             d["answer"], d["faithfulness"], d["answer_relevancy"], now),
        )

    con.commit()
    con.close()

    return {
        "run_id":           run_id,
        "run_type":         "answer_quality",
        "faithfulness":     avg_faith,
        "answer_relevancy": avg_relevancy,
        "n":                n,
        "details":          details,
        "ran_at":           now,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Load historical runs (for the dashboard)
# ──────────────────────────────────────────────────────────────────────────────

def load_eval_runs(db_path: Path = DEFAULT_DB_PATH) -> list[dict]:
    """Return all eval runs, newest first, for the dashboard."""
    _init_eval_schema(db_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM eval_runs ORDER BY ran_at DESC"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def load_run_details(run_id: str, db_path: Path = DEFAULT_DB_PATH) -> list[dict]:
    """Return per-question results for a specific run."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM eval_results WHERE run_id = ? ORDER BY created_at ASC",
        (run_id,),
    ).fetchall()
    con.close()

    results = []
    for r in rows:
        d = dict(r)
        if d.get("retrieved_ids"):
            try:
                d["retrieved_ids"] = json.loads(d["retrieved_ids"])
            except Exception:
                pass
        results.append(d)
    return results