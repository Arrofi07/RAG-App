# planner.py
#
# Agentic query planner for the Study-in-Germany AI Advisor.
#
# WHAT IS AGENTIC PLANNING?
# ──────────────────────────
# A plain RAG system treats every question the same: embed → retrieve → answer.
# That works for "What documents do I need for a student visa?" but breaks for:
#   - "What is the current blocked account amount for 2026/27?" → needs live web
#   - "Which universities suit me for CS with B2 German and €900/month?" → needs
#     the university recommender, not just RAG chunks
#   - "Explain what a Studienkolleg is" → pure RAG is perfect, no web needed
#
# The planner sits BETWEEN the API endpoint and the retrieval/generation steps.
# It classifies the question and assembles the right mix of tools before the
# LLM writes the final answer.
#
# PLAN FLOW
# ─────────
#   1. classify_intent()   — ask a small LLM call what kind of question this is
#   2. execute_plan()      — run the tools the classification calls for
#   3. build_augmented_context() — merge RAG chunks + web results into one block
#
# MEMORY CONSTRAINT (M2 / 8 GB)
# ───────────────────────────────
# The planner itself uses zero extra RAM.  It orchestrates calls to modules
# that are already loaded.  The only new cost is the small Gemini API call
# for intent classification (~200 tokens), which is network-bound, not RAM-bound.

import logging
import json
import re
from typing import Optional

from web_search_tool import search_web, format_search_result_for_prompt, needs_web_search

log = logging.getLogger(__name__)


# ─── Intent taxonomy ─────────────────────────────────────────────────────────

# Each intent maps to a set of tools the planner will activate.
# This dict is the "policy table" — edit it to change routing behaviour
# without touching any other logic.
INTENT_TOOL_MAP = {
    # Pure knowledge questions — RAG PDFs + advisor persona are enough
    "rag_only": {
        "use_rag":         True,
        "use_web_search":  False,
        "use_recommender": False,
        "description":     "Factual / procedural question answerable from ingested docs",
    },
    # Deadline / fee / threshold questions — PDFs go stale, always check live
    "web_search": {
        "use_rag":         True,    # still use RAG for context
        "use_web_search":  True,    # but add live web data on top
        "use_recommender": False,
        "description":     "Time-sensitive question (deadlines, fees, requirements)",
    },
    # University match requests — needs structured filtering, not just RAG
    "university_recommender": {
        "use_rag":         False,   # recommender returns structured matches
        "use_web_search":  False,
        "use_recommender": True,
        "description":     "University / program recommendation request",
    },
    # Hybrid: recommend + live info (e.g. "Which uni for CS AND what's the deadline?")
    "recommend_and_search": {
        "use_rag":         False,
        "use_web_search":  True,
        "use_recommender": True,
        "description":     "Recommendation request that also needs live deadline data",
    },
}


# ─── Intent classifier ───────────────────────────────────────────────────────

_CLASSIFY_PROMPT = """You are a query router for a Study-in-Germany AI advisor.

Given the student's question, output ONLY one of these JSON keys — nothing else:

  "rag_only"               — general knowledge, visa steps, document lists,
                             explanations, language requirements (non-deadline),
                             life in Germany, scholarship overviews
  "web_search"             — anything requiring CURRENT data: deadlines, fees,
                             blocked-account amounts, current rankings, open
                             application windows, recent policy changes
  "university_recommender" — student asks for university or program suggestions
                             based on their profile (GPA, language, city, budget)
  "recommend_and_search"   — university recommendation AND a time-sensitive
                             aspect (e.g. "which university and what's the
                             application deadline for winter 2026/27?")

Student question: {question}

Respond with ONLY the key string, e.g.: "rag_only"
"""


def classify_intent(question: str, call_llm) -> str:
    """
    Ask the LLM to classify the question intent.

    We use a fast heuristic first (needs_web_search keyword scan) to avoid
    burning an LLM call on obvious cases.  Only ambiguous questions get the
    full LLM classification.

    Args:
        question : the (possibly rewritten) user question
        call_llm : callable(str) → str, injected from main.py

    Returns:
        One of the intent keys from INTENT_TOOL_MAP.
    """
    # ── Fast-path: keyword heuristic ─────────────────────────────────────────
    # Avoid an LLM call for clearly time-sensitive questions
    if needs_web_search(question):
        log.info("Planner fast-path → web_search (keyword match)")
        return "web_search"

    # ── LLM classification for nuanced cases ─────────────────────────────────
    prompt = _CLASSIFY_PROMPT.format(question=question)

    try:
        raw = call_llm(prompt).strip().strip('"').strip("'").lower()

        # The LLM should return just the key; clean up if it added extra text
        for key in INTENT_TOOL_MAP:
            if key in raw:
                log.info("Planner LLM intent → %s", key)
                return key

        log.warning("Planner: unrecognised intent '%s', defaulting to rag_only", raw)
        return "rag_only"

    except Exception as exc:
        log.warning("Planner: intent classification failed (%s), defaulting to rag_only", exc)
        return "rag_only"


# ─── Search query builder ─────────────────────────────────────────────────────

_SEARCH_QUERY_PROMPT = """You are helping build a precise Google search query for a student advisor.

The student asked: "{question}"

Their profile context:
- Target intake: {intake}
- Field of study: {field}
- Nationality: {nationality}

Write ONE focused search query (max 12 words) that will find the most current,
official information to answer this question.  Prefer queries that include:
  - The current year (2026 or 2027 if next semester)
  - "Germany" or the specific university name if relevant
  - Official source keywords like "DAAD", "uni-assist", "Ausländerbehörde"

Output ONLY the search query string.
"""


def build_search_query(question: str, profile: dict, call_llm) -> str:
    """
    Ask the LLM to craft an optimised search query from the student's question
    and profile.  Falls back to the raw question if the LLM call fails.

    Using the profile here lets us build queries like:
      "blocked account Germany 2026 Indonesian student" instead of "blocked account"
    """
    prompt = _SEARCH_QUERY_PROMPT.format(
        question=question,
        intake=profile.get("intake_semester", "not specified"),
        field=profile.get("field_of_study", "not specified"),
        nationality=profile.get("nationality", "not specified"),
    )

    try:
        query = call_llm(prompt).strip().strip('"')
        log.info("Planner: generated search query → '%s'", query)
        return query
    except Exception:
        log.warning("Search query generation failed, using raw question.")
        return question


# ─── Plan execution ──────────────────────────────────────────────────────────

class PlanResult:
    """
    Container for the results of executing a plan.
    Passed to the answer generator so it can build a rich, grounded prompt.
    """

    def __init__(self):
        self.intent:           str         = "rag_only"
        self.web_results:      list[dict]  = []   # raw search_web() dicts
        self.web_context:      str         = ""   # formatted for LLM prompt
        self.recommend_result: Optional[dict] = None  # from university_recommender
        self.used_web_search:  bool        = False
        self.used_recommender: bool        = False

    def to_log_summary(self) -> str:
        """One-line summary for log output."""
        parts = [f"intent={self.intent}"]
        if self.used_web_search:
            parts.append(f"web_results={len(self.web_results)}")
        if self.used_recommender and self.recommend_result:
            n = len(self.recommend_result.get("universities", []))
            parts.append(f"uni_matches={n}")
        return " | ".join(parts)


def execute_plan(
    question:   str,
    profile:    dict,
    intent:     str,
    call_llm,
    recommender_fn=None,    # callable(profile, question) → dict  (injected)
) -> PlanResult:
    """
    Execute the plan dictated by the classified intent.

    Args:
        question       : the (possibly rewritten) student question
        profile        : the student's advisor profile dict
        intent         : output of classify_intent()
        call_llm       : LLM callable for sub-tasks (search query building)
        recommender_fn : optional callable for university recommendations;
                         if None, university_recommender intents fall back to rag_only

    Returns:
        A PlanResult with all tool outputs assembled.
    """
    plan_config = INTENT_TOOL_MAP.get(intent, INTENT_TOOL_MAP["rag_only"])
    result      = PlanResult()
    result.intent = intent

    # ── Web search ────────────────────────────────────────────────────────────
    if plan_config["use_web_search"]:
        # Build a precise query rather than sending the raw question
        search_query = build_search_query(question, profile, call_llm)

        raw = search_web(search_query)
        result.web_results.append(raw)
        result.used_web_search = True

        # Format into a string block ready for the LLM prompt
        result.web_context = format_search_result_for_prompt(raw)

        log.info("Plan web search done: %s", raw.get("error") or "ok")

    # ── University recommender ────────────────────────────────────────────────
    if plan_config["use_recommender"]:
        if recommender_fn is not None:
            try:
                result.recommend_result = recommender_fn(profile, question)
                result.used_recommender = True
                log.info(
                    "Recommender returned %d matches",
                    len(result.recommend_result.get("universities", [])),
                )
            except Exception as exc:
                log.warning("Recommender failed: %s — falling back to RAG", exc)
                result.intent = "rag_only"   # degrade gracefully
        else:
            # Recommender not wired up yet — tell the LLM to answer from its
            # advisor knowledge instead of silently giving no results
            log.info("Recommender not available, intent downgraded to rag_only")
            result.intent = "rag_only"

    log.info("Plan executed: %s", result.to_log_summary())
    return result


# ─── Context assembler ────────────────────────────────────────────────────────

def build_augmented_context(
    rag_chunks:   list[dict],
    plan_result:  PlanResult,
) -> str:
    """
    Merge RAG chunks and plan tool outputs into a single context block for
    the final LLM prompt.

    Layout:
    ──────
    ═══ RAG CONTEXT ═══
    [1] chunk text…
    [2] chunk text…

    ═══ LIVE WEB DATA ═══
    ⟨WEB — blocked account Germany 2026⟩
    The required blocked account amount for 2026/27 is €11,208…
    Sources: [1] DAAD — https://…

    ═══ UNIVERSITY RECOMMENDATIONS ═══
    (structured JSON from the recommender, rendered as a readable list)

    The LLM prompt instructs the model to cite [1], [2] for RAG, ⟨WEB⟩ for
    live data, and to present the recommendations as a formatted list.
    """
    sections = []

    # ── RAG chunks ────────────────────────────────────────────────────────────
    if rag_chunks:
        chunk_lines = [f"[{i+1}] {c.get('text_sent_to_llm') or c['text']}"
                       for i, c in enumerate(rag_chunks)]
        sections.append("═══ RAG DOCUMENT CONTEXT ═══\n" + "\n\n".join(chunk_lines))

    # ── Web search results ────────────────────────────────────────────────────
    if plan_result.used_web_search and plan_result.web_context:
        sections.append("═══ LIVE WEB DATA (real-time, may override RAG) ═══\n"
                        + plan_result.web_context)

    # ── University recommendations ────────────────────────────────────────────
    if plan_result.used_recommender and plan_result.recommend_result:
        rec = plan_result.recommend_result
        unis = rec.get("universities", [])

        if unis:
            rec_lines = ["═══ UNIVERSITY RECOMMENDATIONS ═══",
                         f"Found {len(unis)} matching universities based on the student's profile:\n"]
            for i, u in enumerate(unis, 1):
                # Format each university as a readable entry
                name    = u.get("name",        "Unknown")
                city    = u.get("city",         "—")
                match   = u.get("match_reason", "")
                score   = u.get("match_score",  None)
                url     = u.get("url",          "")

                line = f"{i}. {name} ({city})"
                if score is not None:
                    line += f"  [match: {score:.0%}]"
                if match:
                    line += f"\n   Why: {match}"
                if url:
                    line += f"\n   Info: {url}"
                rec_lines.append(line)

            sections.append("\n".join(rec_lines))

    # If nothing was assembled (e.g. web search failed and no RAG) return a
    # safe fallback so the LLM still has something to work with.
    if not sections:
        return "No context retrieved. Please answer from your advisor knowledge."

    return "\n\n".join(sections)