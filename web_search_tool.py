# web_search_tool.py
#
# Real-time web search for the Study-in-Germany AI Advisor.
#
# WHY THIS EXISTS
# ───────────────
# University application deadlines, visa fee amounts, blocked-account
# thresholds, and scholarship deadlines change every semester.  The RAG
# knowledge base (your ingested PDFs) goes stale quickly.  This module
# gives the advisor a "live" fallback: when the planner decides a question
# needs current data, it calls search_web() and the results are injected
# into the LLM prompt alongside the RAG context.
#
# ARCHITECTURE
# ────────────
# We use the Gemini SDK's built-in grounding tool (Google Search) rather
# than a third-party search API.  This means:
#   - No extra API key or service account needed
#   - The model itself decides which snippets to cite
#   - Results are already summarised; we don't need to parse HTML
#
# MEMORY CONSTRAINT (M2 / 8 GB)
# ───────────────────────────────
# This module is pure Python + one HTTP call.  It loads nothing into RAM,
# so it has zero impact on the local model memory budget.

import os
import logging
import re
from typing import Optional

from google import genai
from google.genai import types

log = logging.getLogger(__name__)

# ─── Gemini client (reuses the same API key as main.py) ──────────────────────

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    """Lazy-initialise the Gemini client once and reuse it."""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        _client = genai.Client(api_key=api_key)
    return _client


# ─── Search model ────────────────────────────────────────────────────────────

# We use flash-lite for search summarisation: it's fast, cheap on quota,
# and grounding already handles the "find facts" part — we just need the
# model to synthesise the snippets into a clean answer.
SEARCH_MODEL = os.getenv("SEARCH_MODEL", "gemini-2.5-flash-lite")


# ─── Public API ──────────────────────────────────────────────────────────────

def search_web(query: str, max_results: int = 5) -> dict:
    """
    Run a live Google Search and return a structured result dict.

    HOW IT WORKS
    ────────────
    1. We ask Gemini to answer `query` with the `google_search` tool enabled.
    2. Gemini issues search queries, receives snippets, and writes a grounded
       answer.  The SDK returns both the answer text and the raw search
       candidates (URLs + titles + snippets).
    3. We extract the answer + up to `max_results` source citations and
       return them so the planner can decide how much to include in the
       final LLM prompt.

    RETURN VALUE
    ────────────
    {
        "answer":  str,          # Gemini's grounded answer (already synthesised)
        "sources": [             # raw search results for citation display
            {
                "title": str,
                "url":   str,
                "snippet": str,
            },
            ...
        ],
        "raw_query": str,        # the query we actually sent (for debugging)
        "error":    str | None,  # non-None if the search failed
    }
    """
    log.info("🔍 Web search: '%s'", query)

    try:
        client = _get_client()

        # The google_search tool tells Gemini to retrieve live web results
        # before composing its answer.  This is called "grounding."
        response = client.models.generate_content(
            model=SEARCH_MODEL,
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                # Keep the summarised answer concise so it fits in the main prompt
                max_output_tokens=512,
            ),
        )

        # ── Extract the synthesised answer text ──────────────────────────────
        answer_text = ""
        if hasattr(response, "text") and response.text:
            answer_text = response.text.strip()

        # ── Extract source citations from grounding metadata ─────────────────
        # The SDK returns grounding_metadata inside candidates[0].
        # We pull out web search result chunks (title, url, snippet).
        sources = []
        try:
            candidate = response.candidates[0]
            grounding = getattr(candidate, "grounding_metadata", None)
            if grounding:
                # grounding_chunks contains the actual retrieved documents
                chunks = getattr(grounding, "grounding_chunks", []) or []
                for chunk in chunks[:max_results]:
                    web = getattr(chunk, "web", None)
                    if web:
                        sources.append({
                            "title":   getattr(web, "title",   "") or "",
                            "url":     getattr(web, "uri",     "") or "",
                            "snippet": "",  # snippets live in search_entry_point
                        })
        except Exception as meta_err:
            # Metadata extraction is best-effort; a missing citation doesn't
            # break the answer.
            log.debug("Could not extract grounding metadata: %s", meta_err)

        log.info("✅ Web search returned %d sources for: '%s'", len(sources), query)

        return {
            "answer":    answer_text,
            "sources":   sources,
            "raw_query": query,
            "error":     None,
        }

    except Exception as exc:
        log.warning("Web search failed for '%s': %s", query, exc)
        return {
            "answer":    "",
            "sources":   [],
            "raw_query": query,
            "error":     str(exc),
        }


def format_search_result_for_prompt(result: dict) -> str:
    """
    Format a search_web() result into a compact block suitable for injection
    into the main LLM prompt alongside the RAG context.

    The planner calls this when it decides the LLM should see live web data.
    Keeping the format consistent makes it easy for the LLM to distinguish
    RAG chunks (numbered [1], [2]…) from live web results (marked ⟨WEB⟩).
    """
    if result.get("error") or not result.get("answer"):
        return f"⟨WEB SEARCH FAILED for: {result['raw_query']}⟩"

    lines = [f"⟨WEB — {result['raw_query']}⟩", result["answer"]]

    # Add source URLs so the advisor can tell the student where to verify
    if result["sources"]:
        lines.append("\nSources:")
        for i, src in enumerate(result["sources"], 1):
            title = src["title"] or src["url"]
            url   = src["url"]
            if url:
                lines.append(f"  [{i}] {title} — {url}")

    return "\n".join(lines)


# ─── Query classifier helpers ─────────────────────────────────────────────────
# These are used by the planner (planner.py) to decide whether a question
# needs a live web search before answering.

# Keywords that strongly suggest real-time information is needed.
# A question containing any of these is routed to web search.
_LIVE_DATA_KEYWORDS = [
    # Temporal signals
    "2026", "2027", "current", "latest", "now", "today", "this year",
    "this semester", "upcoming", "recent",
    # Deadline language
    "deadline", "due date", "when is", "last date", "closing date",
    "application period", "opens", "closes",
    # Rapidly changing info
    "blocked account", "sperrkonto", "amount", "how much",
    "fee", "visa fee", "embassy fee",
    "ielts requirement", "toefl requirement",
    "ranking", "qs ranking", "times ranking",
]


def needs_web_search(question: str) -> bool:
    """
    Heuristic check: does this question likely need live web data?

    Returns True if the question contains any live-data keyword.
    The planner uses this as a fast pre-filter before asking the LLM
    to classify intent (which costs an LLM call).

    This is intentionally permissive — false positives just mean an
    extra search call; false negatives mean stale data in the answer.
    """
    q_lower = question.lower()
    return any(kw in q_lower for kw in _LIVE_DATA_KEYWORDS)