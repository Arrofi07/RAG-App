# context_builder.py
#
# Contextual chunk enrichment — inspired by Anthropic's "Contextual Retrieval"
# https://www.anthropic.com/news/contextual-retrieval
#
# Core idea: embedding models have no awareness of where a chunk sits within
# a document.  A chunk saying "revenue increased by 12%" is ambiguous without
# knowing which document, which section, and which time period it refers to.
#
# Solution: before embedding, prepend a short LLM-generated context sentence
# that situates the chunk within its source document.  The embedding then
# captures both the chunk's content and its document position, making
# retrieval dramatically more accurate — especially for chunks that contain
# numbers, pronouns, or references that only make sense in context.
#
# This is an ingest-time operation.  The LLM is called once per chunk, so
# expect ingest to be slower when enabled.  The original chunk text is always
# preserved separately for display.

import logging
from typing import Callable

log = logging.getLogger(__name__)

# How many characters of the full document to feed the LLM for context.
# Longer → richer context, but higher token cost and latency.
DOC_PREVIEW_CHARS = 3000


def generate_chunk_context(
    full_doc_text: str,
    chunk_text: str,
    source: str,
    llm_fn: Callable[[str], str],
) -> str:
    """
    Ask the LLM to write a short, factual context sentence for a chunk.

    The context sentence is prepended to the chunk before embedding —
    it is NOT shown to the user; it only improves vector search accuracy.

    Args:
        full_doc_text : complete text of the source document (used as context)
        chunk_text    : the individual chunk to situate
        source        : filename / document identifier (for the prompt)
        llm_fn        : callable that takes a prompt string and returns a string

    Returns:
        A 1-3 sentence context string, or "" on failure (caller uses raw chunk).
    """
    doc_preview = full_doc_text[:DOC_PREVIEW_CHARS]

    prompt = f"""You are helping to index document chunks for a retrieval system.

Document name: {source}

Document preview (first part of the document):
<document>
{doc_preview}
</document>

Chunk to situate:
<chunk>
{chunk_text}
</chunk>

Write 1-3 short sentences that:
1. Identify the section or topic this chunk belongs to within the document.
2. Clarify what this chunk is specifically about in the document's context.
3. Note any key entities, dates, or figures that give this chunk meaning.

Rules:
- Output ONLY the context sentences, no preamble or explanation.
- Do NOT repeat the chunk content verbatim — just contextualise it.
- Be factual and concise.

Context:"""

    try:
        context = llm_fn(prompt).strip()
        return context
    except Exception as e:
        log.warning("Context generation failed for chunk from '%s': %s", source, e)
        return ""


def build_contextual_text(context: str, chunk_text: str) -> str:
    """
    Combine the LLM-generated context with the raw chunk text.

    This is the string that gets embedded — NOT what the user sees.
    The raw chunk_text is always stored separately in the payload.
    """
    if not context:
        return chunk_text

    return f"{context}\n\n{chunk_text}"