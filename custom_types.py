# custom_types.py

from typing import Optional, Literal
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role:    Literal["user", "assistant"]
    content: str


# ---------------------------------------------------------------------------
# Metadata filter
# ---------------------------------------------------------------------------

class MetaFilter(BaseModel):
    filename:  Optional[str]       = None
    category:  Optional[str]       = None
    author:    Optional[str]       = None
    year_from: Optional[int]       = None
    year_to:   Optional[int]       = None
    tags:      Optional[list[str]] = None


# ---------------------------------------------------------------------------
# API response shapes
# ---------------------------------------------------------------------------

class IngestResult(BaseModel):
    success:             bool
    source:              str
    chunks:              int
    contextual_enriched: bool = False   # True when LLM context was generated


class DocumentListResult(BaseModel):
    documents: list[str]


class MatchItem(BaseModel):
    text:              str
    source:            str
    rrf_score:         Optional[float] = None
    vector_score:      Optional[float] = None
    rerank_score:      Optional[float] = None
    # What was actually sent to the LLM (may be expanded with neighbors)
    text_sent_to_llm:  Optional[str]   = None
    # Whether LLM-generated context prefix was prepended at embed time
    was_enriched:      bool            = False


class QueryResult(BaseModel):
    answer:              str
    sources:             list[str]
    num_contexts:        int
    retrieval_mode:      str
    rewritten_question:  Optional[str]  = None
    window_expanded:     bool           = False
    matches:             list[MatchItem]