# custom_types.py

from typing import Optional, Literal
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """A single turn in the conversation history."""
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
    success: bool
    source:  str
    chunks:  int


class DocumentListResult(BaseModel):
    documents: list[str]


class MatchItem(BaseModel):
    text:         str
    source:       str
    rrf_score:    Optional[float] = None
    vector_score: Optional[float] = None
    rerank_score: Optional[float] = None


class QueryResult(BaseModel):
    answer:            str
    sources:           list[str]
    num_contexts:      int
    retrieval_mode:    str
    rewritten_question: Optional[str] = None   # None when history is empty
    matches:           list[MatchItem]