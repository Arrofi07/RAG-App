# custom_types.py

from typing import Optional, Literal, Any
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role:    Literal["user", "assistant"]
    content: str


# ---------------------------------------------------------------------------
# User profile
# ---------------------------------------------------------------------------

class UserProfile(BaseModel):
    """
    Study-in-Germany advisor profile.
    All fields optional so the advisor works even with a partial profile.
    """
    name:                Optional[str] = None
    nationality:         Optional[str] = None
    current_country:     Optional[str] = None
    education_level:     Optional[str] = None
    field_of_study:      Optional[str] = None
    target_degree:       Optional[str] = None
    german_level:        Optional[str] = None
    english_level:       Optional[str] = None
    target_universities: Optional[str] = None
    target_cities:       Optional[str] = None
    intake_semester:     Optional[str] = None
    budget_monthly_eur:  Optional[int] = None
    visa_status:         Optional[str] = None
    application_status:  Optional[str] = None
    extra_notes:         Optional[str] = None


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
    contextual_enriched: bool = False


class DocumentListResult(BaseModel):
    documents: list[str]


class MatchItem(BaseModel):
    text:             str
    source:           str
    rrf_score:        Optional[float] = None
    vector_score:     Optional[float] = None
    rerank_score:     Optional[float] = None
    text_sent_to_llm: Optional[str]   = None
    was_enriched:     bool            = False


class QueryResult(BaseModel):
    answer:             str
    sources:            list[str]
    num_contexts:       int
    retrieval_mode:     str
    rewritten_question: Optional[str] = None
    window_expanded:    bool          = False
    matches:            list[MatchItem]


class ConversationMeta(BaseModel):
    id:         str
    title:      str
    created_at: str
    updated_at: str


class UserInfo(BaseModel):
    id:         str
    name:       str
    profile:    dict[str, Any]
    created_at: str