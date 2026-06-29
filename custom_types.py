# custom_types.py  (v8.0.0)
#
# Shared Pydantic models for the FastAPI backend and Streamlit frontend.
#
# V8 CHANGES:
# - QueryResult gets two new optional fields:
#     intent          : which plan the agentic planner chose
#     web_sources     : list of URLs found by the web search tool
# - UniversityMatch  : new model for recommender output
# - RecommendResult  : wrapper for the /recommend endpoint response

from typing import Optional, Literal, Any
from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# Conversation
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role:    Literal["user", "assistant"]
    content: str


# ─────────────────────────────────────────────────────────────────────────────
# User profile
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Metadata filter (for RAG retrieval)
# ─────────────────────────────────────────────────────────────────────────────

class MetaFilter(BaseModel):
    filename:  Optional[str]       = None
    category:  Optional[str]       = None
    author:    Optional[str]       = None
    year_from: Optional[int]       = None
    year_to:   Optional[int]       = None
    tags:      Optional[list[str]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Ingest response
# ─────────────────────────────────────────────────────────────────────────────

class IngestResult(BaseModel):
    success:             bool
    source:              str
    chunks:              int
    contextual_enriched: bool = False


class DocumentListResult(BaseModel):
    documents: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Query response (RAG / planner)
# ─────────────────────────────────────────────────────────────────────────────

class WebSource(BaseModel):
    """A single web search result citation."""
    title: str = ""
    url:   str = ""


class MatchItem(BaseModel):
    """One retrieved RAG chunk with its retrieval scores."""
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
    # New v8 fields — optional so v7 clients still work
    intent:             Optional[str]           = None  # planner intent used
    web_sources:        Optional[list[WebSource]] = None  # live web citations


# ─────────────────────────────────────────────────────────────────────────────
# University recommender response
# ─────────────────────────────────────────────────────────────────────────────

class UniversityMatch(BaseModel):
    """
    A single university recommendation from the hybrid recommender.
    All fields except name/city are optional because some programs in the
    seed data may not have every attribute filled in.
    """
    name:             str
    short_name:       Optional[str]  = None
    city:             str
    state:            Optional[str]  = None
    url:              Optional[str]  = None
    degree_type:      Optional[str]  = None
    field:            Optional[str]  = None
    program_name:     Optional[str]  = None
    language:         Optional[str]  = None
    semester_fee_eur: Optional[int]  = None
    german_required:  Optional[str]  = None
    english_required: Optional[str]  = None
    research_areas:   Optional[str]  = None
    strengths:        Optional[str]  = None
    winter_deadline:  Optional[str]  = None
    summer_deadline:  Optional[str]  = None
    match_score:      Optional[float] = None   # 0.0–1.0
    match_reason:     Optional[str]  = None


class RecommendResult(BaseModel):
    universities:   list[UniversityMatch]
    total_filtered: int   # how many passed SQL hard-filter before vector ranking
    profile_used:   dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# User / conversation metadata
# ─────────────────────────────────────────────────────────────────────────────

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