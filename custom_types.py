# custom_types.py
#
# All shared Pydantic models.  Import from here rather than defining inline
# in main.py so both the API and tests share a single source of truth.

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Metadata filter — used in QueryRequest to narrow retrieval to a subset of
# the collection.  Every field is optional; only non-None fields are applied.
# ---------------------------------------------------------------------------

class MetaFilter(BaseModel):
    """
    Filter chunks by payload metadata before vector search.

    Qdrant evaluates ALL supplied conditions with AND logic (Qdrant `must`).
    Fields left as None are ignored — they do not restrict results.

    Examples
    --------
    # Only chunks from a specific file:
    MetaFilter(filename="report_q4.pdf")

    # All annual reports from 2022–2024:
    MetaFilter(category="annual_report", year_from=2022, year_to=2024)

    # Any document tagged "finance" OR "budget":
    MetaFilter(tags=["finance", "budget"])
    """

    filename:    Optional[str]       = None
    category:    Optional[str]       = None
    author:      Optional[str]       = None
    year_from:   Optional[int]       = None  # inclusive lower bound on `year`
    year_to:     Optional[int]       = None  # inclusive upper bound on `year`
    tags:        Optional[list[str]] = None  # match chunks that have ANY of these tags


# ---------------------------------------------------------------------------
# API response shapes — wired into FastAPI as response_model so the OpenAPI
# docs stay accurate and responses are validated on the way out.
# ---------------------------------------------------------------------------

class IngestResult(BaseModel):
    success:  bool
    source:   str
    chunks:   int


class DocumentListResult(BaseModel):
    documents: list[str]


class MatchItem(BaseModel):
    text:          str
    source:        str
    rrf_score:     Optional[float] = None
    vector_score:  Optional[float] = None
    rerank_score:  Optional[float] = None


class QueryResult(BaseModel):
    answer:          str
    sources:         list[str]
    num_contexts:    int
    retrieval_mode:  str
    matches:         list[MatchItem]