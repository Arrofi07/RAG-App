# tests/test_vector_db.py
#
# Tests for vector_db.py — MetaFilter → Qdrant filter translation.
#
# We only test _build_filter() (pure logic, no Qdrant connection needed).
# The search/upsert methods require a live Qdrant server and are covered
# by integration tests (not in this suite).

import pytest


def _build_filter(meta_dict: dict | None):
    """Helper: build a MetaFilter from a dict and call _build_filter."""
    from vector_db import QdrantStorage
    from custom_types import MetaFilter
    from unittest.mock import patch, MagicMock

    # Construct the storage object without connecting to Qdrant
    storage = QdrantStorage.__new__(QdrantStorage)
    storage.collection = "docs"
    storage.dim        = 1024

    meta = MetaFilter(**meta_dict) if meta_dict is not None else None
    return storage._build_filter(meta)


class TestBuildFilter:

    def test_none_meta_returns_none(self):
        assert _build_filter(None) is None

    def test_empty_meta_returns_none(self):
        """All-None MetaFilter should produce no Qdrant filter."""
        result = _build_filter({})
        assert result is None

    def test_filename_filter(self):
        f = _build_filter({"filename": "visa_guide.pdf"})
        assert f is not None
        must = f.must
        assert len(must) == 1
        assert must[0].key == "filename"
        assert must[0].match.value == "visa_guide.pdf"

    def test_category_filter(self):
        f = _build_filter({"category": "visa"})
        assert f is not None
        assert f.must[0].key == "category"

    def test_author_filter(self):
        f = _build_filter({"author": "DAAD"})
        assert f is not None
        assert f.must[0].key == "author"

    def test_tags_filter_uses_match_any(self):
        from qdrant_client.models import MatchAny
        f = _build_filter({"tags": ["visa", "scholarship"]})
        assert f is not None
        cond = f.must[0]
        assert cond.key == "tags"
        assert isinstance(cond.match, MatchAny)
        assert set(cond.match.any) == {"visa", "scholarship"}

    def test_year_range_both_bounds(self):
        from qdrant_client.models import Range
        f = _build_filter({"year_from": 2022, "year_to": 2024})
        assert f is not None
        cond = f.must[0]
        assert cond.key == "year"
        assert isinstance(cond.range, Range)
        assert cond.range.gte == 2022
        assert cond.range.lte == 2024

    def test_year_range_lower_bound_only(self):
        from qdrant_client.models import Range
        f = _build_filter({"year_from": 2023})
        cond = f.must[0]
        assert cond.range.gte == 2023
        assert cond.range.lte is None

    def test_year_range_upper_bound_only(self):
        from qdrant_client.models import Range
        f = _build_filter({"year_to": 2024})
        cond = f.must[0]
        assert cond.range.gte is None
        assert cond.range.lte == 2024

    def test_multiple_filters_combine_as_must(self):
        """filename + category → two must clauses (AND logic)."""
        f = _build_filter({"filename": "guide.pdf", "category": "visa"})
        assert f is not None
        assert len(f.must) == 2
        keys = {c.key for c in f.must}
        assert "filename" in keys
        assert "category" in keys

    def test_all_filters_together(self):
        f = _build_filter({
            "filename":  "guide.pdf",
            "category":  "visa",
            "author":    "DAAD",
            "tags":      ["official"],
            "year_from": 2023,
            "year_to":   2025,
        })
        assert f is not None
        assert len(f.must) == 5   # one clause per non-None field

    def test_none_tags_do_not_add_clause(self):
        f = _build_filter({"filename": "x.pdf", "tags": None})
        assert len(f.must) == 1