# tests/conftest.py
#
# Shared fixtures and environment setup for the entire test suite.
#
# PHILOSOPHY
# ──────────
# Every test in this suite is ISOLATED: no real Qdrant, no real LLM, no real
# HuggingFace downloads, no real SQLite file on disk.  We mock all I/O at the
# boundary so tests run in < 5 seconds on any machine with no internet.
#
# MOCK STRATEGY (why each mock exists)
# ─────────────────────────────────────
# embed()              → bge-m3 would download 2.3 GB on first run; returns
#                        deterministic 1024-dim vectors instead.
# rerank()             → cross-encoder would download 2.3 GB; returns candidates
#                        unchanged with a fake rerank_score.
# QdrantStorage        → requires a running Qdrant server; replaced with a simple
#                        in-memory dict so tests work offline.
# call_llm / _call_llm → requires a Gemini API key; returns a canned string.
# build_registry()     → reads .env; stubbed so no env vars are needed.

import os
import sys
import types
import pytest

# ── Make project root importable ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Minimal env so modules that read .env at import time don't blow up ────────
os.environ.setdefault("GEMINI_API_KEY",  "test-key-not-real")
os.environ.setdefault("JWT_SECRET_KEY",  "test-secret-32-chars-minimum-xx")
os.environ.setdefault("QDRANT_URL",      "http://localhost:6333")
os.environ.setdefault("HF_HUB_OFFLINE",  "1")   # never ping HuggingFace Hub


# ── Stub heavy ML packages before any project module imports them ─────────────
# This is faster and safer than patching after import — if a module does
# `from FlagEmbedding import BGEM3FlagModel` at the top level, we need
# the stub to already be in sys.modules before that import runs.

def _make_stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# FlagEmbedding stub — pretends BGEM3FlagModel is importable
class _FakeBGEM3:
    def __init__(self, *a, **kw): pass
    def encode(self, texts, **kw):
        import numpy as np
        n = len(texts)
        lw = [{str(i): 0.5 for i in range(5)} for _ in range(n)]
        return {
            "dense_vecs":      np.zeros((n, 1024), dtype="float32"),
            "lexical_weights": lw,
        }

_flag_mod = _make_stub_module("FlagEmbedding", BGEM3FlagModel=_FakeBGEM3)
sys.modules.setdefault("FlagEmbedding", _flag_mod)

# sentence_transformers stub — pretends CrossEncoder is importable
class _FakeCrossEncoder:
    def __init__(self, *a, **kw): pass
    def predict(self, pairs):
        return [0.9 - i * 0.1 for i in range(len(pairs))]

_st_mod = _make_stub_module("sentence_transformers", CrossEncoder=_FakeCrossEncoder)
sys.modules.setdefault("sentence_transformers", _st_mod)

# llama_index stubs
for _m in [
    "llama_index", "llama_index.core",
    "llama_index.core.node_parser",
    "llama_index.readers", "llama_index.readers.file",
]:
    sys.modules.setdefault(_m, _make_stub_module(_m))

class _FakeSplitter:
    def __init__(self, **kw): pass
    def split_text(self, text): return [text] if text else []

sys.modules["llama_index.core.node_parser"].SentenceSplitter = _FakeSplitter

class _FakePDFReader:
    def load_data(self, file):
        class _Doc:
            text = "Sample PDF content for testing."
            metadata = {"page_label": "1"}
        return [_Doc()]

sys.modules["llama_index.readers.file"].PDFReader = _FakePDFReader

# google.genai stubs (Gemini SDK)
_google     = _make_stub_module("google")
_genai_mod  = _make_stub_module("google.genai")
_types_mod  = _make_stub_module("google.genai.types")
_errors_mod = _make_stub_module("google.genai.errors")

class _FakeGenaiClient:
    def __init__(self, **kw): pass
    class models:
        @staticmethod
        def generate_content(model, contents, tools=None, **kw):
            class _R:
                text = "Mocked LLM response."
                candidates = []
            return _R()

_genai_mod.Client = _FakeGenaiClient
_google.genai = _genai_mod
sys.modules.setdefault("google",            _google)
sys.modules.setdefault("google.genai",      _genai_mod)
sys.modules.setdefault("google.genai.types", _types_mod)
sys.modules.setdefault("google.genai.errors", _errors_mod)

# jose / passlib — real packages; only stub if not installed
try:
    import jose          # noqa
    import passlib       # noqa
except ImportError:
    sys.modules.setdefault("jose",            _make_stub_module("jose"))
    sys.modules.setdefault("jose.jwt",        _make_stub_module("jose.jwt"))
    sys.modules.setdefault("passlib",         _make_stub_module("passlib"))
    sys.modules.setdefault("passlib.context", _make_stub_module("passlib.context"))


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture()
def fake_llm():
    """A callable that mimics call_llm — returns a canned string."""
    def _llm(prompt: str, **kw) -> str:
        if "intent" in prompt.lower() or "router" in prompt.lower():
            return "rag_only"
        if "rewrite" in prompt.lower():
            return "What are the visa requirements for Germany?"
        if "search query" in prompt.lower():
            return "Germany student visa requirements 2026"
        return "Mocked LLM response."
    return _llm


@pytest.fixture()
def sample_profile() -> dict:
    """A realistic student profile for parameterised tests."""
    return {
        "name":                "Budi Santoso",
        "nationality":         "Indonesian",
        "current_country":     "Indonesia",
        "education_level":     "Bachelor's graduate",
        "field_of_study":      "Computer Science",
        "target_degree":       "Master's",
        "german_level":        "None (complete beginner)",
        "english_level":       "IELTS 7.0",
        "target_universities": "TU Munich, KIT",
        "target_cities":       "Munich, Berlin",
        "intake_semester":     "Winter 2026/27",
        "budget_monthly_eur":  900,
        "visa_status":         "Not started yet",
        "application_status":  "Just researching",
        "extra_notes":         "",
    }


@pytest.fixture()
def in_memory_storage(tmp_path):
    """A Storage instance backed by a temporary SQLite file."""
    from storage import Storage
    return Storage(db_path=tmp_path / "test.db")