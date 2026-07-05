# tests/test_api.py
#
# Contract tests for the FastAPI endpoints via TestClient.
#
# APPROACH
# ────────
# We use FastAPI's TestClient (Starlette's test client, built on httpx) so
# every test hits the real routing/validation/serialisation stack.
#
# Heavy dependencies (Qdrant, bge-m3, Gemini, Ollama) are all patched at the
# module level BEFORE the app is imported.  This means:
#   - No models are downloaded
#   - No servers need to be running
#   - Tests run in < 2 seconds total
#
# WHAT WE TEST
# ────────────
# - Correct HTTP status codes for valid and invalid requests
# - Response payload shapes match the Pydantic response_model
# - Auth is enforced on protected endpoints (401 without token)
# - Error cases return sensible messages

import os
import pytest
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# App fixture — patch everything heavy BEFORE importing main
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """
    Build a FastAPI TestClient with all external dependencies mocked out.
    Scoped to 'module' so we pay the (small) startup cost once per file.
    """
    tmp = tmp_path_factory.mktemp("data")

    # Patch list — each tuple is (target, replacement)
    patches = [
        # Storage — use a real SQLite but in a temp dir
        ("storage.Storage", lambda db_path=None: __import__("storage").Storage(
            db_path=tmp / "test.db"
        )),
        # Qdrant — stub the entire QdrantStorage class
        ("vector_db.QdrantStorage", _MockQdrantStorage),
        # LLM registry — don't read .env, don't instantiate real providers
        ("llm_providers.build_registry", lambda: {}),
        ("llm_providers.check_providers", lambda: {}),
        ("llm_providers.call_llm", lambda prompt, role="answer", **kw: "Mocked LLM answer."),
        # Embedding model — return zeros, no download
        ("data_loader.embed", _fake_embed),
        ("data_loader._get_embed_model", lambda: None),
        # Reranker — return candidates unchanged
        ("reranker.rerank", lambda q, cands, top_k=5: _fake_rerank(q, cands, top_k)),
        ("reranker.warmup", lambda: None),
        # University recommender — return empty results
        ("university_recommender.UniversityRecommender", _MockRecommender),
        # Context builder — no-op
        ("context_builder.generate_chunk_context", lambda *a, **kw: ""),
        ("context_builder.build_contextual_text",  lambda ctx, text: text),
        # data_loader.load_chunks — return fake chunks without touching disk
        ("data_loader.load_chunks", _fake_load_chunks),
    ]

    with _apply_patches(patches):
        from fastapi.testclient import TestClient
        import main as app_module
        test_client = TestClient(app_module.app, raise_server_exceptions=False)
        yield test_client


# ── Mock helpers ──────────────────────────────────────────────────────────────

class _MockQdrantStorage:
    def __init__(self, **kw):
        self._data = []

    def upsert(self, ids, dense_vectors, sparse_vectors, payloads):
        for i, pid in enumerate(ids):
            self._data.append({"id": pid, "payload": payloads[i]})

    def search_hybrid_candidates(self, *a, **kw):
        return [{"id": "1", "text": "Germany requires a blocked account.",
                 "source": "visa_guide.pdf", "rrf_score": 0.8,
                 "prev_chunk": "", "next_chunk": "", "contextual_text": None}]

    def search_candidates(self, *a, **kw):
        return self.search_hybrid_candidates()

    def list_documents(self):
        sources = list({p["payload"].get("filename") for p in self._data
                        if p["payload"].get("filename")})
        return [{"filename": s, "category": None, "author": None,
                 "year": None, "tags": []} for s in sources]


class _MockRecommender:
    def __init__(self, **kw): pass
    def ensure_seeded(self, **kw): pass
    def compute_missing_embeddings(self, **kw): return 0
    def recommend(self, **kw):
        return {"universities": [], "total_filtered": 0, "profile_used": {}}


def _fake_embed(texts):
    import numpy as np
    n = len(texts)
    return {
        "dense":  [[0.0] * 1024 for _ in range(n)],
        "sparse": [{"indices": [1, 2], "values": [0.5, 0.5]} for _ in range(n)],
    }


def _fake_rerank(query, candidates, top_k=5):
    for i, c in enumerate(candidates):
        c["rerank_score"] = 0.9 - i * 0.1
    return candidates[:top_k]


def _fake_load_chunks(path):
    return {
        "chunks": [
            {"text": "Test chunk 1.", "prev_chunk": "", "next_chunk": "Test chunk 2.",
             "chunk_index": 0, "page": "1"},
            {"text": "Test chunk 2.", "prev_chunk": "Test chunk 1.", "next_chunk": "",
             "chunk_index": 1, "page": "1"},
        ],
        "full_text": "Test chunk 1. Test chunk 2.",
    }


from contextlib import contextmanager, ExitStack

@contextmanager
def _apply_patches(patch_list):
    with ExitStack() as stack:
        for target, new in patch_list:
            try:
                stack.enter_context(patch(target, new))
            except AttributeError:
                pass   # module not yet imported — conftest stubs handle it
        yield


# ─────────────────────────────────────────────────────────────────────────────
# Auth endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestAuthEndpoints:

    def test_register_returns_201_and_token(self, client):
        resp = client.post("/auth/register", json={
            "name": "Test Student",
            "email": "test@example.com",
            "password": "SecurePass1",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "user_id" in data

    def test_register_duplicate_email_returns_409(self, client):
        payload = {"name": "Dup", "email": "dup@example.com", "password": "SecurePass1"}
        client.post("/auth/register", json=payload)
        resp = client.post("/auth/register", json=payload)
        assert resp.status_code == 409

    def test_register_weak_password_returns_422_or_400(self, client):
        resp = client.post("/auth/register", json={
            "name": "Weak", "email": "weak@example.com", "password": "123",
        })
        assert resp.status_code in (400, 422)

    def test_login_with_correct_credentials(self, client):
        client.post("/auth/register", json={
            "name": "Login Test", "email": "login@example.com", "password": "GoodPass1",
        })
        resp = client.post("/auth/login", json={
            "email": "login@example.com", "password": "GoodPass1",
        })
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_login_wrong_password_returns_401(self, client):
        client.post("/auth/register", json={
            "name": "WrongPW", "email": "wrongpw@example.com", "password": "RightPass1",
        })
        resp = client.post("/auth/login", json={
            "email": "wrongpw@example.com", "password": "WrongPass1",
        })
        assert resp.status_code == 401

    def test_login_unknown_email_returns_401(self, client):
        resp = client.post("/auth/login", json={
            "email": "nobody@example.com", "password": "AnyPass1",
        })
        assert resp.status_code == 401

    def _get_token(self, client):
        email = "auth_helper@example.com"
        client.post("/auth/register", json={
            "name": "Auth", "email": email, "password": "Helper123",
        })
        r = client.post("/auth/login", json={"email": email, "password": "Helper123"})
        return r.json()["access_token"]

    def test_me_endpoint_returns_user_info(self, client):
        token = self._get_token(client)
        resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert "user_id" in resp.json()

    def test_me_without_token_returns_401(self, client):
        resp = client.get("/auth/me")
        assert resp.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# Protected endpoint gate
# ─────────────────────────────────────────────────────────────────────────────

class TestAuthGate:

    def test_documents_requires_auth(self, client):
        resp = client.get("/documents")
        assert resp.status_code == 401

    def test_documents_with_token_succeeds(self, client):
        email = "docuser@example.com"
        client.post("/auth/register", json={
            "name": "DocUser", "email": email, "password": "DocPass1",
        })
        r = client.post("/auth/login", json={"email": email, "password": "DocPass1"})
        token = r.json()["access_token"]
        resp = client.get("/documents", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Query endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryEndpoint:

    def test_query_returns_answer(self, client):
        resp = client.post("/query", json={"question": "What is a Studienkolleg?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_query_response_shape(self, client):
        resp = client.post("/query", json={"question": "Visa requirements?"})
        data = resp.json()
        for field in ("answer", "sources", "num_contexts", "retrieval_mode", "matches"):
            assert field in data, f"Missing field: {field}"

    def test_empty_question_returns_400(self, client):
        resp = client.post("/query", json={"question": ""})
        assert resp.status_code == 400

    def test_query_with_history(self, client):
        resp = client.post("/query", json={
            "question": "What about the blocked account?",
            "history": [
                {"role": "user",      "content": "Tell me about studying in Germany."},
                {"role": "assistant", "content": "Germany is a popular destination..."},
            ],
        })
        assert resp.status_code == 200

    def test_force_intent_accepted(self, client):
        resp = client.post("/query", json={
            "question": "Latest visa fee?",
            "force_intent": "web_search",
        })
        assert resp.status_code == 200

    def test_dense_only_mode(self, client):
        resp = client.post("/query", json={
            "question": "What is a Studienkolleg?",
            "use_hybrid": False,
        })
        assert resp.status_code == 200
        assert resp.json()["retrieval_mode"] == "dense"


# ─────────────────────────────────────────────────────────────────────────────
# Ingest endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestEndpoint:

    def _fake_pdf_bytes(self) -> bytes:
        """Minimal bytes that passes the filename check (not real PDF parsing)."""
        return b"%PDF-1.4 fake content for testing"

    def test_ingest_valid_pdf(self, client):
        resp = client.post(
            "/ingest",
            files={"file": ("test.pdf", self._fake_pdf_bytes(), "application/pdf")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["source"] == "test.pdf"
        assert data["chunks"] > 0

    def test_ingest_non_pdf_returns_400(self, client):
        resp = client.post(
            "/ingest",
            files={"file": ("doc.docx", b"not a pdf", "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_ingest_with_metadata(self, client):
        resp = client.post(
            "/ingest",
            files={"file": ("meta_test.pdf", self._fake_pdf_bytes(), "application/pdf")},
            data={
                "category": "visa",
                "author":   "DAAD",
                "year":     "2024",
                "tags":     "official,scholarship",
            },
        )
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Recommend endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestRecommendEndpoint:

    def test_recommend_returns_universities_list(self, client):
        resp = client.post("/recommend", json={
            "profile": {
                "field_of_study":    "Computer Science",
                "target_degree":     "Master's",
                "german_level":      "None (complete beginner)",
                "budget_monthly_eur": 900,
            },
            "top_k": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "universities" in data

    def test_recommend_empty_profile_does_not_crash(self, client):
        resp = client.post("/recommend", json={"profile": {}})
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthCheck:

    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "status" in resp.json()