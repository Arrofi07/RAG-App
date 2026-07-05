# tests/test_llm_providers.py
#
# Tests for llm_providers.py — retry logic, error parsing, registry wiring.
#
# All network calls are replaced with mocks that raise or return strings.

import os
import pytest
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# GeminiProvider — retry delay parser
# ─────────────────────────────────────────────────────────────────────────────

class TestGeminiRetryDelayParser:
    """
    _parse_retry_delay() must extract seconds from the string representation
    of a google-genai exception, which serialises as:
        "429 RESOURCE_EXHAUSTED. {'error': {'details': [{'retryDelay': '22s'}]}}"
    """

    def _parser(self):
        from llm_providers import GeminiProvider
        return GeminiProvider._parse_retry_delay

    def test_parses_integer_delay(self):
        exc = Exception("429 RESOURCE_EXHAUSTED. {'error': {'details': [{'retryDelay': '22s'}]}}")
        assert self._parser()(exc) == 22.0

    def test_parses_fractional_delay(self):
        exc = Exception("429 RESOURCE_EXHAUSTED. {'error': {'details': [{'retryDelay': '22.685s'}]}}")
        assert self._parser()(exc) == pytest.approx(22.685)

    def test_returns_zero_when_absent(self):
        exc = Exception("Some other error with no retryDelay")
        assert self._parser()(exc) == 0.0

    def test_handles_non_string_exception(self):
        exc = Exception({"code": 429})
        assert self._parser()(exc) == 0.0   # fallback, no crash


# ─────────────────────────────────────────────────────────────────────────────
# GeminiProvider — HTTP code extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestGeminiHttpCode:

    def _extractor(self):
        from llm_providers import GeminiProvider
        return GeminiProvider._http_code

    def test_extracts_429(self):
        exc = Exception("429 RESOURCE_EXHAUSTED. details...")
        assert self._extractor()(exc) == 429

    def test_extracts_503(self):
        exc = Exception("503 UNAVAILABLE. details...")
        assert self._extractor()(exc) == 503

    def test_extracts_404(self):
        exc = Exception("404 NOT_FOUND. model not found")
        assert self._extractor()(exc) == 404

    def test_returns_none_for_unknown_format(self):
        exc = Exception("Something unexpected")
        assert self._extractor()(exc) is None


# ─────────────────────────────────────────────────────────────────────────────
# GeminiProvider — cascade and retry logic
# ─────────────────────────────────────────────────────────────────────────────

class TestGeminiProviderCascade:

    def _make_provider(self, models=None):
        from llm_providers import GeminiProvider
        p = GeminiProvider.__new__(GeminiProvider)
        p.models = models or ["model-a", "model-b"]
        return p

    def test_returns_first_model_response_on_success(self):
        provider = self._make_provider()

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(text="Hello!")
        provider._client = mock_client

        result = provider.call("What is a Studienkolleg?")
        assert result == "Hello!"
        assert mock_client.models.generate_content.call_count == 1

    def test_cascades_to_second_model_on_404(self):
        """404 (model not found) is non-transient — skip immediately, no retry."""
        provider = self._make_provider(["model-a", "model-b"])

        responses = [
            Exception("404 NOT_FOUND. model-a is not available"),
            MagicMock(text="Fallback answer"),
        ]
        call_count = [0]

        def fake_generate(model, contents, **kw):
            idx = call_count[0]
            call_count[0] += 1
            r = responses[idx]
            if isinstance(r, Exception):
                raise r
            return r

        provider._client = MagicMock()
        provider._client.models.generate_content.side_effect = fake_generate

        with patch("time.sleep"):   # don't actually wait
            result = provider.call("test prompt")

        assert result == "Fallback answer"

    def test_raises_when_all_models_exhausted(self):
        provider = self._make_provider(["model-a"])
        provider._client = MagicMock()
        provider._client.models.generate_content.side_effect = \
            Exception("503 UNAVAILABLE.")

        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="failed"):
                provider.call("test prompt")


# ─────────────────────────────────────────────────────────────────────────────
# OllamaProvider
# ─────────────────────────────────────────────────────────────────────────────

class TestOllamaProvider:

    def _make_provider(self):
        from llm_providers import OllamaProvider
        return OllamaProvider(
            model="qwen3:1.7b",
            base_url="http://localhost:11434",
            timeout=5,
        )

    def test_returns_response_text(self):
        import requests as req_mod
        provider = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"response": "Mocked Ollama response"}

        with patch.object(req_mod, "post", return_value=mock_resp):
            result = provider._raw_call("test prompt")

        assert result == "Mocked Ollama response"

    def test_raises_on_model_not_pulled(self):
        import requests as req_mod
        provider = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.text = "model 'qwen3:1.7b' isn't pulled yet"

        with patch.object(req_mod, "post", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="isn't pulled yet"):
                provider._raw_call("test prompt")

    def test_health_check_true_when_running(self):
        import requests as req_mod
        provider = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"models": [{"name": "qwen3:1.7b"}]}

        with patch.object(req_mod, "get", return_value=mock_resp):
            assert provider.health_check() is True

    def test_health_check_false_when_connection_error(self):
        import requests as req_mod
        provider = self._make_provider()
        with patch.object(req_mod, "get", side_effect=ConnectionError):
            assert provider.health_check() is False


# ─────────────────────────────────────────────────────────────────────────────
# Provider registry
# ─────────────────────────────────────────────────────────────────────────────

class TestProviderRegistry:

    def test_build_registry_creates_three_roles(self):
        from llm_providers import build_registry, _registry
        with patch.dict(os.environ, {
            "GEMINI_API_KEY":      "test-key",
            "ENRICHMENT_PROVIDER": "ollama",
            "ENRICHMENT_MODEL":    "qwen3:1.7b",
            "PLANNING_PROVIDER":   "ollama",
            "PLANNING_MODEL":      "qwen3:1.7b",
            "ANSWER_PROVIDER":     "gemini",
            "ANSWER_MODELS":       "gemini-2.5-flash-lite",
        }):
            registry = build_registry()

        assert "enrichment" in registry
        assert "planning"   in registry
        assert "answer"     in registry

    def test_get_provider_after_build(self):
        from llm_providers import build_registry, get_provider
        with patch.dict(os.environ, {
            "GEMINI_API_KEY":  "test-key",
            "ANSWER_PROVIDER": "gemini",
            "ANSWER_MODELS":   "gemini-2.5-flash-lite",
        }):
            build_registry()
            p = get_provider("answer")
        assert p is not None

    def test_get_provider_unknown_role_raises(self):
        from llm_providers import build_registry, get_provider
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            build_registry()
        with pytest.raises(KeyError):
            get_provider("nonexistent_role")

    def test_call_llm_routes_to_correct_provider(self):
        from llm_providers import build_registry, call_llm
        with patch.dict(os.environ, {
            "GEMINI_API_KEY":  "test-key",
            "ANSWER_PROVIDER": "gemini",
            "ANSWER_MODELS":   "gemini-2.5-flash-lite",
        }):
            build_registry()

        # Patch the answer provider's call method directly
        from llm_providers import get_provider
        provider = get_provider("answer")
        with patch.object(provider, "call", return_value="mocked answer") as mock_call:
            result = call_llm("hello", role="answer")

        mock_call.assert_called_once_with("hello")
        assert result == "mocked answer"