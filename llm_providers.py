# llm_providers.py
#
# Unified LLM provider interface for the Study-in-Germany AI Advisor.
#
# DESIGN GOAL
# ───────────
# Every part of the system that needs an LLM (contextual enrichment, query
# rewriting, intent classification, final answer generation) calls ONE function:
#
#     text = call_llm(prompt, role="enrichment")   # uses local Qwen via Ollama
#     text = call_llm(prompt, role="answer")        # uses Gemini
#     text = call_llm(prompt, role="planning")      # uses whichever is configured
#
# The `role` argument maps to a provider+model configured in .env, so you can
# swap any role to any provider without touching application code.
#
# SUPPORTED PROVIDERS
# ───────────────────
# 1. gemini   — Google Gemini via google-genai SDK (cloud, needs API key)
# 2. ollama   — Any model served by Ollama (local, free, no API key)
#               Default models: Qwen3:1.7b (enrichment), Qwen3:8b (planning)
# 3. openai   — OpenAI or any OpenAI-compatible API (Kimi, DeepSeek, etc.)
#               Set OPENAI_BASE_URL to point at the right endpoint.
#
# ROLE → PROVIDER MAPPING (set in .env)
# ──────────────────────────────────────
# ENRICHMENT_PROVIDER=ollama           # contextual enrichment (local, free)
# ENRICHMENT_MODEL=qwen3:1.7b          # fast, tiny, good at summarisation
#
# PLANNING_PROVIDER=ollama             # intent classification, search queries
# PLANNING_MODEL=qwen3:1.7b            # same model is fine for short tasks
#
# ANSWER_PROVIDER=gemini               # final answer to the student
# ANSWER_MODELS=gemini-2.5-flash-lite,gemini-2.5-flash   # cascade
#
# ALTERNATIVE EXAMPLES
# ────────────────────
# Use DeepSeek for everything:
#   ANSWER_PROVIDER=openai
#   OPENAI_API_KEY=sk-...
#   OPENAI_BASE_URL=https://api.deepseek.com/v1
#   ANSWER_MODELS=deepseek-chat
#
# Use Kimi (Moonshot) for answers:
#   ANSWER_PROVIDER=openai
#   OPENAI_API_KEY=sk-...
#   OPENAI_BASE_URL=https://api.moonshot.cn/v1
#   ANSWER_MODELS=moonshot-v1-8k
#
# Use Qwen via DashScope (cloud Qwen, not local):
#   ANSWER_PROVIDER=openai
#   OPENAI_API_KEY=sk-...
#   OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
#   ANSWER_MODELS=qwen-turbo
#
# MEMORY IMPACT (M2 / 8 GB)
# ──────────────────────────
# Ollama runs models as a SEPARATE process outside Python.
# Python only makes HTTP calls to http://localhost:11434 — zero extra RAM
# inside the FastAPI process.  Ollama itself uses:
#   qwen3:1.7b  → ~1.1 GB RAM
#   qwen3:8b    → ~5.0 GB RAM  (too large alongside bge-m3 + reranker on 8 GB)
#   qwen3:1.7b is the recommended choice for M2 / 8 GB.
#
# WHY QWEN3:1.7B FOR ENRICHMENT?
# ───────────────────────────────
# Contextual enrichment only needs to write one short sentence per chunk.
# A 1.7B model handles this perfectly at ~15 tokens/s on M2.
# It never touches the Gemini API → zero quota usage during ingestion.

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """
    Abstract base for all LLM providers.
    Subclasses implement `_raw_call(prompt) -> str` and optionally override
    `call(prompt) -> str` if they need custom retry logic.
    """

    # Default retry settings — subclasses or env vars can override
    MAX_RETRIES: int   = int(os.getenv("LLM_MAX_RETRIES", "2"))
    BACKOFF_503: float = float(os.getenv("LLM_503_BACKOFF", "5"))
    MIN_DELAY:   float = float(os.getenv("LLM_MIN_429_DELAY", "5"))

    @abstractmethod
    def _raw_call(self, prompt: str) -> str:
        """Send prompt to the backend, return raw response text. No retry logic here."""
        ...

    # Substrings that indicate a NON-transient error — retrying won't help,
    # so we fail immediately instead of burning MAX_RETRIES * BACKOFF_503
    # seconds (e.g. 3 × 5s = 15s) on something that needs a manual fix.
    _NON_TRANSIENT_MARKERS = ("isn't pulled yet", "Is Ollama running")

    def call(self, prompt: str) -> str:
        """
        Call the LLM with basic retry (up to MAX_RETRIES on transient errors).
        Raises RuntimeError if all retries fail, or immediately on errors
        flagged as non-transient (e.g. model not pulled, server not running).
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 2):
            try:
                result = self._raw_call(prompt)
                if result:
                    return result
                log.warning("%s: empty response on attempt %d", self.__class__.__name__, attempt)
            except Exception as exc:
                last_exc = exc
                log.warning("%s error (attempt %d): %s", self.__class__.__name__, attempt, exc)

                # Fail fast — don't waste time retrying a missing model or
                # a server that isn't running. The message itself signals this.
                if any(marker in str(exc) for marker in self._NON_TRANSIENT_MARKERS):
                    log.warning("%s: non-transient error, skipping remaining retries.",
                                self.__class__.__name__)
                    break

                if attempt <= self.MAX_RETRIES:
                    time.sleep(self.BACKOFF_503)

        raise RuntimeError(
            f"{self.__class__.__name__} failed after {self.MAX_RETRIES + 1} attempts: {last_exc}"
        )

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ─────────────────────────────────────────────────────────────────────────────
# 1. Gemini provider
# ─────────────────────────────────────────────────────────────────────────────

class GeminiProvider(LLMProvider):
    """
    Google Gemini via the google-genai SDK.
    Supports a cascade of models: tries them in order, moving to the next on
    429 (after LLM_429_PER_MODEL_MAX consecutive hits) or 503.

    Rate-limit behaviour:
    - Parses `retryDelay` from the 429 response and sleeps that long.
    - Enforces a minimum sleep of MIN_DELAY (default 5s) even when Gemini
      returns retryDelay=0, which would otherwise cause an instant re-fire.
    - After LLM_429_PER_MODEL_MAX hits on one model, cascades to the next.
    """

    # How many consecutive 429s before we give up on a model and try the next
    PER_MODEL_MAX_429 = int(os.getenv("LLM_429_PER_MODEL_MAX", "3"))

    def __init__(self, api_key: str, models: list[str]):
        import google.generativeai  # validate import at construction time
        from google import genai as _genai
        self._client = _genai.Client(api_key=api_key)
        self.models  = models   # cascade order, e.g. ["gemini-2.5-flash-lite", "gemini-2.5-flash"]

    @staticmethod
    def _parse_retry_delay(exc: Exception) -> float:
        """Extract the server-suggested retry delay (seconds) from a 429 exception."""
        import re
        try:
            m = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", str(exc))
            if m:
                return float(m.group(1))
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _http_code(exc: Exception) -> Optional[int]:
        """Extract HTTP status code from a google-genai exception string."""
        import re
        try:
            m = re.match(r"^(\d{3})\s", str(exc))
            if m:
                return int(m.group(1))
            code = getattr(exc, "code", None)
            if isinstance(code, int):
                return code
        except Exception:
            pass
        return None

    def _raw_call(self, prompt: str) -> str:
        # Not used directly — GeminiProvider overrides call() for cascade logic
        raise NotImplementedError("Use call() directly.")

    def call(self, prompt: str, unlimited_retries: bool = False) -> str:
        """
        Full cascade + retry logic for Gemini.
        - Tries each model in self.models order.
        - On 429: sleep max(retryDelay, MIN_DELAY), retry up to PER_MODEL_MAX_429 times.
        - On 503: sleep BACKOFF_503, retry up to MAX_RETRIES times.
        - In unlimited_retries mode (used during ingestion): after all models fail,
          wait MIN_DELAY*2 and start the cascade again from the first model.
        """
        last_exc: Optional[Exception] = None

        def _one_round() -> Optional[str]:
            nonlocal last_exc
            for model_name in self.models:
                consecutive_429 = 0

                for attempt in range(1, self.MAX_RETRIES + 2):
                    try:
                        resp = self._client.models.generate_content(
                            model=model_name, contents=prompt
                        )
                        if hasattr(resp, "text") and resp.text:
                            if attempt > 1:
                                log.info("Gemini %s succeeded on attempt %d", model_name, attempt)
                            return resp.text.strip()
                        log.warning("Gemini %s: empty response (attempt %d)", model_name, attempt)

                    except Exception as exc:
                        last_exc  = exc
                        http_code = self._http_code(exc)

                        if http_code == 429:
                            consecutive_429 += 1
                            raw_delay = self._parse_retry_delay(exc)
                            # CRITICAL: enforce minimum to avoid retryDelay=0 spin
                            delay = max(raw_delay, self.MIN_DELAY)
                            if delay != raw_delay:
                                log.debug("retryDelay=%.1fs below min; using %.1fs", raw_delay, delay)

                            if consecutive_429 >= self.PER_MODEL_MAX_429:
                                log.warning(
                                    "Gemini %s: %d consecutive 429s → cascade. Sleeping %.1fs first.",
                                    model_name, consecutive_429, delay,
                                )
                                time.sleep(delay)
                                break  # → next model

                            log.warning(
                                "Gemini %s: 429 (attempt %d/%d). Waiting %.1fs…",
                                model_name, attempt, self.MAX_RETRIES + 1, delay,
                            )
                            time.sleep(delay)
                            continue  # retry same model

                        elif http_code == 503:
                            if attempt <= self.MAX_RETRIES:
                                log.warning(
                                    "Gemini %s: 503 (attempt %d/%d). Waiting %.1fs…",
                                    model_name, attempt, self.MAX_RETRIES + 1, self.BACKOFF_503,
                                )
                                time.sleep(self.BACKOFF_503)
                                continue
                            log.warning("Gemini %s: 503 exhausted → cascade.", model_name)
                            break

                        else:
                            log.warning("Gemini %s: error %s: %s", model_name, http_code, exc)
                            break  # non-transient → next model

            return None  # all models failed this round

        if unlimited_retries:
            # Keep cycling until success — used during ingestion.
            round_num = 0
            while True:
                round_num += 1
                result = _one_round()
                if result is not None:
                    return result
                pause = self.MIN_DELAY * 2
                log.warning(
                    "Gemini: all models rate-limited (round %d). Waiting %.1fs…",
                    round_num, pause,
                )
                time.sleep(pause)
        else:
            result = _one_round()
            if result is not None:
                return result
            raise RuntimeError(f"Gemini: all models failed. Last error: {last_exc}")

    @property
    def name(self) -> str:
        return f"Gemini({self.models[0]})"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ollama provider (local — Qwen, Llama, DeepSeek, Mistral, etc.)
# ─────────────────────────────────────────────────────────────────────────────

class OllamaProvider(LLMProvider):
    """
    Local LLM via Ollama's REST API (http://localhost:11434).

    Ollama is a free, open-source runtime that runs quantised models locally.
    It auto-downloads models on first use.

    Setup (one-time):
        brew install ollama       # macOS
        ollama serve              # start the server (keep running in background)
        ollama pull qwen3:1.7b    # download the model (~1.1 GB)

    Why Qwen3:1.7b for enrichment on M2 / 8 GB:
    - Fits in ~1.1 GB RAM — leaves plenty for bge-m3 + reranker
    - Fast enough: ~15 tokens/s on M2 CPU, ~40 tok/s on MPS (GPU cores)
    - Instruction-following is solid for the short "contextualise this chunk" task
    - Completely offline — no API key, no rate limits, no quota

    Ollama uses llama.cpp under the hood and auto-selects MPS on Apple Silicon.
    """

    def __init__(
        self,
        model:    str   = "qwen3:1.7b",
        base_url: str   = "http://localhost:11434",
        timeout:  int   = 120,       # seconds — local inference can be slow on first call
        options:  dict  = None,      # Ollama model options (temperature, num_ctx, etc.)
    ):
        self.model    = model
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        # Default options tuned for contextual enrichment (short, factual output)
        self.options  = options or {
            "temperature":  0.1,   # low temperature → more deterministic, less hallucination
            "num_ctx":      4096,  # context window; 4096 is enough for batch enrichment
            "num_predict":  200,   # max output tokens per call
        }

    def _raw_call(self, prompt: str) -> str:
        """
        POST to Ollama's /api/generate endpoint (non-streaming).
        Returns the generated text or raises on error.
        """
        import requests  # lightweight HTTP — already in deps via FastAPI

        url  = f"{self.base_url}/api/generate"
        body = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  False,       # wait for full response, simpler to parse
            "options": self.options,
        }

        try:
            resp = requests.post(url, json=body, timeout=self.timeout)

            # ── Special case: 404 means the MODEL isn't pulled, not that the
            # server is down. Give an actionable error instead of a generic
            # "404 Client Error" that retrying 3 times won't fix.
            if resp.status_code == 404:
                raise RuntimeError(
                    f"Ollama returned 404 for model '{self.model}'. "
                    f"The Ollama server is running, but this model isn't pulled yet. "
                    f"Fix: run `ollama pull {self.model}` in a terminal, then retry. "
                    f"Check what's installed with `ollama list`."
                )

            resp.raise_for_status()
            data = resp.json()
            # Ollama returns {"response": "...", "done": true, ...}
            text = data.get("response", "").strip()
            if not text:
                raise RuntimeError(f"Ollama returned empty response: {data}")
            return text
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Run: ollama serve"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama timed out after {self.timeout}s for model '{self.model}'. "
                "Try a smaller model or increase OLLAMA_TIMEOUT."
            )

    def health_check(self) -> bool:
        """Return True if Ollama is reachable and the model is available."""
        import requests
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if not resp.ok:
                return False
            tags   = resp.json().get("models", [])
            names  = [t.get("name", "") for t in tags]
            # Match e.g. "qwen3:1.7b" or "qwen3:1.7b-instruct-q4_K_M"
            return any(self.model.split(":")[0] in n for n in names)
        except Exception:
            return False

    @property
    def name(self) -> str:
        return f"Ollama({self.model})"


# ─────────────────────────────────────────────────────────────────────────────
# 3. OpenAI-compatible provider (DeepSeek, Kimi/Moonshot, cloud Qwen, etc.)
# ─────────────────────────────────────────────────────────────────────────────

class OpenAICompatibleProvider(LLMProvider):
    """
    Any API that follows the OpenAI /v1/chat/completions schema.

    Works with:
    - DeepSeek      : OPENAI_BASE_URL=https://api.deepseek.com/v1
                      MODEL=deepseek-chat
    - Kimi/Moonshot : OPENAI_BASE_URL=https://api.moonshot.cn/v1
                      MODEL=moonshot-v1-8k
    - Alibaba Qwen  : OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
                      MODEL=qwen-turbo
    - Local vLLM    : OPENAI_BASE_URL=http://localhost:8000/v1
                      MODEL=Qwen/Qwen3-8B-Instruct
    - OpenAI        : OPENAI_BASE_URL=https://api.openai.com/v1  (default)
                      MODEL=gpt-4o-mini

    The openai Python package must be installed: pip install openai
    """

    def __init__(
        self,
        api_key:  str,
        model:    str,
        base_url: str  = "https://api.openai.com/v1",
        temperature: float = 0.3,
        max_tokens:  int   = 1024,
    ):
        # Import lazily so openai is optional (not needed if using Gemini + Ollama)
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            )
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens

    def _raw_call(self, prompt: str) -> str:
        """Call the /v1/chat/completions endpoint."""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content.strip()

    @property
    def name(self) -> str:
        return f"OpenAICompat({self.model})"


# ─────────────────────────────────────────────────────────────────────────────
# Provider registry — single source of truth, read from .env
# ─────────────────────────────────────────────────────────────────────────────
#
# ROLES
# ─────
# "enrichment" — contextual chunk enrichment during PDF ingestion
#                Default: Ollama + qwen3:1.7b  (local, zero quota)
#
# "planning"   — intent classification, search query building, query rewriting
#                Default: Ollama + qwen3:1.7b  (fast, no cloud cost)
#
# "answer"     — final answer generation shown to the student
#                Default: Gemini cascade        (best quality)
#
# You can set any role to any provider via .env.

_registry: dict[str, LLMProvider] = {}   # populated by build_registry()


def build_registry() -> dict[str, LLMProvider]:
    """
    Read .env and construct one provider instance per role.
    Call this ONCE at startup (from main.py).

    Returns the registry dict so callers can inspect what was built.
    Stores it in _registry for get_provider() lookups.
    """
    global _registry

    # ── Enrichment provider ───────────────────────────────────────────────────
    enrich_provider  = os.getenv("ENRICHMENT_PROVIDER", "ollama").lower()
    enrich_model     = os.getenv("ENRICHMENT_MODEL",    "qwen3:1.7b")

    # ── Planning provider ─────────────────────────────────────────────────────
    planning_provider = os.getenv("PLANNING_PROVIDER", "ollama").lower()
    planning_model    = os.getenv("PLANNING_MODEL",    "qwen3:1.7b")

    # ── Answer provider ───────────────────────────────────────────────────────
    answer_provider  = os.getenv("ANSWER_PROVIDER",  "gemini").lower()
    answer_models    = os.getenv(
        "ANSWER_MODELS",
        "gemini-2.5-flash-lite,gemini-2.5-flash",
    ).split(",")

    def _make(provider_name: str, model: str, models: list[str] = None) -> LLMProvider:
        """Instantiate the right LLMProvider subclass."""
        if provider_name == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is required for provider=gemini")
            return GeminiProvider(api_key=api_key, models=models or [model])

        elif provider_name == "ollama":
            return OllamaProvider(
                model    = model,
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                timeout  = int(os.getenv("OLLAMA_TIMEOUT", "120")),
            )

        elif provider_name in ("openai", "deepseek", "kimi", "moonshot", "qwen_cloud"):
            # All OpenAI-compatible APIs use the same class
            base_urls = {
                "openai":     "https://api.openai.com/v1",
                "deepseek":   "https://api.deepseek.com/v1",
                "kimi":       "https://api.moonshot.cn/v1",
                "moonshot":   "https://api.moonshot.cn/v1",
                "qwen_cloud": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            }
            default_url = base_urls.get(provider_name, "https://api.openai.com/v1")
            return OpenAICompatibleProvider(
                api_key  = os.getenv("OPENAI_API_KEY", ""),
                model    = model,
                base_url = os.getenv("OPENAI_BASE_URL", default_url),
            )

        else:
            raise ValueError(
                f"Unknown provider '{provider_name}'. "
                "Valid options: gemini, ollama, openai, deepseek, kimi, moonshot, qwen_cloud"
            )

    _registry = {
        "enrichment": _make(enrich_provider,  enrich_model),
        "planning":   _make(planning_provider, planning_model),
        "answer":     _make(answer_provider,   answer_models[0], models=answer_models),
    }

    # Log what was built
    for role, prov in _registry.items():
        log.info("LLM registry: %-12s → %s", role, prov.name)

    return _registry


def get_provider(role: str) -> LLMProvider:
    """
    Retrieve a provider by role.  Raises if build_registry() was not called yet.

    Args:
        role: "enrichment" | "planning" | "answer"
    """
    if not _registry:
        raise RuntimeError("call build_registry() at startup before get_provider()")
    if role not in _registry:
        raise KeyError(f"Unknown role '{role}'. Valid: {list(_registry.keys())}")
    return _registry[role]


def call_llm(prompt: str, role: str = "answer", **kwargs) -> str:
    """
    Top-level convenience function: route a prompt to the right provider.

    Args:
        prompt : the prompt text
        role   : "enrichment" | "planning" | "answer"
        **kwargs: passed through to provider.call() — e.g. unlimited_retries=True
                  for GeminiProvider during ingestion.

    Usage:
        from llm_providers import call_llm

        # In context_builder / ingestion:
        ctx = call_llm(batch_prompt, role="enrichment")

        # In planner:
        intent = call_llm(classify_prompt, role="planning")

        # In query endpoint:
        answer = call_llm(final_prompt, role="answer")
    """
    provider = get_provider(role)
    return provider.call(prompt, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Health-check utility (called at startup)
# ─────────────────────────────────────────────────────────────────────────────

def check_providers() -> dict[str, str]:
    """
    Quick connectivity check for all registered providers.
    Returns a dict of {role: "ok" | "warning: <msg>"}.
    Logs warnings for any provider that can't be reached — does NOT raise,
    so a missing Ollama server doesn't prevent startup.

    Failures are printed as a multi-line banner (not a single log line) so
    they're impossible to scroll past in a busy startup log.
    """
    status   = {}
    failures = []

    for role, provider in _registry.items():
        if isinstance(provider, OllamaProvider):
            if provider.health_check():
                status[role] = "ok"
                log.info("Provider health [%s] %s → ok", role, provider.name)
            else:
                msg = (
                    f"Ollama unreachable OR model '{provider.model}' not pulled "
                    f"(role='{role}')"
                )
                status[role] = f"warning: {msg}"
                failures.append((role, provider.model, provider.base_url))
        else:
            # Gemini and OpenAI-compat are network calls — don't ping at startup
            status[role] = "assumed ok (not health-checked)"
            log.info("Provider health [%s] %s → skipped (cloud API)", role, provider.name)

    if failures:
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════════╗",
            "║  ⚠️  OLLAMA PROVIDER NOT READY — enrichment/planning will degrade  ║",
            "╠══════════════════════════════════════════════════════════════════╣",
        ]
        for role, model, base_url in failures:
            lines.append(f"║  role={role:<11} model={model:<20} url={base_url:<22}║")
        lines += [
            "╠══════════════════════════════════════════════════════════════════╣",
            "║  Fix:                                                               ║",
            "║    1. ollama serve            (start the server, separate terminal)║",
            "║    2. ollama pull <model>      (download the model — see above)    ║",
            "║    3. ollama list               (verify it shows up)               ║",
            "║  Until fixed, affected roles will fail fast and skip that step     ║",
            "║  (e.g. ingestion proceeds WITHOUT contextual enrichment).          ║",
            "╚══════════════════════════════════════════════════════════════════╝",
            "",
        ]
        log.warning("\n".join(lines))

    return status