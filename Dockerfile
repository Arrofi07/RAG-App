# Dockerfile — Study-in-Germany AI Advisor (FastAPI backend)
#
# BUILD STAGES
# ────────────
# Stage 1 (builder): install Python deps with uv into /app/.venv
# Stage 2 (runtime): copy only the venv + source; no build tools in final image
#
# This keeps the final image small (~1.2 GB vs ~2.5 GB single-stage) and
# means pip/uv are not present in the container that actually serves traffic.
#
# MODELS ARE NOT BAKED INTO THE IMAGE
# ─────────────────────────────────────
# bge-m3 (~2.3 GB) and bge-reranker-v2-m3 (~2.3 GB) are downloaded from
# HuggingFace on first startup and cached in a Docker volume.
# Baking them into the image would make it ~5 GB and rebuild slowly.
# See docker-compose.yml for the volume mount.
#
# USAGE
# ─────
# docker build -t rag-advisor .
# docker run -p 8000:8000 --env-file .env rag-advisor

# ── Stage 1: dependency installer ────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv (fast pip replacement)
RUN pip install --no-cache-dir uv

# Copy dependency spec first (layer-cache: only reinstall when deps change)
COPY pyproject.toml .
COPY uv.lock        .

# Install all runtime deps into a virtual environment
RUN uv sync --no-dev --frozen

# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Non-root user for security (never run a web server as root)
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy the pre-built venv from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY advisor_config.py     .
COPY auth.py               .
COPY context_builder.py    .
COPY custom_types.py       .
COPY data_loader.py        .
COPY eval_engine.py        .
COPY llm_providers.py      .
COPY main.py               .
COPY planner.py            .
COPY reranker.py           .
COPY storage.py            .
COPY university_recommender.py .
COPY vector_db.py          .
COPY web_search_tool.py    .

# Create directories that the app writes to at runtime
RUN mkdir -p /app/data /app/uploads && chown -R appuser:appuser /app

USER appuser

# Put the venv on PATH so `python` and `uvicorn` resolve correctly
ENV PATH="/app/.venv/bin:$PATH"

# Tell HuggingFace to store model cache in the Docker volume (see compose)
ENV HF_HOME="/app/.cache/huggingface"

# Expose the FastAPI port
EXPOSE 8000

# Healthcheck: ping the root endpoint every 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" \
    || exit 1

# Start uvicorn with:
#   --host 0.0.0.0   → listen on all interfaces (required inside Docker)
#   --workers 1      → one process (the embedding model is NOT fork-safe)
#   --timeout-keep-alive 75  → slightly above typical load balancer timeout
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-keep-alive", "75"]