# reranker.py

import os
import logging
from sentence_transformers import CrossEncoder

log = logging.getLogger(__name__)

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# Use MPS on Apple Silicon, CUDA on NVIDIA, CPU otherwise.
# Kept self-contained so reranker.py has no dependency on data_loader.py.
def _detect_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

DEVICE = os.getenv("RERANKER_DEVICE", _detect_device())

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        log.info("Loading reranker '%s' on device='%s'…", RERANKER_MODEL, DEVICE)
        _model = CrossEncoder(RERANKER_MODEL, max_length=512, device=DEVICE)
        log.info("Reranker ready.")
    return _model


def warmup() -> None:
    """
    Pre-load the reranker into memory at server startup so the first real
    request doesn't pay the cold-start penalty (~7-10 s on CPU).
    """
    _get_model()


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Re-score and re-order retrieved candidates using a cross-encoder.

    A cross-encoder reads the query and each candidate chunk together —
    much more precise than comparing separate embeddings, but too slow to
    run over the full collection, which is why it only sees the top-N
    candidates that dense/sparse retrieval already narrowed down.
    """
    if not candidates:
        return []

    model  = _get_model()
    pairs  = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]