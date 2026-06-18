# reranker.py

from sentence_transformers import CrossEncoder

# Multilingual cross-encoder reranker (works well on English + German content,
# which matters since your test PDF is German). ~2.3GB, downloads on first run.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    """Lazily load the cross-encoder so importing this module is cheap and
    the (large) model only gets pulled into memory the first time it's used."""
    global _model

    if _model is None:
        _model = CrossEncoder(RERANKER_MODEL, max_length=512)

    return _model


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Re-score and re-order retrieved candidates using a cross-encoder.

    A cross-encoder reads the query and a candidate chunk together (instead
    of comparing two separately-computed embeddings), which makes it much
    more precise at judging relevance than the dense vector search alone —
    at the cost of being too slow to run over the whole collection, which is
    why it only runs on the top N candidates dense search already narrowed
    down.

    Args:
        query: the user's question.
        candidates: list of dicts with at least a "text" key, as returned
            by QdrantStorage.search_candidates().
        top_k: how many candidates to keep after reranking.

    Returns:
        The top_k candidates, sorted best-to-worst, each with a
        "rerank_score" key added (float, higher = more relevant).
    """

    if not candidates:
        return []

    model = _get_model()

    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    ranked = sorted(
        candidates,
        key=lambda c: c["rerank_score"],
        reverse=True,
    )

    return ranked[:top_k]
