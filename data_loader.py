# data_loader.py

import os
from dotenv import load_dotenv

from FlagEmbedding import BGEM3FlagModel
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

# -------------------------
# Local embedding model
# -------------------------

# BAAI/bge-m3 supports three retrieval modes from a single model:
#   - dense:   classic semantic embedding (1024-dim float vector)
#   - sparse:  lexical weights, like a neural BM25 (token_id → weight dict)
#   - colbert: multi-vector (skipped — too memory-heavy for most setups)
#
# We use FlagEmbedding instead of sentence-transformers because
# sentence-transformers only exposes the dense output; FlagEmbedding exposes
# all three.  Same model weights, different Python wrapper.
EMBED_MODEL_NAME = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-m3")

# Dense output dimensionality — must match the Qdrant collection.
# Changing this (or switching models) requires deleting the collection
# and re-ingesting all documents.
EMBED_DIM = 1024

_embed_model: BGEM3FlagModel | None = None


def _get_embed_model() -> BGEM3FlagModel:
    """Lazily load the model so importing this module stays cheap."""
    global _embed_model

    if _embed_model is None:
        _embed_model = BGEM3FlagModel(
            EMBED_MODEL_NAME,
            device="cpu",
        )

    return _embed_model


# -------------------------
# Text splitter
# -------------------------

splitter = SentenceSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


# -------------------------
# Load PDF and split
# -------------------------

def load_and_chunk_pdf(path: str) -> list[str]:
    docs = PDFReader().load_data(file=path)

    texts = [
        doc.text
        for doc in docs
        if getattr(doc, "text", None)
    ]

    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))

    return chunks


# -------------------------
# Embed
# -------------------------

def _lexical_weights_to_qdrant_sparse(lexical_weights: dict) -> dict:
    """
    Convert bge-m3's lexical_weights dict → Qdrant SparseVector format.

    bge-m3 returns: {str(token_id): float}
    Qdrant expects: {"indices": [int, ...], "values": [float, ...]}
    """
    indices, values = [], []

    for token_id, weight in lexical_weights.items():
        w = float(weight)
        if w > 0:
            indices.append(int(token_id))
            values.append(w)

    return {"indices": indices, "values": values}


def embed(texts: list[str]) -> dict:
    """
    Embed a list of texts with bge-m3, returning both dense and sparse vectors.

    Returns:
        {
            "dense":  list[list[float]],                             # (N, 1024)
            "sparse": list[{"indices": [...], "values": [...]}],     # N sparse vecs
        }

    One call handles both ingest (all chunks) and query (single question).
    """
    if not texts:
        return {"dense": [], "sparse": []}

    model = _get_embed_model()

    output = model.encode(
        texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    dense  = output["dense_vecs"].tolist()
    sparse = [
        _lexical_weights_to_qdrant_sparse(lw)
        for lw in output["lexical_weights"]
    ]

    return {"dense": dense, "sparse": sparse}


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convenience wrapper — dense vectors only (backward compat)."""
    return embed(texts)["dense"]