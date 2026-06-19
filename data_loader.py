# data_loader.py

import os
from dotenv import load_dotenv

#from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

# -------------------------
# Local embedding model
# -------------------------

# Multilingual (handles the German PDF) and pairs naturally with the
# BAAI/bge-reranker-v2-m3 reranker -- both are BAAI/BGE models designed to
# work together. Runs fully locally: no API key, no rate limits, no
# per-token cost. ~2.3GB download on first use.
EMBED_MODEL_NAME = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-m3")

# bge-small-en-v1.5 supports three retrieval modes from a single model:
#   - dense: classic semantic embedding (1024-dim float vector)
#   - sparse: lexical weights, like a neural BM25 (token_id → weight dict)
#   - colbert: multi-vector (we skip this for now — too memory-heavy)
# IMPORTANT: this must match the `dim` your Qdrant collection was created
# with. If you're switching from the old Gemini embeddings (3072-dim),
# you must recreate the Qdrant collection and re-ingest every document --
# embeddings from different models are never compatible, even when sizes
# happen to match.

_embed_model: BGEM3FlagModel | None = None


def _get_embed_model() -> BGEM3FlagModel:
    """Lazily load the embedding model so importing this module stays cheap."""
    global _embed_model

    if _embed_model is None:
        _embed_model = BGEM3FlagModel(
            EMBED_MODEL_NAME,
            device="cpu",
            )

    return _embed_model

def get_embed_dim():
    model = _get_embed_model()

    if hasattr(model, "get_sentence_embedding_dimension"):
        return model.get_sentence_embedding_dimension()

    return 1024

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
        chunks.extend(
            splitter.split_text(text)
        )

    return chunks


# -------------------------
# Generate embeddings
# -------------------------

def _lexical_weights_to_qdrant_sparse(
    lexical_weights: dict,
) -> dict:
    """
    Convert bge-small-en-v1.5's lexical_weights dict → Qdrant SparseVector format.
 
    bge-small-en-v1.5 returns sparse weights as {str(token_id): float}.
    Qdrant expects {"indices": [int, ...], "values": [float, ...]}.
    """
    indices = []
    values  = []
 
    for token_id, weight in lexical_weights.items():
        w = float(weight)
        if w > 0:
            indices.append(int(token_id))
            values.append(w)
 
    return {"indices": indices, "values": values}
 
 
def embed(texts: list[str]) -> dict:
    """
    Embed a list of texts with bge-small-en-v1.5, returning both dense and sparse vectors.
 
    Returns:
        {
            "dense":  list[list[float]],         # shape (N, 1024)
            "sparse": list[{"indices": [...], "values": [...]}],  # N sparse vecs
        }
 
    This single call is all you need for both ingest and query — just pass
    the query as a length-1 list and take index [0] from each output.
    """
    if not texts:
        return {"dense": [], "sparse": []}
 
    model = _get_embed_model()
 
    output = model.encode(
        texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,  # skip multi-vector for now
    )
 
    dense  = output["dense_vecs"].tolist()
    sparse = [
        _lexical_weights_to_qdrant_sparse(lw)
        for lw in output["lexical_weights"]
    ]
 
    return {"dense": dense, "sparse": sparse}
 
 
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convenience wrapper — returns dense vectors only (backward compat)."""
    return embed(texts)["dense"]
