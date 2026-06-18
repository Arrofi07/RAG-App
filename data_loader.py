# data_loader.py

import os
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
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
EMBED_MODEL_NAME = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Output dimensionality of BAAI/bge-small-en-v1.5's dense embeddings.
# IMPORTANT: this must match the `dim` your Qdrant collection was created
# with. If you're switching from the old Gemini embeddings (3072-dim),
# you must recreate the Qdrant collection and re-ingest every document --
# embeddings from different models are never compatible, even when sizes
# happen to match.

_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    """Lazily load the embedding model so importing this module stays cheap."""
    global _embed_model

    if _embed_model is None:
        _embed_model = SentenceTransformer(
            EMBED_MODEL_NAME,
            device="cpu",
            )

    return _embed_model

def get_embed_dim() -> int:
    """Return the dimensionality of the embedding model's output vectors."""
    return _get_embed_model().get_sentence_embedding_dimension()

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

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts locally using BAAI/bge-small-en-v1.5.

    No API calls and no rate limits -- sentence-transformers handles
    batching internally, so this scales to large documents without the
    429 issues the Gemini API version had. bge-small-en-v1.5 doesn't need a special
    instruction prefix for queries (unlike some other BGE models), so the
    same call works for both document chunks and user questions.
    """

    if not texts:
        return []

    model = _get_embed_model()

    embeddings = model.encode(
        texts,
        batch_size=8,
        normalize_embeddings=True,  # so cosine similarity == dot product
        show_progress_bar=False,
    )

    return embeddings.tolist()
