# data_loader.py

import os
import logging
from dotenv import load_dotenv

from FlagEmbedding import BGEM3FlagModel
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

log = logging.getLogger(__name__)

# -------------------------
# Local embedding model
# -------------------------

# BAAI/bge-m3 supports three retrieval modes from a single model:
#   - dense:   classic semantic embedding (1024-dim float vector)
#   - sparse:  lexical weights, like a neural BM25 (token_id → weight dict)
#   - colbert: multi-vector (skipped — too memory-heavy for most setups)
EMBED_MODEL_NAME = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM        = 1024

_embed_model: BGEM3FlagModel | None = None


def _get_embed_model() -> BGEM3FlagModel:
    global _embed_model
    if _embed_model is None:
        _embed_model = BGEM3FlagModel(EMBED_MODEL_NAME, device="cpu")
    return _embed_model


# -------------------------
# Text splitters
# -------------------------

# Small chunks → precise embedding targets for retrieval
child_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)


# -------------------------
# PDF loading
# -------------------------

def _load_pages(path: str) -> list:
    return PDFReader().load_data(file=path)


def _pages_to_full_text(pages: list) -> str:
    return "\n\n".join(doc.text for doc in pages if getattr(doc, "text", None))


def load_chunks(path: str) -> dict:
    """
    Load a PDF and produce chunks with their immediate neighbors stored inline.

    Returns
    -------
    {
        "chunks": [
            {
                "text":        str,         # the chunk's own text
                "prev_chunk":  str,         # preceding chunk (empty string if first)
                "next_chunk":  str,         # following chunk (empty string if last)
                "chunk_index": int,
                "page":        str | None,
            },
            ...
        ],
        "full_text": str,  # entire document, used for contextual enrichment
    }

    Storing neighbors in the payload (instead of fetching them at query time)
    means window expansion is a pure in-memory operation with zero extra Qdrant
    round-trips.
    """
    pages     = _load_pages(path)
    full_text = _pages_to_full_text(pages)

    raw: list[tuple[str, str | None]] = []
    for doc in pages:
        if not getattr(doc, "text", None):
            continue
        page_label = (doc.metadata or {}).get("page_label")
        for chunk_text in child_splitter.split_text(doc.text):
            raw.append((chunk_text, page_label))

    chunks = []
    for i, (text, page) in enumerate(raw):
        chunks.append({
            "text":        text,
            "prev_chunk":  raw[i - 1][0] if i > 0            else "",
            "next_chunk":  raw[i + 1][0] if i < len(raw) - 1 else "",
            "chunk_index": i,
            "page":        page,
        })

    log.info("Loaded %d chunks from '%s' (%d pages)", len(chunks), path, len(pages))
    return {"chunks": chunks, "full_text": full_text}


# -------------------------
# Contextual enrichment
# -------------------------

# Characters of the document shown to the LLM when generating chunk context.
# Using the full document for every chunk would be slow and expensive for long
# documents. Taking the first ~3000 and last ~1000 chars gives the LLM a
# strong sense of the document's topic, structure, and conclusion without
# sending the entire thing in every prompt.
_CONTEXT_DOC_PREFIX_CHARS = 3000
_CONTEXT_DOC_SUFFIX_CHARS = 1000


def _build_doc_summary(full_text: str) -> str:
    """Truncate document text to a representative excerpt for the LLM prompt."""
    if len(full_text) <= _CONTEXT_DOC_PREFIX_CHARS + _CONTEXT_DOC_SUFFIX_CHARS:
        return full_text

    prefix = full_text[:_CONTEXT_DOC_PREFIX_CHARS]
    suffix = full_text[-_CONTEXT_DOC_SUFFIX_CHARS:]
    return f"{prefix}\n\n[... middle of document omitted ...]\n\n{suffix}"


def generate_chunk_contexts(
    full_text: str,
    chunks:    list[str],
    call_llm,              # callable(str) -> str  ← injected from main.py
) -> list[str]:
    """
    For each chunk, ask the LLM to write a short context that situates it
    within the whole document.  This prefix is prepended to the chunk text
    before embedding so the resulting vector captures the chunk's role, not
    just its local content.

    This is the technique from Anthropic's "Contextual Retrieval" (2024):
      https://www.anthropic.com/news/contextual-retrieval

    The call_llm argument is injected so this module stays free of Google
    genai imports (no circular dependency with main.py).

    Returns a list of context prefix strings, one per chunk.
    If the LLM call for a chunk fails, an empty string is used as fallback
    so ingestion continues rather than crashing.
    """
    doc_summary = _build_doc_summary(full_text)
    prefixes    = []

    for i, chunk in enumerate(chunks):
        prompt = f"""<document>
{doc_summary}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context (1-2 sentences) to situate this chunk \
within the overall document for the purpose of improving search retrieval. \
Answer only with the succinct context and nothing else."""

        try:
            prefix = call_llm(prompt).strip()
        except Exception as exc:
            log.warning(
                "Context generation failed for chunk %d/%d: %s — using empty prefix.",
                i + 1, len(chunks), exc,
            )
            prefix = ""

        prefixes.append(prefix)

        if (i + 1) % 10 == 0:
            log.info(
                "Contextual enrichment: %d / %d chunks done", i + 1, len(chunks)
            )

    return prefixes


def build_embed_texts(chunks: list[str], prefixes: list[str]) -> list[str]:
    """
    Combine each chunk with its context prefix to form the text that gets
    embedded.  Empty prefixes (enrichment disabled or LLM failure) are
    handled gracefully — we just embed the raw chunk.

    The resulting embed text is NOT stored in Qdrant; only the original
    chunk text and the prefix are stored separately so the LLM receives
    clean, non-duplicated text at generation time.
    """
    embed_texts = []
    for chunk, prefix in zip(chunks, prefixes):
        if prefix:
            embed_texts.append(f"{prefix}\n\n{chunk}")
        else:
            embed_texts.append(chunk)
    return embed_texts


# -------------------------
# Embed
# -------------------------

def _lexical_weights_to_qdrant_sparse(lexical_weights: dict) -> dict:
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
    """Dense vectors only — backward compat."""
    return embed(texts)["dense"]