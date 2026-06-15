# data_loader.py

import os
from dotenv import load_dotenv

from google import genai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

# -------------------------
# Gemini Client
# -------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=GEMINI_API_KEY)

# Gemini embedding model
EMBED_MODEL = "gemini-embedding-001"

# Dimension used by your Qdrant collection
EMBED_DIM = 3072

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
    embeddings = []

    for text in texts:
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
        )

        # SDK versions expose either `embeddings` or `embedding`
        if hasattr(response, "embeddings"):
            embeddings.append(response.embeddings[0].values)
        elif hasattr(response, "embedding"):
            embeddings.append(response.embedding.values)
        else:
            raise RuntimeError(
                f"Unexpected embedding response: {response}"
            )

    return embeddings