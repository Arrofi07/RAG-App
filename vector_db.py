# vector_db.py

import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    SparseIndexParams,
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    FusionQuery,
    Fusion,
    NamedVector,
    NamedSparseVector,
)

from qdrant_client.models import Filter, FieldCondition, MatchValue

from data_loader import get_embed_dim

load_dotenv()

EMBED_DIM = get_embed_dim()

class QdrantStorage:
    """
    Qdrant wrapper that stores both dense and sparse vectors per chunk,
    enabling hybrid (dense + BM25-style sparse) retrieval with built-in
    Reciprocal Rank Fusion.
 
    Collection layout
    -----------------
    named dense vector  → "dense"   (1024-dim, Cosine)
    named sparse vector → "sparse"  (SparseVectorParams)
 
    This is a breaking change from the flat-vector layout used in v1/v2.
    If an existing collection is detected with the wrong structure, a clear
    ValueError is raised so the user knows to recreate it.
    """
 
    DENSE_NAME  = "dense"
    SPARSE_NAME = "sparse"

    def __init__(
        self,
        url: str | None = None,
        collection: str = "docs",
        dim: int = EMBED_DIM,
    ):
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection = collection
        self.dim = dim
 
        self.client = QdrantClient(url=self.url, timeout=30)
 
        if self.client.collection_exists(collection_name=self.collection):
            self._validate_existing_collection()
        else:
            self._create_collection()

    def _validate_existing_collection(self) -> None:
        """
        Verify the existing collection has the expected named-vector structure.
        Raises ValueError with a clear migration message if it doesn't.
        """
        info = self.client.get_collection(collection_name=self.collection)
        vectors_config = info.config.params.vectors
 
        # Flat vector config (pre-hybrid) → wrong structure
        if not isinstance(vectors_config, dict):
            raise ValueError(
                f"Collection '{self.collection}' uses a flat (non-named) "
                f"vector layout from an older version of this app. "
                f"Hybrid search requires a named-vector collection. "
                f"Please delete the collection and re-ingest your documents:\n"
                f"  curl -X DELETE http://localhost:6333/collections/{self.collection}"
            )
 
        # Named vector dict but wrong dense dimension
        dense_cfg = vectors_config.get(self.DENSE_NAME)
        if dense_cfg is None or dense_cfg.size != self.dim:
            existing_dim = dense_cfg.size if dense_cfg else "unknown"
            raise ValueError(
                f"Collection '{self.collection}' has a '{self.DENSE_NAME}' "
                f"vector of size {existing_dim}, expected {self.dim}. "
                f"Delete the collection and re-ingest."
            )
 
    def _create_collection(self) -> None:
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                self.DENSE_NAME: VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                self.SPARSE_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
 
    def upsert(
        self,
        ids: list[str],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict],   # each: {"indices": [...], "values": [...]}
        payloads: list[dict],
    ) -> None:
        points = [
            PointStruct(
                id=ids[i],
                vector={
                    self.DENSE_NAME: dense_vectors[i],
                    self.SPARSE_NAME: SparseVector(
                        indices=sparse_vectors[i]["indices"],
                        values=sparse_vectors[i]["values"],
                    ),
                },
                payload=payloads[i],
            )
            for i in range(len(ids))
        ]
 
        self.client.upsert(         
            collection_name=self.collection,
            points=points,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
 
    def _build_filter(self, filename: str | None) -> Filter | None:
        if not filename:
            return None
        return Filter(
            must=[
                FieldCondition(
                    key="filename",
                    match=MatchValue(value=filename),
                )
            ]
        )
 
    def _points_to_candidates(
        self,
        points,
        score_key: str = "vector_score",
    ) -> list[dict]:
        candidates = []
        for r in points:
            payload = r.payload or {}
            text = payload.get("text", "")
            if not text:
                continue
            candidates.append({
                "id":       str(r.id),
                "text":     text,
                "source":   payload.get("source", ""),
                "page":     payload.get("page"),
                score_key:  r.score,
            })
        return candidates
 
    def search_hybrid_candidates(
        self,
        dense_vector: list[float],
        sparse_vector: dict,
        fetch_k: int = 30,
        filename: str | None = None,
    ) -> list[dict]:
        """
        Hybrid retrieval using Qdrant's built-in Reciprocal Rank Fusion.
 
        Internally, Qdrant runs two sub-searches in parallel:
          1. Dense (cosine similarity on the 1024-dim vector)
          2. Sparse (dot product on lexical BM25-style weights)
        …then merges both ranked lists with RRF into a single ranked list.
 
        RRF score for a document d = Σ  1 / (rank_i(d) + k)
        where rank_i(d) is d's rank in retrieval list i and k=60 by default.
        It's parameter-free and robust to score-scale differences between
        the two retrievers — no need to tune alpha/weights.
        """
 
        query_filter = self._build_filter(filename)
 
        results = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using=self.DENSE_NAME,
                    limit=fetch_k,
                    filter=query_filter,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_vector["indices"],
                        values=sparse_vector["values"],
                    ),
                    using=self.SPARSE_NAME,
                    limit=fetch_k,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=fetch_k,
            with_payload=True,
        )
 
        points = getattr(results, "points", results)
        return self._points_to_candidates(points, score_key="rrf_score")
 
    def search_candidates(
        self,
        query_vector: list[float],
        fetch_k: int = 30,
        filename: str | None = None,
    ) -> list[dict]:
        """Dense-only candidate retrieval (kept for fallback / ablation)."""
 
        query_filter = self._build_filter(filename)
 
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            using=self.DENSE_NAME,
            with_payload=True,
            limit=fetch_k,
            query_filter=query_filter,
        )
 
        points = getattr(results, "points", results)
        return self._points_to_candidates(points, score_key="vector_score")
 
    def list_documents(self) -> list[str]:
        records, _ = self.client.scroll(
            collection_name=self.collection,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
 
        docs = set()
        for r in records:
            payload = r.payload or {}
            if "filename" in payload:
                docs.add(payload["filename"])
 
        return sorted(docs)
