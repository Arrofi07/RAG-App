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
    MatchAny,
    Range,
    Prefetch,
    FusionQuery,
    Fusion,
)

from custom_types import MetaFilter

load_dotenv()


class QdrantStorage:
    """
    Qdrant wrapper storing both dense and sparse vectors per chunk,
    enabling hybrid (dense + BM25-style sparse) retrieval with built-in
    Reciprocal Rank Fusion, and metadata filtering on arbitrary payload fields.

    Collection layout
    -----------------
    named dense vector  → "dense"   (1024-dim, Cosine)
    named sparse vector → "sparse"  (SparseVectorParams)

    Payload fields available for MetaFilter
    ----------------------------------------
    filename, category, author, year, tags, document_type, uploaded_at
    """

    DENSE_NAME  = "dense"
    SPARSE_NAME = "sparse"

    def __init__(
        self,
        url: str | None = None,
        collection: str = "docs",
        dim: int = 1024,
    ):
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection = collection
        self.dim = dim

        self.client = QdrantClient(url=self.url, timeout=30)

        if self.client.collection_exists(collection_name=self.collection):
            self._validate_existing_collection()
        else:
            self._create_collection()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _validate_existing_collection(self) -> None:
        info = self.client.get_collection(collection_name=self.collection)
        vectors_config = info.config.params.vectors

        if not isinstance(vectors_config, dict):
            raise ValueError(
                f"Collection '{self.collection}' uses a flat vector layout "
                f"from an older version. Delete it and re-ingest:\n"
                f"  curl -X DELETE http://localhost:6333/collections/{self.collection}"
            )

        dense_cfg = vectors_config.get(self.DENSE_NAME)
        if dense_cfg is None or dense_cfg.size != self.dim:
            existing = dense_cfg.size if dense_cfg else "unknown"
            raise ValueError(
                f"Collection '{self.collection}' has '{self.DENSE_NAME}' "
                f"size={existing}, expected {self.dim}. Delete and re-ingest."
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
        sparse_vectors: list[dict],
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
    # Filter building
    # ------------------------------------------------------------------

    def _build_filter(self, meta: MetaFilter | None) -> Filter | None:
        """
        Translate a MetaFilter into a Qdrant Filter with AND logic.

        Each non-None field becomes one `must` clause:
          - filename / category / author  → exact MatchValue
          - tags                          → MatchAny  (chunk matches if it has ANY tag)
          - year_from / year_to           → Range on the `year` integer field
        """
        if meta is None:
            return None

        must = []

        if meta.filename:
            must.append(FieldCondition(
                key="filename",
                match=MatchValue(value=meta.filename),
            ))

        if meta.category:
            must.append(FieldCondition(
                key="category",
                match=MatchValue(value=meta.category),
            ))

        if meta.author:
            must.append(FieldCondition(
                key="author",
                match=MatchValue(value=meta.author),
            ))

        if meta.tags:
            # MatchAny: chunk matches if its `tags` payload list contains
            # at least one of the requested tags.
            must.append(FieldCondition(
                key="tags",
                match=MatchAny(any=meta.tags),
            ))

        if meta.year_from is not None or meta.year_to is not None:
            must.append(FieldCondition(
                key="year",
                range=Range(
                    gte=meta.year_from,  # None → no lower bound
                    lte=meta.year_to,    # None → no upper bound
                ),
            ))

        return Filter(must=must) if must else None

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

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
                "id":      str(r.id),
                "text":    text,
                "source":  payload.get("source", ""),
                "page":    payload.get("page"),
                score_key: r.score,
            })
        return candidates

    def search_hybrid_candidates(
        self,
        dense_vector: list[float],
        sparse_vector: dict,
        fetch_k: int = 30,
        meta: MetaFilter | None = None,
    ) -> list[dict]:
        """
        Hybrid retrieval: dense + sparse → server-side RRF fusion.

        RRF score = Σ 1 / (rank_i + 60)  for each retrieval list i.
        Parameter-free and robust to score-scale differences.
        """
        query_filter = self._build_filter(meta)

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
        meta: MetaFilter | None = None,
    ) -> list[dict]:
        """Dense-only retrieval (ablation / fallback)."""
        query_filter = self._build_filter(meta)

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

    def list_documents(self) -> list[dict]:
        """
        Return all unique documents with their stored metadata.
        Used by the UI to populate filter dropdowns.
        """
        records, _ = self.client.scroll(
            collection_name=self.collection,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )

        seen = {}
        for r in records:
            payload = r.payload or {}
            fname = payload.get("filename")
            if fname and fname not in seen:
                seen[fname] = {
                    "filename": fname,
                    "category": payload.get("category"),
                    "author":   payload.get("author"),
                    "year":     payload.get("year"),
                    "tags":     payload.get("tags", []),
                }

        return sorted(seen.values(), key=lambda d: d["filename"])