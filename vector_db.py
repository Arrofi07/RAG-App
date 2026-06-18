# vector_db.py

import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)

from qdrant_client.models import Filter, FieldCondition, MatchValue

from data_loader import get_embed_dim

load_dotenv()

EMBED_DIM = get_embed_dim()

class QdrantStorage:
    def __init__(
        self,
        url: str | None = None,
        collection: str = "docs",
        dim: int = EMBED_DIM,
    ):
        self.url = url or os.getenv(
            "QDRANT_URL",
            "http://localhost:6333",
        )

        self.collection = collection

        self.client = QdrantClient(
            url=self.url,
            timeout=30,
        )

        if self.client.collection_exists(
            collection_name=self.collection
        ):
            info = self.client.get_collection(
                collection_name=self.collection
            )
            existing_dim = info.config.params.vectors.size

            if existing_dim != dim:
                raise ValueError(
                    f"Qdrant collection '{self.collection}' already holds "
                    f"{existing_dim}-dim vectors, but this app is "
                    f"configured for {dim}-dim vectors. This usually means "
                    f"you switched embedding models. Embeddings from "
                    f"different models are never compatible -- delete the "
                    f"old collection (or point QdrantStorage at a new "
                    f"collection name) and re-ingest your documents."
                )
        else:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i],
            )
            for i in range(len(ids))
        ]

        self.client.upsert(
            collection_name=self.collection,
            points=points,
        )

    def search(
        self,
        query_vector,
        top_k=5,
        filename=None,
    ):

        query_filter = None

        if filename:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="filename",
                        match=MatchValue(value=filename),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
            query_filter=query_filter,
        )

        points = getattr(results, "points", results)

        contexts = []
        sources = set()

        for r in points:
            payload = r.payload or {}

            text = payload.get("text", "")
            source = payload.get("source", "")

            if text:
                contexts.append(text)

            if source:
                sources.add(source)

        return {
            "contexts": contexts,
            "sources": list(sources),
        }

    def search_candidates(
        self,
        query_vector,
        fetch_k=30,
        filename=None,
    ):
        """
        Retrieve a wider candidate pool for downstream reranking.

        Unlike `search()`, this does NOT deduplicate sources and keeps every
        chunk separate (with its own text + vector similarity score), since
        the reranker needs to score each chunk individually.
        """

        query_filter = None

        if filename:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="filename",
                        match=MatchValue(value=filename),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=fetch_k,
            query_filter=query_filter,
        )

        points = getattr(results, "points", results)

        candidates = []

        for r in points:
            payload = r.payload or {}
            text = payload.get("text", "")

            if not text:
                continue

            candidates.append({
                "text": text,
                "source": payload.get("source", ""),
                "page": payload.get("page"),
                "vector_score": r.score,
            })

        return candidates

    def list_documents(self):
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

