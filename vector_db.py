# vector_db.py

import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)

load_dotenv()


class QdrantStorage:
    def __init__(
        self,
        url: str | None = None,
        collection: str = "docs",
        dim: int = 3072,
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

        if not self.client.collection_exists(
            collection_name=self.collection
        ):
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
        query_vector: list[float],
        top_k: int = 5,
    ) -> dict:

        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        )

        contexts = []
        sources = set()

        points = getattr(response, "points", [])

        for point in points:
            payload = point.payload or {}

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

