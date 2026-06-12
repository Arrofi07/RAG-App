from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct,VectorParams,Distance, Distance

class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection_name="docs", dim=3072):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection_name = collection_name
        self._ensure_collection()

    def upsert_vector(self, vector_id: str, vector: list[float], payload: dict):
        point = [PointStruct(id=vector_id, vector=vector, payload=payload) for i in range(len(vector))]
        self.client.upsert(collection_name=self.collection_name, points=point)

    def _ensure_collection(self):
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )

    def add_vector(self, vector_id: str, vector: list[float], payload: dict):
        point = PointStruct(id=vector_id, vector=vector, payload=payload)
        self.client.upsert(collection_name=self.collection_name, points=[point])

    def search(self, query_vector: list[float], top_k: int = 5):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            with_payload=True,
            limit=top_k
        )
        contexts = []
        sources = []
        for result in results:
            payload = getattr(result, 'payload', {})
            text = payload.get('text', '')
            source = payload.get('source', '')
            if text:
                contexts.append(text)
                sources.append(source)

        return {"context": contexts, "sources": list(sources)}