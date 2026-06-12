import pydantic

class RAGChunkAndSrc(pydantic.BaseModel):
    chunk: list[str]
    source: str = None

class RAGUpsertResult(pydantic.BaseModel):
    ingested_chunks: int

class RAGSearchResult(pydantic.BaseModel):
    context: list[str]
    sources: list[str]

class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_context: int