
import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import os
import uuid
import datetime
from data_loader import load_and_chunk_pdf, get_embedding
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag-app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)

async def rag_ingest_pdf(ctx: inngest.Context):
    file_path = ctx.event.data.get("file_path")
    if not file_path:
        return RAGUpsertResult(ingested_chunks=0, error="No file path provided")

    try:
        chunks = load_and_chunk_pdf(file_path)
        embeddings = [get_embedding(chunk) for chunk in chunks]
        vector_db = QdrantStorage()
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_db.add_vector(str(uuid()), embedding, {"text": chunk, "source": file_path})
        return RAGUpsertResult(ingested_chunks=len(chunks))
    except Exception as e:
        logging.error(f"Error occurred while ingesting PDF: {e}")
        return RAGUpsertResult(ingested_chunks=0, error=str(e))

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf])