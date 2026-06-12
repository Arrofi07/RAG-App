from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(file_path: str):
    reader = PDFReader()
    documents = reader.load_data(file=file_path)
    text = [d.text for d in documents if getattr(d, 'text', None)]
    chunks = []
    for t in text:
        if t:
            chunks.extend(splitter.split_text(t))
    return chunks

def get_embedding(text: str):
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return [item.embedding for item in response.data][0]