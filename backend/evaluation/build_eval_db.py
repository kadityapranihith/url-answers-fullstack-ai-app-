import json
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from app.rag_chain import (
    qdrant_client,
    get_embedding_model
)

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams


# Load dataset
with open("evaluation/dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Get unique URLs
urls = list(dict.fromkeys(item["url"] for item in dataset))

print(f"Found {len(urls)} unique URLs")


# Load articles
loader = WebBaseLoader(urls)
docs = loader.load()

print(f"Loaded {len(docs)} documents")


# Use EXACT same chunking as your current RAG
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

print(f"Created {len(chunks)} chunks")


# Add stable chunk IDs
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i


collection_name = "evaluation-testing"


# Create fresh evaluation collection
if qdrant_client.collection_exists(collection_name):
    qdrant_client.delete_collection(collection_name)

qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE
    )
)


# Store chunks in Qdrant
QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=get_embedding_model(),
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    collection_name=collection_name
)

print("Evaluation database created successfully!")