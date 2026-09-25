from app.rag_chain import (
    qdrant_client,
    get_embedding_model,
)

from langchain_qdrant import QdrantVectorStore

from app.reranker import rerank_documents


COLLECTION_NAME = "evaluation-testing"


question = (
    "Which organization was responsible for "
    "carrying out the September 11 attacks?"
)


# -----------------------------
# Dense retrieval
# -----------------------------

vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=get_embedding_model(),
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 10}
)

docs = retriever.invoke(question)


print("=" * 70)
print("DENSE RETRIEVAL")
print("=" * 70)

for rank, doc in enumerate(docs, start=1):
    print(f"\nRank {rank}")
    print(f"Chunk ID: {doc.metadata.get('chunk_id')}")
    print(doc.page_content[:400])


# -----------------------------
# Reranking
# -----------------------------

results = rerank_documents(
    question=question,
    documents=docs,
    top_n=4,
)


print("\n")
print("=" * 70)
print("RERANKED RESULTS")
print("=" * 70)

for rank, result in enumerate(results, start=1):

    doc = result["document"]

    print(f"\nRank {rank}")
    print(f"Score: {result['score']:.6f}")
    print(
        f"Original dense rank: "
        f"{result['original_index'] + 1}"
    )
    print(
        f"Chunk ID: "
        f"{doc.metadata.get('chunk_id')}"
    )
    print(doc.page_content[:500])