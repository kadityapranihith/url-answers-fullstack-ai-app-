import os
import time
import statistics

from dotenv import load_dotenv

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.rag_chain import get_embedding_model
from app.reranker import rerank_documents


load_dotenv()


# ============================================================
# CONFIG
# ============================================================

COLLECTION_NAME = "evaluation-testing"
TOP_K = 10
RERANK_TOP_N = 4

QUESTIONS = [
    "What organization was responsible for the 9/11 attacks?",
    "What are the main causes of climate change?",
    "Who was the first person to walk on the Moon?",
    "What is the purpose of the World Wide Web?",
    "What are the main symptoms of diabetes?",
]


# ============================================================
# QDRANT CLIENT
# ============================================================

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)


# ============================================================
# VECTORSTORE / RETRIEVER
# ============================================================

vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=get_embedding_model()
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": TOP_K}
)


# ============================================================
# BENCHMARK
# ============================================================

dense_times = []
rerank_times = []
total_times = []


print("=" * 70)
print("Reranker Latency Benchmark")
print("=" * 70)
print(f"Collection: {COLLECTION_NAME}")
print(f"Dense Top-K: {TOP_K}")
print(f"Reranker Top-N: {RERANK_TOP_N}")
print(f"Questions: {len(QUESTIONS)}")
print()


for i, question in enumerate(QUESTIONS, start=1):

    print(f"[{i}/{len(QUESTIONS)}] {question}")

    # --------------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------------

    dense_start = time.perf_counter()

    candidate_docs = retriever.invoke(question)

    dense_end = time.perf_counter()

    dense_latency = (dense_end - dense_start) * 1000

    # --------------------------------------------------------
    # Cohere reranking
    # --------------------------------------------------------

    rerank_start = time.perf_counter()

    reranked_docs = rerank_documents(
        question=question,
        documents=candidate_docs,
        top_n=RERANK_TOP_N
    )

    rerank_end = time.perf_counter()

    rerank_latency = (rerank_end - rerank_start) * 1000

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    total_latency = dense_latency + rerank_latency

    dense_times.append(dense_latency)
    rerank_times.append(rerank_latency)
    total_times.append(total_latency)

    print(f"  Dense retrieval : {dense_latency:.2f} ms")
    print(f"  Cohere rerank   : {rerank_latency:.2f} ms")
    print(f"  Total retrieval : {total_latency:.2f} ms")
    print()


# ============================================================
# SUMMARY
# ============================================================

def print_stats(name, values):
    print(f"{name}:")
    print(f"  Average : {statistics.mean(values):.2f} ms")
    print(f"  Median  : {statistics.median(values):.2f} ms")
    print(f"  Min     : {min(values):.2f} ms")
    print(f"  Max     : {max(values):.2f} ms")
    print()


print("=" * 70)
print("RESULTS")
print("=" * 70)

print_stats("Dense Retrieval", dense_times)
print_stats("Cohere Reranking", rerank_times)
print_stats("Total Retrieval", total_times)

print("=" * 70)
print("Per-question total latencies")
print("=" * 70)

for i, latency in enumerate(total_times, start=1):
    print(f"Question {i}: {latency:.2f} ms")