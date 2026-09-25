"""
Hybrid retrieval evaluation for URL Answers RAG.

Compares hybrid retrieval (Dense Qdrant + BM25 + RRF)
against the existing manually judged retrieval baseline.

Run from backend:
    python -m evaluation.retrieval.hybrid_retrieval_eval
"""

import json
import os
import sys
import time

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)

from app.hybrid_retriever import hybrid_retrieve


DATASET_PATH = "evaluation/dataset.json"
OUTPUT_PATH = "evaluation/retrieval/hybrid_retrieval_results.json"

COLLECTION_NAME = "evaluation-testing"

# Retrieve 10 because our retrieval metrics are evaluated at
# K = 1, 3, 5, and 10.
FINAL_K = 10
DENSE_K = 10
BM25_K = 10


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


def main():

    dataset = load_json(DATASET_PATH)

    print("=" * 60)
    print("HYBRID RETRIEVAL EVALUATION")
    print("=" * 60)

    print(f"Questions: {len(dataset)}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Dense K: {DENSE_K}")
    print(f"BM25 K: {BM25_K}")
    print(f"Final K: {FINAL_K}")
    print()

    results = []

    latencies = []

    for index, item in enumerate(dataset, start=1):

        question_id = item["id"]
        question = item["question"]

        print(
            f"[{index}/{len(dataset)}] "
            f"ID {question_id}"
        )

        start = time.perf_counter()

        docs = hybrid_retrieve(
            collection_name=COLLECTION_NAME,
            question=question,
            final_k=FINAL_K,
            dense_k=DENSE_K,
            bm25_k=BM25_K,
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(latency_ms)

        chunks = []

        for rank, doc in enumerate(docs, start=1):

            chunks.append({
                "rank": rank,
                "chunk_id": doc.metadata.get(
                    "chunk_id"
                ),
                "text": doc.page_content,
            })

        results.append({
            "id": question_id,
            "question": question,
            "url": item["url"],
            "reference_answer": item[
                "reference_answer"
            ],
            "retrieved_chunks": chunks,
            "latency_ms": latency_ms,
        })

    save_json(
        OUTPUT_PATH,
        results
    )

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        f"Average latency: "
        f"{sum(latencies) / len(latencies):.2f} ms"
    )

    print(
        f"Median latency: "
        f"{sorted(latencies)[len(latencies) // 2]:.2f} ms"
    )


if __name__ == "__main__":
    main()