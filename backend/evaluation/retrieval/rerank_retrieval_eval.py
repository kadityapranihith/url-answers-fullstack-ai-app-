"""
Reranker retrieval evaluation.

Pipeline:
    Qdrant Dense Top 10
        ↓
    Cohere Reranker
        ↓
    Reranked Top 10

Run from backend:
    python -m evaluation.retrieval.rerank_retrieval_eval
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

from app.rag_chain import (
    qdrant_client,
    get_embedding_model,
)

from app.reranker import rerank_documents

from langchain_qdrant import QdrantVectorStore


DATASET_PATH = "evaluation/dataset.json"

OUTPUT_PATH = (
    "evaluation/retrieval/"
    "rerank_retrieval_results.json"
)

COLLECTION_NAME = "evaluation-testing"

DENSE_K = 10
RERANK_TOP_N = 10

# Cohere Trial key limit:
# 10 API calls / minute.
# 7 seconds between calls keeps us safely below that.
REQUEST_DELAY_SECONDS = 7

# If a 429 still occurs, wait this long before retrying.
RATE_LIMIT_RETRY_DELAY = 65

MAX_RETRIES = 3


def load_json(path, default):
    if not os.path.exists(path):
        return default

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


def rerank_with_retry(question, documents):
    """
    Call the reranker with retry handling for
    Cohere rate limits.
    """

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            return rerank_documents(
                question=question,
                documents=documents,
                top_n=RERANK_TOP_N,
            )

        except Exception as exc:

            error_text = str(exc)

            # Detect Cohere 429 rate limit.
            if (
                "429" in error_text
                or "TooManyRequestsError" in error_text
                or "10 API calls / minute" in error_text
            ):

                print(
                    f"  Rate limit reached."
                    f" Waiting {RATE_LIMIT_RETRY_DELAY}s..."
                )

                time.sleep(
                    RATE_LIMIT_RETRY_DELAY
                )

                continue

            # Any other error should be surfaced.
            raise

    raise RuntimeError(
        "Reranker failed after maximum retries."
    )


def main():

    dataset = load_json(
        DATASET_PATH,
        []
    )

    if not dataset:
        raise FileNotFoundError(
            f"Dataset not found or empty: "
            f"{DATASET_PATH}"
        )

    # --------------------------------------------------
    # Load previous progress
    # --------------------------------------------------

    existing_results = load_json(
        OUTPUT_PATH,
        []
    )

    completed_ids = {
        item["id"]
        for item in existing_results
        if "id" in item
    }

    results = existing_results

    # --------------------------------------------------
    # Create vectorstore once
    # --------------------------------------------------

    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=get_embedding_model(),
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": DENSE_K
        }
    )

    # --------------------------------------------------
    # Header
    # --------------------------------------------------

    print("=" * 60)
    print("RERANKER RETRIEVAL EVALUATION")
    print("=" * 60)

    print(
        f"Questions: {len(dataset)}"
    )

    print(
        f"Already completed: "
        f"{len(completed_ids)}"
    )

    print(
        f"Collection: "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Dense K: {DENSE_K}"
    )

    print(
        f"Rerank K: {RERANK_TOP_N}"
    )

    print(
        f"Delay between requests: "
        f"{REQUEST_DELAY_SECONDS}s"
    )

    print()

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    for i, item in enumerate(
        dataset,
        start=1
    ):

        question_id = item["id"]

        # Resume support
        if question_id in completed_ids:

            print(
                f"[{i}/{len(dataset)}] "
                f"Skipping ID {question_id}"
            )

            continue

        question = item["question"]

        print(
            f"[{i}/{len(dataset)}] "
            f"ID {question_id}"
        )

        print(
            f"Question: {question}"
        )

        try:

            # ------------------------------------------
            # Dense retrieval
            # ------------------------------------------

            start_time = time.perf_counter()

            dense_docs = retriever.invoke(
                question
            )

            # ------------------------------------------
            # Respect Cohere rate limit
            # ------------------------------------------

            print(
                f"  Waiting "
                f"{REQUEST_DELAY_SECONDS}s "
                f"before reranking..."
            )

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

            # ------------------------------------------
            # Reranking
            # ------------------------------------------

            reranked = rerank_with_retry(
                question=question,
                documents=dense_docs,
            )

            latency_ms = (
                time.perf_counter()
                - start_time
            ) * 1000

            # ------------------------------------------
            # Save chunks
            # ------------------------------------------

            chunks = []

            for rank, result in enumerate(
                reranked,
                start=1
            ):

                doc = result["document"]

                chunks.append({
                    "rank": rank,

                    "chunk_id":
                        doc.metadata.get(
                            "chunk_id"
                        ),

                    "rerank_score":
                        result["score"],

                    "original_dense_rank":
                        result["original_index"]
                        + 1,

                    "text":
                        doc.page_content,
                })

            # ------------------------------------------
            # Save result
            # ------------------------------------------

            result = {
                "id": question_id,

                "question": question,

                "url": item["url"],

                "reference_answer":
                    item["reference_answer"],

                "retrieved_chunks":
                    chunks,

                "latency_ms":
                    latency_ms,
            }

            results.append(result)

            completed_ids.add(
                question_id
            )

            # ------------------------------------------
            # Checkpoint immediately
            # ------------------------------------------

            save_json(
                OUTPUT_PATH,
                results
            )

            print(
                f"  Completed ID "
                f"{question_id}"
            )

            print(
                f"  Latency: "
                f"{latency_ms:.2f} ms"
            )

            print(
                f"  Progress saved."
            )

            print()

        except Exception as exc:

            print(
                f"  FAILED ID "
                f"{question_id}"
            )

            print(
                f"  Error: "
                f"{type(exc).__name__}: {exc}"
            )

            print(
                "  Progress up to this point "
                "has already been saved."
            )

            print(
                "  Re-run the same command "
                "to resume."
            )

            break

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("RERANKER EVALUATION FINISHED")
    print("=" * 60)

    print(
        f"Completed: "
        f"{len(results)}/{len(dataset)}"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()