"""
Generation evaluation for URL Answers RAG.

Run from the backend project directory:
    python evaluation/generation_eval.py

Uses the same evaluation collection, embedding model, prompt, and LLM
from app.rag_chain. The reference answer is NEVER sent to the LLM.
"""

import json
import os
import sys
import time
import re
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from app.rag_chain import qdrant_client, get_embedding_model, prompt, llm
from langchain_qdrant import QdrantVectorStore

DATASET_PATH = "evaluation/dataset.json"
OUTPUT_PATH = "evaluation/generation/generation_results_v2.json"
COLLECTION_NAME = "evaluation-testing"

# This matches the current production retriever in app.rag_chain.py.
K = 4

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def retrieve(question):
    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=get_embedding_model(),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": K})
    return retriever.invoke(question)


def generate(question, docs):
    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.format(
        history="",
        context=context,
        question=question,
    )

    response = llm.invoke(final_prompt)
    answer = response.content.strip()

    # Never return empty output
    if not answer:
        return "I couldn't find the answer in the provided information."

    # Remove Markdown formatting
    answer = re.sub(r"\*\*(.*?)\*\*", r"\1", answer)
    answer = re.sub(r"\*(.*?)\*", r"\1", answer)
    answer = answer.replace("`", "")

    return answer


def run_question(question):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            docs = retrieve(question)
            answer = generate(question, docs)

            chunks = []
            for rank, doc in enumerate(docs, start=1):
                chunks.append({
                    "rank": rank,
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "text": doc.page_content,
                })

            return answer, chunks

        except Exception as exc:
            last_error = exc
            print(
                f"  Attempt {attempt}/{MAX_RETRIES} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * attempt)

    raise last_error


def main():
    dataset = load_json(DATASET_PATH, None)

    if dataset is None:
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    if not qdrant_client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Qdrant collection '{COLLECTION_NAME}' does not exist."
        )

    existing = load_json(OUTPUT_PATH, [])
    completed_ids = {
        item["id"] for item in existing if "id" in item
    }

    results = existing

    print(f"Questions: {len(dataset)}")
    print(f"Already completed: {len(completed_ids)}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"K: {K}")
    print()

    for i, item in enumerate(dataset, start=1):
        qid = item["id"]

        if qid in completed_ids:
            print(f"[{i}/{len(dataset)}] Skipping ID {qid}")
            continue

        print(f"[{i}/{len(dataset)}] ID {qid}")
        print(f"Question: {item['question']}")

        try:
            answer, chunks = run_question(item["question"])

            result = {
                "id": qid,
                "question": item["question"],
                "url": item["url"],
                "reference_answer": item["reference_answer"],
                "retrieved_chunks": chunks,
                "generated_answer": answer,
            }

            results.append(result)
            save_json(OUTPUT_PATH, results)
            completed_ids.add(qid)

            print(f"Generated answer: {answer}")
            print(f"Progress saved -> {OUTPUT_PATH}")
            print()

        except Exception as exc:
            print(
                f"FAILED ID {qid}: {type(exc).__name__}: {exc}"
            )
            print("Re-run the same command to resume.")
            break

    print(
        f"Finished. Completed {len(results)}/{len(dataset)} questions."
    )



if __name__ == "__main__":
    main()
