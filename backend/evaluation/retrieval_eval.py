import json
import sys
import os
import time

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from app.rag_chain import retrieve_documents


# Load dataset
with open("evaluation/dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

collection_name = "evaluation-testing"

output_path = "evaluation/retrieval_results.json"

# Load existing results if the script was interrupted
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    completed_ids = {item["id"] for item in results}
    print(f"Resuming evaluation. Already completed: {len(results)}")
else:
    results = []
    completed_ids = set()


for i, item in enumerate(dataset, start=1):

    # Skip already completed questions
    if item["id"] in completed_ids:
        print(f"Skipping {i}/{len(dataset)} - already completed")
        continue

    question = item["question"]
    url = item["url"]

    print(f"\nProcessing {i}/{len(dataset)}")
    print(f"Question: {question}")

    # Retry retrieval if Jina/API temporarily fails
    docs = None

    for attempt in range(3):
        try:
            docs = retrieve_documents(
                collection_name=collection_name,
                question=question,
                k=10
            )
            break

        except Exception as e:
            print(
                f"Attempt {attempt + 1}/3 failed: {type(e).__name__}: {e}"
            )

            if attempt < 2:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("Failed after 3 attempts.")

    # If retrieval failed completely, don't crash the whole evaluation
    if docs is None:
        print("Skipping this question.")
        continue

    retrieved_chunks = []

    for rank, doc in enumerate(docs, start=1):
        retrieved_chunks.append({
            "rank": rank,
            "chunk_id": doc.metadata.get("chunk_id"),
            "text": doc.page_content
        })

    results.append({
        "id": item["id"],
        "question": question,
        "url": url,
        "reference_answer": item["reference_answer"],
        "retrieved_chunks": retrieved_chunks
    })

    # Save immediately after every successful question
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    # Small delay between requests
    time.sleep(1)


print("\nEvaluation retrieval completed.")
print(f"Results saved to: {output_path}")
print(f"Successfully processed: {len(results)}/{len(dataset)}")