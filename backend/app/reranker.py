import os
from dotenv import load_dotenv
import cohere

load_dotenv()

api_key = os.getenv("COHERE_API_KEY")

if not api_key:
    raise RuntimeError("COHERE_API_KEY is not set.")

co = cohere.ClientV2(api_key=api_key)


def rerank_documents(
    question,
    documents,
    top_n=4,
    model="rerank-v4.0-fast",
):
    if not documents:
        return []

    texts = [
        doc.page_content
        for doc in documents
    ]

    response = co.rerank(
        model=model,
        query=question,
        documents=texts,
        top_n=min(top_n, len(texts)),
    )

    reranked = []

    for result in response.results:
        doc = documents[result.index]

        reranked.append({
            "document": doc,
            "score": result.relevance_score,
            "original_index": result.index,
        })

    return reranked