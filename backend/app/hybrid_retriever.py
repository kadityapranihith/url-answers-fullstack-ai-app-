import re
import hashlib
import numpy as np

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from rank_bm25 import BM25Okapi

from app.rag_chain import qdrant_client, get_embedding_model


# Cache BM25 indexes so we don't rebuild them for every question.
_bm25_cache = {}


def tokenize(text: str):
    return re.findall(r"[A-Za-z0-9'-]+", text.lower())


def _load_documents_from_qdrant(collection_name):
    documents = []
    offset = None

    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}

            text = payload.get("page_content", "")
            metadata = payload.get("metadata", {})

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        if offset is None:
            break

    return documents


def _get_bm25_index(collection_name):
    if collection_name in _bm25_cache:
        return _bm25_cache[collection_name]

    documents = _load_documents_from_qdrant(collection_name)

    corpus = [
        tokenize(doc.page_content)
        for doc in documents
    ]

    bm25 = BM25Okapi(corpus)

    _bm25_cache[collection_name] = {
        "bm25": bm25,
        "documents": documents,
    }

    return _bm25_cache[collection_name]


def clear_bm25_cache(collection_name=None):
    if collection_name is None:
        _bm25_cache.clear()
    else:
        _bm25_cache.pop(collection_name, None)


def _doc_id(doc):
    chunk_id = doc.metadata.get("chunk_id")

    if chunk_id is not None:
        return str(chunk_id)

    return hashlib.md5(
        doc.page_content.encode("utf-8")
    ).hexdigest()


def hybrid_retrieve(
    collection_name,
    question,
    final_k=4,
    dense_k=10,
    bm25_k=10,
    rrf_k=60,
):
    # ----------------------------
    # Dense retrieval
    # ----------------------------
    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=get_embedding_model(),
    )

    dense_retriever = vectorstore.as_retriever(
        search_kwargs={"k": dense_k}
    )

    dense_docs = dense_retriever.invoke(question)

    # ----------------------------
    # BM25 retrieval
    # ----------------------------
    bm25_data = _get_bm25_index(collection_name)

    bm25 = bm25_data["bm25"]
    bm25_documents = bm25_data["documents"]

    query_tokens = tokenize(question)

    bm25_scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(bm25_scores)[::-1][:bm25_k]

    bm25_docs = [
        bm25_documents[i]
        for i in top_indices
    ]

    # ----------------------------
    # RRF fusion
    # ----------------------------
    fused_scores = {}
    documents = {}

    for rank, doc in enumerate(dense_docs, start=1):
        doc_id = _doc_id(doc)

        documents[doc_id] = doc

        fused_scores[doc_id] = (
            fused_scores.get(doc_id, 0)
            + 1 / (rrf_k + rank)
        )

    for rank, doc in enumerate(bm25_docs, start=1):
        doc_id = _doc_id(doc)

        documents[doc_id] = doc

        fused_scores[doc_id] = (
            fused_scores.get(doc_id, 0)
            + 1 / (rrf_k + rank)
        )

    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )

    return [
        documents[doc_id]
        for doc_id in ranked_ids[:final_k]
    ]