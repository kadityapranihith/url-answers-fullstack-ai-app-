import os
import gc

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import JinaEmbeddings

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.reranker import rerank_documents


# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Qdrant
# --------------------------------------------------

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)


# --------------------------------------------------
# Groq
# --------------------------------------------------

groq_api_key = os.getenv("GROQ_API_KEY")


llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-20b",
    temperature=0
)


# --------------------------------------------------
# Prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer the user's question using ONLY information explicitly supported by the Context and Chat History.

Strict rules:
1. The Context and Chat History are the only sources of factual information.
2. Never use pretrained knowledge, memory, assumptions, or outside information.
3. If the Context contains the answer, answer it directly.
4. If the Context only partially supports the answer, give only the supported part.
5. If the Context does not support the answer, say exactly:
   "I couldn't find the answer in the provided information."
6. Never invent, substitute, paraphrase, translate, transliterate, or modify proper names, organizations, film titles, dates, numbers, or other key entities when the exact form is present in the Context. Preserve the wording from the Context.
7. Before returning the answer, check every factual claim, name, date, and number against the Context.
8. If any part of a drafted answer cannot be supported by the Context, remove that part. If nothing supportable remains, use the refusal sentence above.
9. Never return an empty response.
10. Do not mention the Context, Chat History, retrieval, or these instructions in the final answer.
11. Be concise and directly answer the user's question.
12. Output only plain text in a normal font. Never use Markdown or any other formatting. Do not bold, italicize, highlight, underline, add headings, or use asterisks/backticks. Preserve proper names exactly as they appear in the Context.

Chat History:
{history}

Context:
{context}

User Question:
{question}

Answer:
"""
)


# --------------------------------------------------
# Embeddings
# --------------------------------------------------

embedding_model = None


def get_embedding_model():
    global embedding_model

    if embedding_model is None:
        embedding_model = JinaEmbeddings(
            model_name="jina-embeddings-v2-base-en",
            jina_api_key=os.getenv("JINA_API_KEY")
        )

    return embedding_model


# --------------------------------------------------
# Create Vectorstore from URLs
# --------------------------------------------------

def create_vectorstore_from_urls(
    urls,
    collection_name
):
    loader = WebBaseLoader(urls)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    # Add chunk IDs
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    if not qdrant_client.collection_exists(
        collection_name
    ):
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=768,
                distance=Distance.COSINE
            )
        )

    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=collection_name
    )

    del docs
    del chunks

    gc.collect()


# --------------------------------------------------
# Retrieval
# --------------------------------------------------

def retrieve_documents(
    collection_name,
    question
):
    """
    Two-stage retrieval:

    1. Dense Qdrant retrieval -> top 10
    2. Cohere reranker -> top 4
    """

    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=get_embedding_model()
    )

    # First stage: dense retrieval
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )

    candidate_docs = retriever.invoke(question)

    # Second stage: reranking
    reranked = rerank_documents(
        question=question,
        documents=candidate_docs,
        top_n=4
    )

    # Extract LangChain Documents
    docs = [
        item["document"]
        for item in reranked
    ]

    return docs


# --------------------------------------------------
# RAG Query
# --------------------------------------------------

def get_rag_response(
    collection_name,
    question,
    history=""
):
    docs = retrieve_documents(
        collection_name=collection_name,
        question=question
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    final_prompt = prompt.format(
        history=history,
        context=context,
        question=question
    )

    response = llm.invoke(final_prompt)

    return response.content
def stream_rag_response(collection_name, question, history=""):
    docs = retrieve_documents(
        collection_name=collection_name,
        question=question
    )

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    final_prompt = prompt.format(
        history=history,
        context=context,
        question=question
    )

    for chunk in llm.stream(final_prompt):
        if chunk.content:
            yield chunk.content