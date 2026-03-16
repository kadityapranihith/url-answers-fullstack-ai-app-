import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import JinaEmbeddings

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# Embeddings


# LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0
)

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful, friendly, and knowledgeable AI assistant.

Your job is to answer the user's question using ONLY the information provided in the Context and Chat History. Your responses should be clear, conversational, and easy to understand.

Guidelines:

* Use ONLY the information available in the Context and Chat History to answer the question.
* Do NOT use any external knowledge, assumptions, or information that is not present in the provided material.
* If the Context clearly contains the answer, explain it clearly in a natural and friendly way.
* If the Context partially answers the question, respond using only the relevant available information without adding anything outside the provided material.
* If the answer cannot be found in the Context or Chat History, politely say:
  "I couldn't find the answer in the provided information."
* Do NOT guess, fabricate, or hallucinate any information.
* Do NOT mention that the answer comes from "context", "chat history", or "provided information".
* Avoid robotic or one-line responses.
* Write clear and complete explanations when appropriate.
* Use simple language that is easy for the user to understand.
* If helpful, organize the answer into short paragraphs or bullet points.
* Maintain a friendly and helpful tone similar to ChatGPT.

Chat History:
{history}

Context:
{context}

User Question:
{question}

Answer:

"""
)
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = JinaEmbeddings(
            model_name="jina-embeddings-v2-base-en",
            jina_api_key=os.getenv("JINA_API_KEY")
        )
    return embedding_model

# -------- Create Vectorstore from URLs --------
def create_vectorstore_from_urls(urls):

    loader = WebBaseLoader(urls)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, get_embedding_model())
    return vectorstore

# -------- RAG Query --------
def get_rag_response(vectorstore, question, history=""):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.format(
        history=history,
        context=context,
        question=question
    )

    response = llm.invoke(final_prompt)
    return response.content
