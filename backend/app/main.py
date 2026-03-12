from fastapi import FastAPI
from pydantic import BaseModel
import uuid
from fastapi import Request
from app.auth import verify_token
from fastapi.middleware.cors import CORSMiddleware
from app.rag_chain import create_vectorstore_from_urls, get_rag_response
from app.database import (
    create_chat,
    save_message,
    get_user_chats,
    get_chat_history,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://127.0.0.1:5173",
        "https://url-answers.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory FAISS storage
chat_vectorstores = {}

# -----------------------------
# Request Models
# -----------------------------

class URLRequest(BaseModel):
    urls: list[str]

class ChatRequest(BaseModel):
    chat_id: str
    message: str


# -----------------------------
# Health Check
# -----------------------------

@app.get("/")
def home():
    return {"message": "RAG Backend Running"}


# -----------------------------
# Create Chat + Process URLs
# -----------------------------

@app.post("/process_urls")
def process_urls(request: URLRequest, req: Request):

    user_id = verify_token(req)

    chat_id = str(uuid.uuid4())

    vectorstore = create_vectorstore_from_urls(request.urls)

    chat_vectorstores[chat_id] = vectorstore

    create_chat(user_id, chat_id, request.urls)

    return {
        "chat_id": chat_id,
        "message": "Vectorstore created"
    }


# -----------------------------
# Ask Question (with history)
# -----------------------------

@app.post("/chat")
def chat(request: ChatRequest, req: Request):

    user_id = verify_token(req)

    # Get vectorstore
    vectorstore = chat_vectorstores.get(request.chat_id)

    if not vectorstore:

        chat_data = get_chat_history(user_id, request.chat_id)

        urls = chat_data.get("urls", [])

        vectorstore = create_vectorstore_from_urls(urls)

        chat_vectorstores[request.chat_id] = vectorstore


    # -----------------------------
    # Load previous chat messages
    # -----------------------------

    chat_data = get_chat_history(user_id, request.chat_id)
    messages = chat_data.get("messages", [])

    # Build conversation history
    history = ""

    for m in messages[-6:]:  # last 6 messages
        history += f"{m['role']}: {m['content']}\n"


    # -----------------------------
    # Generate response
    # -----------------------------

    response = get_rag_response(
        vectorstore,
        request.message,
        history
    )

    # Save messages
    save_message(user_id, request.chat_id, "user", request.message)
    save_message(user_id, request.chat_id, "assistant", response)

    return {"response": response}


# -----------------------------
# Get All Chats
# -----------------------------

@app.get("/chats")
def get_chats(req: Request):

    user_id = verify_token(req)

    chats = get_user_chats(user_id)

    return {"chats": chats}


# -----------------------------
# Load Chat History
# -----------------------------

@app.get("/chat/{chat_id}")
def load_chat(chat_id: str, req: Request):

    user_id = verify_token(req)

    chat_data = get_chat_history(user_id, chat_id)

    if not chat_data:
        return {"error": "Chat not found"}

    return {
        "chat_id": chat_id,
        "urls": chat_data.get("urls", []),
        "messages": chat_data.get("messages", []),
    }


# -----------------------------
# Delete Chat
# -----------------------------

@app.delete("/chat/{chat_id}")
def delete_chat(chat_id: str, req: Request):

    user_id = verify_token(req)

    from app.firebase_config import db

    chat_ref = (
        db.collection("users")
        .document(user_id)
        .collection("chats")
        .document(chat_id)
    )

    chat_ref.delete()

    if chat_id in chat_vectorstores:
        del chat_vectorstores[chat_id]

    return {"message": "Chat deleted"}