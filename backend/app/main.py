import os
import psutil
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
import uuid
from fastapi import Request
from app.auth import verify_token
from fastapi.middleware.cors import CORSMiddleware
from app.rag_chain import (
    create_vectorstore_from_urls,
    stream_rag_response,
    qdrant_client
)
from app.database import (
    create_chat,
    save_message,
    get_user_chats,
    get_chat_history,
)

from fastapi.responses import StreamingResponse


# -----------------------------
# Memory Logging Utility
# -----------------------------

def log_memory(label):
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    print(f"[MEMORY] {label}: {ram_mb:.2f} MB", flush=True)


# -----------------------------
# Lifespan (startup logging)
# -----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log_memory("SERVER STARTUP")
    yield
app = FastAPI(lifespan=lifespan)

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
# Keep only one active vectorstore in RAM


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
    log_memory("ROOT")
    return {"message": "RAG Backend Running"}


# -----------------------------
# Create Chat + Process URLs
# -----------------------------

@app.post("/process_urls")
def process_urls(request: URLRequest, req: Request):

    log_memory("PROCESS_URLS - START")

    log_memory("BEFORE verify_token")
    user_id = verify_token(req)
    log_memory("AFTER verify_token")

    chat_id = str(uuid.uuid4())
    log_memory("AFTER chat_id creation")

    log_memory("BEFORE create_vectorstore_from_urls")
    create_vectorstore_from_urls(
        request.urls,
        chat_id
    )
    log_memory("AFTER create_vectorstore_from_urls")

    log_memory("BEFORE create_chat")
    create_chat(user_id, chat_id, request.urls)
    log_memory("AFTER create_chat")

    log_memory("PROCESS_URLS - END")

    return {
        "chat_id": chat_id,
        "message": "Vectorstore created"
    }


# -----------------------------
# Ask Question (with history)
# -----------------------------

@app.post("/chat")
def chat(request: ChatRequest, req: Request):

    log_memory("CHAT - START")

    log_memory("CHAT - BEFORE AUTH")
    user_id = verify_token(req)
    log_memory("CHAT - AFTER AUTH")

    # Load previous chat messages
    log_memory("CHAT - BEFORE RETRIEVAL")
    chat_data = get_chat_history(user_id, request.chat_id)
    messages = chat_data.get("messages", [])

    # Build conversation history
    history = ""

    for m in messages[-6:]:
        history += f"{m['role']}: {m['content']}\n"

    log_memory("CHAT - AFTER RETRIEVAL")

    # Save user message immediately
    log_memory("CHAT - BEFORE LLM")
    save_message(
        user_id,
        request.chat_id,
        "user",
        request.message
    )

    def generate():

        full_response = ""

        for chunk in stream_rag_response(
            collection_name=request.chat_id,
            question=request.message,
            history=history
        ):
            full_response += chunk
            yield chunk

        # Save complete assistant response after streaming finishes
        save_message(
            user_id,
            request.chat_id,
            "assistant",
            full_response
        )

    log_memory("CHAT - RESPONSE CREATED")
    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )

# -----------------------------
# Get All Chats
# -----------------------------

@app.get("/chats")
def get_chats(req: Request):

    log_memory("GET_CHATS - START")

    user_id = verify_token(req)

    chats = get_user_chats(user_id)

    log_memory("GET_CHATS - END")
    return {"chats": chats}


# -----------------------------
# Load Chat History
# -----------------------------

@app.get("/chat/{chat_id}")
def load_chat(chat_id: str, req: Request):

    log_memory("LOAD_CHAT - START")

    user_id = verify_token(req)

    chat_data = get_chat_history(user_id, chat_id)

    if not chat_data:
        log_memory("LOAD_CHAT - END")
        return {"error": "Chat not found"}

    log_memory("LOAD_CHAT - END")
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

    log_memory("DELETE_CHAT - START")

    user_id = verify_token(req)

    from app.firebase_config import db

    chat_ref = (
        db.collection("users")
        .document(user_id)
        .collection("chats")
        .document(chat_id)
    )

    chat_ref.delete()

    try:
        qdrant_client.delete_collection(
            collection_name=chat_id
        )
    except Exception as e:
        print(e)

    log_memory("DELETE_CHAT - END")
    return {"message": "Chat deleted"}