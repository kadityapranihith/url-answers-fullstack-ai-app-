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
    return {"message": "RAG Backend Running"}


# -----------------------------
# Create Chat + Process URLs
# -----------------------------

@app.post("/process_urls")
def process_urls(request: URLRequest, req: Request):

    user_id = verify_token(req)

    chat_id = str(uuid.uuid4())

    create_vectorstore_from_urls(
        request.urls,
        chat_id
    )

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

    # Load previous chat messages
    chat_data = get_chat_history(user_id, request.chat_id)
    messages = chat_data.get("messages", [])

    # Build conversation history
    history = ""

    for m in messages[-6:]:
        history += f"{m['role']}: {m['content']}\n"

    # Save user message immediately
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

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )

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

    try:
        qdrant_client.delete_collection(
            collection_name=chat_id
        )
    except Exception as e:
        print(e)

    return {"message": "Chat deleted"}