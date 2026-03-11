# AI Research Tool (RAG Chat Application)

An AI-powered research assistant that allows users to create chat sessions from web URLs and ask questions about the content using Retrieval-Augmented Generation (RAG).

The system extracts information from provided web pages, converts them into embeddings, stores them in a vector database, and uses an LLM to generate context-aware answers.

---

## Features

* URL-based research chat
* Retrieval-Augmented Generation (RAG)
* Conversational follow-up questions
* Multiple chat sessions per user
* Firebase authentication
* Chat history persistence
* Vector search using FAISS
* ChatGPT-style interface
* Automatic rebuilding of embeddings when needed

---

## System Architecture

User Browser
↓
Frontend (HTML / CSS / JavaScript)
↓
Firebase Authentication
↓
FastAPI Backend
↓
RAG Pipeline
↓
FAISS Vector Search
↓
LLM (Groq - Llama 3.1)

---

## Tech Stack

Frontend

* HTML
* CSS
* JavaScript

Backend

* FastAPI
* Python

AI / RAG

* LangChain
* FAISS Vector Store
* HuggingFace Embeddings (all-MiniLM-L6-v2)
* Groq LLM (Llama 3.1)

Database / Auth

* Firebase Authentication
* Firebase Firestore

---

## Project Structure

```
AI-research-tool
│
├── backend
│   ├── app
│   │   ├── auth.py
│   │   ├── database.py
│   │   ├── firebase_config.py
│   │   ├── main.py
│   │   └── rag_chain.py
│   │
│   ├── requirements.txt
│   └── .env
│
├── frontend
│   ├── index.html
│   ├── script.js
│   └── style.css
│
└── README.md
```

---

## How It Works

### 1. User Login

Users authenticate using Firebase Authentication.

### 2. Create Research Chat

Users paste URLs of articles or research sources.

### 3. Document Processing

The backend:

* Scrapes web pages
* Splits documents into chunks
* Converts chunks to embeddings
* Stores them in a FAISS vector database

### 4. Question Answering

When the user asks a question:

1. Chat history is included
2. Relevant chunks are retrieved from FAISS
3. Context + question is sent to the LLM
4. The LLM generates an answer

### 5. Chat Memory

Chat history is stored in Firebase Firestore and reused for follow-up questions.

---

## Installation

### 1. Clone Repository

```
git clone https://github.com/yourusername/AI-research-tool.git
cd AI-research-tool
```

---

### 2. Setup Backend

```
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env`

```
GROQ_API_KEY=your_api_key_here
```

---

### 3. Run Backend

```
uvicorn app.main:app --reload
```

Backend runs at:

```
http://127.0.0.1:8000
```

---

### 4. Run Frontend

Open:

```
frontend/index.html
```

in a browser.

---

## API Endpoints

### Health Check

```
GET /
```

---

### Create Chat from URLs

```
POST /process_urls
```

Body:

```
{
  "urls": ["https://example.com/article"]
}
```

---

### Ask Question

```
POST /chat
```

Body:

```
{
  "chat_id": "uuid",
  "message": "Your question"
}
```

---

### Get User Chats

```
GET /chats
```

---

### Load Chat History

```
GET /chat/{chat_id}
```

---

### Delete Chat

```
DELETE /chat/{chat_id}
```

---

## Key Design Decisions

### Conversational RAG

The system includes recent chat history in the prompt so the AI can answer follow-up questions.

### FAISS Vector Database

FAISS enables fast similarity search over document embeddings.

### Stateless Backend

Vectorstores are rebuilt automatically if the server restarts.

### Secure Authentication

Firebase handles authentication and token verification.

---

## Future Improvements

* Streaming responses
* Chat title generation using LLM
* Vector database persistence
* Multi-document retrieval optimization
* UI improvements
* Docker deployment
* Production hosting

---

## Example Use Case

Input URLs:

```
https://en.wikipedia.org/wiki/Artificial_intelligence
https://en.wikipedia.org/wiki/Machine_learning
```

User question:

```
What is machine learning?
```

Follow-up:

```
Who invented it?
```

The AI understands the context and answers correctly.

---

## Author

K.Aditya Pranihith
B.Tech CSE (AI/ML) – VIT AP

