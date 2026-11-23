import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rag import RAGEngine

AI_BASE_URL = os.getenv("AI_BASE_URL", "https://aipipe.org/openai/v1").rstrip("/")
AI_API_KEY = os.getenv("AI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
KB_PATH = os.getenv("KB_PATH", "knowledge_base.json")

engine = RAGEngine(
    kb_path=KB_PATH,
    base_url=AI_BASE_URL,
    api_key=AI_API_KEY,
    embedding_model=EMBEDDING_MODEL,
    chat_model=CHAT_MODEL,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not AI_API_KEY:
        raise RuntimeError("AI_API_KEY is missing. Add your AI Pipe token in Render environment variables.")
    await engine.initialize()
    yield

app = FastAPI(title="GA5 RAG Colosseum", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)

@app.get("/")
async def root():
    return {"service": "GA5 RAG Colosseum", "status": "ok", "endpoint": "POST /query"}

@app.get("/health")
async def health():
    return {"status": "ok", "ready": engine.ready}

@app.post("/query")
async def query(req: QueryRequest):
    if not engine.ready:
        raise HTTPException(status_code=503, detail="RAG engine is initializing")
    try:
        return await engine.query(req.question, req.top_k)
    except Exception as exc:
        # Do not expose provider credentials or raw upstream response details.
        raise HTTPException(status_code=502, detail="RAG query failed; check provider configuration")
