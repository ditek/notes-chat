import os
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import (
    DEFAULT_NOTES_DIR,
    answer_question,
    create_embeddings,
    create_llm,
    index_notes,
    load_vector_store,
    sync_notes_from_github,
)


def _allowed_origins():
    raw_origins = os.getenv("API_ALLOWED_ORIGINS", "*")
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


app = FastAPI(title="Notes Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=3, ge=1, le=5)
    chat_history: list[ChatMessage] = Field(default_factory=list)


class Source(BaseModel):
    source: str
    chunk_id: int | str
    content: str


class AskResponse(BaseModel):
    answer: str
    retrieval_query: str
    sources: list[Source]


class IndexRequest(BaseModel):
    notes_dir: str | None = None
    reset: bool = True


class SyncNotesRequest(BaseModel):
    notes_dir: str | None = None
    rebuild_index: bool = False
    reset: bool = True


@lru_cache(maxsize=1)
def get_rag_resources():
    embeddings = create_embeddings()
    vector_store = load_vector_store(embeddings)
    llm = create_llm()
    return vector_store, llm


def require_admin(authorization: str | None = Header(default=None)):
    token = os.getenv("API_ADMIN_TOKEN")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Admin API is disabled.",
        )

    expected = f"Bearer {token}"
    if authorization != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token.",
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    vector_store, llm = get_rag_resources()
    answer, docs, retrieval_query = answer_question(
        request.question,
        vector_store,
        llm,
        k=request.k,
        chat_history=[message.model_dump() for message in request.chat_history],
    )

    sources = [
        Source(
            source=doc.metadata.get("source", "unknown"),
            chunk_id=doc.metadata.get("chunk_id", "?"),
            content=doc.page_content,
        )
        for doc in docs
    ]
    return AskResponse(answer=answer, retrieval_query=retrieval_query, sources=sources)


@app.post("/admin/index", dependencies=[Depends(require_admin)])
def rebuild_index(request: IndexRequest):
    notes_dir = request.notes_dir or DEFAULT_NOTES_DIR
    index_notes(notes_dir, reset=request.reset)
    get_rag_resources.cache_clear()
    return {"status": "ok", "indexed": True, "notes_dir": str(notes_dir)}


@app.post("/admin/sync-notes", dependencies=[Depends(require_admin)])
def sync_notes(request: SyncNotesRequest):
    notes_dir = request.notes_dir or DEFAULT_NOTES_DIR
    count = sync_notes_from_github(notes_dir)

    indexed = False
    if request.rebuild_index:
        index_notes(notes_dir, reset=request.reset)
        get_rag_resources.cache_clear()
        indexed = True

    return {
        "status": "ok",
        "downloaded": count,
        "indexed": indexed,
        "notes_dir": str(notes_dir),
    }
