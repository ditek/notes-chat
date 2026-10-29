import os
from dataclasses import dataclass, field

# Tests and CLI subprocesses must never load real developer credentials from .env.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

import pytest
from langchain_core.embeddings import Embeddings

import rag


@dataclass
class LocalTestEmbeddings(Embeddings):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs: dict = field(default_factory=lambda: {"device": "cpu"})
    encode_kwargs: dict = field(default_factory=lambda: {"normalize_embeddings": False})
    query_encode_kwargs: dict = field(default_factory=dict)

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        words = text.lower().split()
        return [float(words.count(word)) for word in ("ansible", "docker", "python")]


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
    monkeypatch.setattr(rag, "DEFAULT_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(rag, "DEFAULT_COLLECTION_NAME", "test-notes")
    rag._cached_embeddings.cache_clear()
    yield
    rag._cached_embeddings.cache_clear()


@pytest.fixture
def embeddings():
    return LocalTestEmbeddings()
