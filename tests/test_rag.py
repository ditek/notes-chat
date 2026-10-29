import json
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from openai import APIConnectionError, APIStatusError

import rag


def completion(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_local_cpu_embeddings_are_loaded_once_and_encode_without_network(monkeypatch):
    encoder = Mock()
    encoder.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    constructor = Mock(return_value=encoder)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=constructor),
    )
    monkeypatch.setattr(
        socket.socket, "connect", Mock(side_effect=AssertionError("network"))
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
        resources = list(executor.map(lambda _: rag.create_embeddings(), range(8)))
    assert all(resource is resources[0] for resource in resources)
    constructor.assert_called_once_with(
        rag.DEFAULT_EMBED_MODEL, cache_folder=None, device="cpu"
    )
    assert resources[0].embed_query("a question") == [0.1, 0.2, 0.3]
    assert resources[0].embed_documents(["a document"]) == [[0.1, 0.2, 0.3]]
    assert (
        encoder.encode.call_args_list[0].kwargs
        == encoder.encode.call_args_list[1].kwargs
    )


def test_openrouter_client_configuration():
    client = rag.create_llm()
    assert str(client.base_url) == "https://openrouter.ai/api/v1/"
    assert client.api_key == "test-key"
    assert client.max_retries == 0
    client.close()


def test_real_sdk_serializes_openrouter_request_and_extracts_response(monkeypatch):
    client = rag.create_llm()
    captured = []

    def send(request, **kwargs):
        captured.append(request)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "mock-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "google/gemma-4-31b-it:free",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "An inventory lists hosts.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    monkeypatch.setattr(client._client, "send", send)
    assert rag._ask_llm("system", "question", client) == "An inventory lists hosts."
    assert len(captured) == 1
    request = captured[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body["provider"]["max_price"] == {"prompt": 0, "completion": 0, "request": 0}
    assert body["reasoning"] == {"enabled": False}
    client.close()


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_key_is_actionable(monkeypatch, value):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    if value is not None:
        monkeypatch.setenv("OPENROUTER_API_KEY", value)
    with pytest.raises(rag.RAGError, match="Missing OPENROUTER_API_KEY"):
        rag.create_llm()


@pytest.mark.parametrize("model", ["qwen/qwen3.8-27b", "openrouter/auto", ""])
def test_paid_and_automatic_models_are_rejected(monkeypatch, model):
    monkeypatch.setenv("OPENROUTER_MODEL", model)
    with pytest.raises(rag.RAGError, match=":free"):
        rag.create_llm()


def test_rewrite_retrieval_and_answer_contract(embeddings):
    docs = [
        Document(
            page_content="ansible inventory lists hosts",
            metadata={"source": "a.md", "chunk_id": 0},
        )
    ]
    store = Mock()
    store.similarity_search.return_value = docs
    llm = Mock()
    llm.chat.completions.create.side_effect = [
        completion("What is an ansible inventory?"),
        completion("An inventory lists hosts."),
    ]
    result = rag.answer_question(
        "What is it?",
        store,
        llm,
        k=2,
        chat_history=[{"role": "user", "content": "Explain ansible"}],
    )
    assert result == (
        "An inventory lists hosts.",
        docs,
        "What is an ansible inventory?",
    )
    store.similarity_search.assert_called_once_with(
        "What is an ansible inventory?", k=2
    )
    calls = llm.chat.completions.create.call_args_list
    assert "Rewrite the user's question" in calls[0].kwargs["messages"][0]["content"]
    assert (
        "using only the provided context" in calls[1].kwargs["messages"][0]["content"]
    )
    assert "ansible inventory lists hosts" in calls[1].kwargs["messages"][1]["content"]
    for call in calls:
        assert call.kwargs["model"] == "google/gemma-4-31b-it:free"
        assert call.kwargs["temperature"] == 0.1
        assert call.kwargs["max_tokens"] == 250
        assert call.kwargs["extra_body"]["provider"]["max_price"] == {
            "prompt": 0,
            "completion": 0,
            "request": 0,
        }
    assert rag._rewrite_question("original", [], llm) == "original"
    assert llm.chat.completions.create.call_count == 2


@pytest.mark.parametrize(
    "status,match",
    [
        (401, "rejected"),
        (402, "rejected"),
        (403, "rejected"),
        (404, "unavailable"),
        (429, "rate limit"),
        (503, "unavailable"),
        (500, "retry later"),
    ],
)
def test_provider_errors_are_safe_and_never_fall_back(status, match):
    llm = Mock()
    response = httpx.Response(
        status, request=httpx.Request("POST", rag.OPENROUTER_BASE_URL)
    )
    llm.chat.completions.create.side_effect = APIStatusError(
        "sensitive provider payload", response=response, body=None
    )
    with pytest.raises(rag.RAGError, match=match) as error:
        rag._ask_llm("system", "user", llm)
    assert "sensitive" not in str(error.value)
    assert error.value.status_code == (429 if status == 429 else 503)
    llm.chat.completions.create.assert_called_once()


def test_connection_failure():
    llm = Mock()
    llm.chat.completions.create.side_effect = APIConnectionError(
        request=httpx.Request("POST", rag.OPENROUTER_BASE_URL)
    )
    with pytest.raises(rag.RAGError, match="reach OpenRouter"):
        rag._ask_llm("system", "user", llm)


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_generation_is_not_an_answer(content):
    llm = Mock()
    llm.chat.completions.create.return_value = completion(content)
    with pytest.raises(rag.RAGError, match="(no|empty) answer"):
        rag._ask_llm("system", "user", llm)


def test_index_persists_and_retrieves_with_same_embedding_config(
    tmp_path, embeddings, monkeypatch
):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "ansible.md").write_text("ansible inventory lists hosts")
    (notes / "docker.md").write_text("docker builds containers")
    monkeypatch.setattr(rag, "create_embeddings", lambda: embeddings)
    rag.index_notes(notes, persist_directory=rag.DEFAULT_CHROMA_DIR)
    loaded = rag.load_vector_store(embeddings)
    assert (
        loaded.similarity_search("ansible", k=1)[0]
        .metadata["source"]
        .endswith("ansible.md")
    )
    metadata = loaded._collection.metadata
    assert (
        json.loads(metadata[rag._EMBEDDING_METADATA_KEY])["model"]
        == embeddings.model_name
    )


@pytest.mark.parametrize(
    "metadata", [None, {"notes_chat_embedding_config": "other-configuration"}]
)
def test_legacy_and_incompatible_indexes_cannot_be_queried_or_appended(
    metadata, embeddings
):
    legacy = Chroma(
        collection_name=rag.DEFAULT_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=rag.DEFAULT_CHROMA_DIR,
        collection_metadata=metadata,
    )
    legacy.add_documents([Document(page_content="ansible legacy data")])
    for operation in [
        lambda: rag.load_vector_store(embeddings),
        lambda: rag._build_index(
            [Document(page_content="docker")],
            embeddings,
            persist_directory=rag.DEFAULT_CHROMA_DIR,
        ),
    ]:
        with pytest.raises(rag.RAGError, match="incompatible"):
            operation()
    assert legacy._collection.count() == 1
    assert legacy._collection.get()["documents"] == ["ansible legacy data"]


def test_reset_is_explicit_and_updates_signature(embeddings):
    legacy = Chroma(
        collection_name=rag.DEFAULT_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=rag.DEFAULT_CHROMA_DIR,
    )
    legacy.add_documents([Document(page_content="ansible legacy data")])
    rebuilt = rag._build_index(
        [Document(page_content="docker")],
        embeddings,
        reset=True,
        persist_directory=rag.DEFAULT_CHROMA_DIR,
    )
    assert rebuilt._collection.get()["documents"] == ["docker"]
    assert rag.load_vector_store(embeddings)._collection.count() == 1


def test_new_collection_preserves_legacy_vectors(monkeypatch, embeddings):
    legacy = Chroma(
        collection_name=rag.DEFAULT_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=rag.DEFAULT_CHROMA_DIR,
    )
    legacy.add_documents([Document(page_content="ansible legacy data")])
    monkeypatch.setattr(rag, "DEFAULT_COLLECTION_NAME", "notes-local-minilm-v1")
    rag._build_index(
        [Document(page_content="docker")],
        embeddings,
        persist_directory=rag.DEFAULT_CHROMA_DIR,
    )
    assert legacy._collection.get()["documents"] == ["ansible legacy data"]
    assert rag.load_vector_store(embeddings)._collection.get()["documents"] == [
        "docker"
    ]


def test_empty_index_has_rebuild_instructions(embeddings):
    with pytest.raises(rag.RAGError, match="index is empty"):
        rag.load_vector_store(embeddings)


def test_github_sync_preserves_local_notes(tmp_path, monkeypatch):
    (tmp_path / "personal.md").write_text("keep")
    listing = Mock()
    listing.json.return_value = [
        {"name": "Notes-ansible.md", "download_url": "https://example.com/note"},
        {"name": "other.txt"},
    ]
    content = Mock(text="ansible inventory")
    monkeypatch.setattr(rag.requests, "get", Mock(side_effect=[listing, content]))
    assert rag.sync_notes_from_github(tmp_path) == 1
    assert (tmp_path / "personal.md").read_text() == "keep"
    assert (tmp_path / "Notes-ansible.md").read_text() == "ansible inventory"
