import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import streamlit as st
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

import api
import main
import rag


@pytest.fixture(autouse=True)
def clear_resources():
    resources = api.get_rag_resources
    resources.cache_clear()
    st.cache_resource.clear()
    yield
    resources.cache_clear()
    st.cache_resource.clear()


def test_api_health_and_answer_schema(monkeypatch):
    store, llm = Mock(), Mock()
    monkeypatch.setattr(api, "get_rag_resources", lambda: (store, llm))
    answer = Mock(
        return_value=(
            "An inventory lists hosts.",
            [
                Document(
                    page_content="ansible", metadata={"source": "a.md", "chunk_id": 0}
                )
            ],
            "ansible inventory",
        )
    )
    monkeypatch.setattr(api, "answer_question", answer)
    with TestClient(api.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.post(
            "/ask",
            json={
                "question": "What is it?",
                "k": 2,
                "chat_history": [{"role": "user", "content": "ansible"}],
            },
        )
    assert response.status_code == 200
    assert response.json() == {
        "answer": "An inventory lists hosts.",
        "retrieval_query": "ansible inventory",
        "sources": [{"source": "a.md", "chunk_id": 0, "content": "ansible"}],
    }
    assert answer.call_args.kwargs["chat_history"] == [
        {"role": "user", "content": "ansible"}
    ]


def test_api_missing_key_is_a_clear_service_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with TestClient(api.app) as client:
        response = client.post("/ask", json={"question": "hello"})
    assert response.status_code == 503
    assert "Missing OPENROUTER_API_KEY" in response.json()["detail"]


@pytest.mark.parametrize("status", [409, 429, 503])
def test_api_preserves_actionable_error_status(monkeypatch, status):
    monkeypatch.setattr(
        api, "get_rag_resources", Mock(side_effect=rag.RAGError("try later", status))
    )
    with TestClient(api.app) as client:
        response = client.post("/ask", json={"question": "hello"})
    assert response.status_code == status
    assert response.json() == {"detail": "try later"}


def test_admin_routes_and_auth_are_preserved(monkeypatch):
    monkeypatch.setenv("API_ADMIN_TOKEN", "test-admin")
    index = Mock()
    sync = Mock(return_value=2)
    monkeypatch.setattr(api, "index_notes", index)
    monkeypatch.setattr(api, "sync_notes_from_github", sync)
    with TestClient(api.app) as client:
        assert client.post("/admin/index", json={}).status_code == 401
        headers = {"Authorization": "Bearer test-admin"}
        assert client.post(
            "/admin/index",
            headers=headers,
            json={"notes_dir": "test-notes", "reset": False},
        ).json()["indexed"]
        assert (
            client.post(
                "/admin/sync-notes",
                headers=headers,
                json={"notes_dir": "test-notes", "rebuild_index": True},
            ).json()["downloaded"]
            == 2
        )
    assert index.call_count == 2
    sync.assert_called_once_with("test-notes")


def test_streamlit_missing_key_does_not_crash_or_pollute_history(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    assert not app.exception
    app.chat_input[0].set_value("hello").run()
    assert not app.exception
    assert "Missing OPENROUTER_API_KEY" in app.error[0].value
    assert app.session_state.messages == []


def test_streamlit_suggestions_sources_and_followups(monkeypatch, embeddings):
    from test_rag import completion

    store, llm = Mock(), Mock()
    store.similarity_search.return_value = [
        Document(
            page_content="ansible inventory lists hosts",
            metadata={"source": "a.md", "chunk_id": 0},
        )
    ]
    llm.chat.completions.create.side_effect = [
        completion("An inventory lists hosts."),
        completion("What does ansible inventory list?"),
        completion("Hosts."),
    ]
    monkeypatch.setattr(rag, "create_llm", lambda: llm)
    monkeypatch.setattr(rag, "create_embeddings", lambda: embeddings)
    monkeypatch.setattr(rag, "load_vector_store", lambda _: store)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state.messages[-1]["sources"][0].metadata["source"] == "a.md"
    app.chat_input[0].set_value("What does it list?").run()
    assert not app.exception
    assert app.session_state.messages[-1]["content"] == "Hosts."
    assert (
        app.session_state.messages[-1]["retrieval_query"]
        == "What does ansible inventory list?"
    )
    assert len(app.session_state.messages) == 4


def test_cli_missing_credentials_exits_cleanly(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    result = subprocess.run(
        [sys.executable, "main.py", "ask", "hello"], text=True, capture_output=True
    )
    assert result.returncode == 1
    assert "Missing OPENROUTER_API_KEY" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "command",
    [["index", "--reset"], ["sync-notes", "--index"], ["ask", "ansible", "--k", "2"]],
)
def test_cli_commands_remain_compatible(monkeypatch, command, embeddings, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", *command])
    index, sync = Mock(), Mock(return_value=2)
    monkeypatch.setattr(main, "index_notes", index)
    monkeypatch.setattr(main, "sync_notes_from_github", sync)
    monkeypatch.setattr(main, "create_llm", Mock())
    monkeypatch.setattr(main, "create_embeddings", lambda: embeddings)
    monkeypatch.setattr(main, "load_vector_store", Mock())
    monkeypatch.setattr(
        main, "answer_question", Mock(return_value=("answer", [], "ansible"))
    )
    main.main()
    if command[0] == "ask":
        assert "answer" in capsys.readouterr().out
    else:
        index.assert_called_once_with("notes", reset=True)
