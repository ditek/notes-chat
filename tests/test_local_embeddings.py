import os
import socket
from unittest.mock import Mock

import numpy as np
import pytest
from langchain_core.documents import Document

import rag


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_LOCAL_EMBEDDINGS_TEST") != "1",
    reason="Opt in to downloading the public model",
)
def test_real_cpu_model_and_offline_retrieval(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    embeddings = rag.create_embeddings()
    assert embeddings._client.device.type == "cpu"
    assert rag.create_embeddings() is embeddings
    # All model loading is complete. Embedding and Chroma operations must be local.
    monkeypatch.setattr(
        socket.socket,
        "connect",
        Mock(side_effect=AssertionError("unexpected network access")),
    )
    query = embeddings.embed_query("What is an Ansible inventory?")
    assert len(query) == 384
    assert np.isfinite(query).all()
    docs = [
        Document(
            page_content="An Ansible inventory lists the managed servers and groups of hosts.",
            metadata={"source": "ansible.md", "chunk_id": 0},
        ),
        Document(
            page_content="Bananas are a yellow fruit often used in desserts.",
            metadata={"source": "fruit.md", "chunk_id": 0},
        ),
    ]
    rag._build_index(docs, embeddings, persist_directory=rag.DEFAULT_CHROMA_DIR)
    store = rag.load_vector_store(embeddings)
    assert (
        store.similarity_search("What is an Ansible inventory?", k=1)[0].metadata[
            "source"
        ]
        == "ansible.md"
    )
