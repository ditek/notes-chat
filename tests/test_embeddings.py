import os
import socket
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

with patch.dict(os.environ, {"PYTHON_DOTENV_DISABLED": "1"}):
    import rag


class TestEmbeddings(Embeddings):
    model_name = "test-local-model"

    def embed_query(self, text):
        return [float(word in text.lower()) for word in ("ansible", "docker")]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory(prefix="notes-chat-test-")
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        config = patch.multiple(
            rag,
            DEFAULT_CHROMA_DIR=self.directory,
            DEFAULT_COLLECTION_NAME="embedding-tests",
        )
        config.start()
        self.addCleanup(config.stop)
        self.embeddings = TestEmbeddings()
        self.documents = [
            Document(
                page_content="Ansible inventory", metadata={"source": "ansible.md"}
            ),
            Document(
                page_content="Docker containers", metadata={"source": "docker.md"}
            ),
        ]

    def build_index(self, reset=False):
        return rag._build_index(
            self.documents,
            self.embeddings,
            reset=reset,
            persist_directory=self.directory,
        )

    def legacy_index(self):
        store = rag.Chroma(
            collection_name=rag.DEFAULT_COLLECTION_NAME,
            persist_directory=self.directory,
            embedding_function=self.embeddings,
        )
        store.add_documents([Document(page_content="Legacy vectors")])
        return store

    def test_cpu_model_is_shared_and_embeddings_work_without_network(self):
        rag._cached_embeddings.cache_clear()
        self.addCleanup(rag._cached_embeddings.cache_clear)
        encoder = Mock()
        encoder.encode.return_value = np.array([[0.1, 0.2]])

        def load_model(*args, **kwargs):
            time.sleep(0.05)  # Allow first requests to overlap during model loading.
            return encoder

        constructor = Mock(side_effect=load_model)
        module = SimpleNamespace(SentenceTransformer=constructor)
        with patch.dict(sys.modules, {"sentence_transformers": module}):
            with patch.object(
                socket.socket,
                "connect",
                side_effect=AssertionError("Unexpected network access"),
            ):
                with ThreadPoolExecutor(max_workers=4) as executor:
                    models = list(
                        executor.map(lambda _: rag.create_embeddings(), range(8))
                    )
                self.assertTrue(all(model is models[0] for model in models))
                self.assertEqual(models[0].embed_query("question"), [0.1, 0.2])
                self.assertEqual(models[0].embed_documents(["note"]), [[0.1, 0.2]])
        constructor.assert_called_once_with(
            rag.DEFAULT_EMBED_MODEL, cache_folder=None, device="cpu"
        )
        self.assertEqual(
            encoder.encode.call_args_list[0].kwargs,
            encoder.encode.call_args_list[1].kwargs,
        )

    def test_index_can_be_reopened_and_queried(self):
        self.build_index()
        store = rag.load_vector_store(self.embeddings)
        result = store.similarity_search("ansible", k=1)
        self.assertEqual(result[0].metadata["source"], "ansible.md")

    def test_legacy_vectors_cannot_be_queried_or_appended(self):
        legacy = self.legacy_index()
        for operation in (
            lambda: rag.load_vector_store(self.embeddings),
            self.build_index,
        ):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(RuntimeError, "incompatible embeddings"):
                    operation()
        self.assertEqual(legacy._collection.get()["documents"], ["Legacy vectors"])

    def test_changed_model_cannot_reuse_existing_vectors(self):
        self.build_index()
        self.embeddings.model_name = "another-model"
        with self.assertRaisesRegex(RuntimeError, "incompatible embeddings"):
            rag.load_vector_store(self.embeddings)
        with self.assertRaisesRegex(RuntimeError, "incompatible embeddings"):
            self.build_index()

    def test_new_collection_preserves_legacy_vectors(self):
        legacy = self.legacy_index()
        with patch.object(rag, "DEFAULT_COLLECTION_NAME", "local-embeddings"):
            self.build_index()
            self.assertEqual(
                rag.load_vector_store(self.embeddings)._collection.count(), 2
            )
        self.assertEqual(legacy._collection.get()["documents"], ["Legacy vectors"])

    def test_explicit_reset_replaces_legacy_vectors(self):
        self.legacy_index()
        self.build_index(reset=True)
        store = rag.load_vector_store(self.embeddings)
        self.assertCountEqual(
            store._collection.get()["documents"],
            [doc.page_content for doc in self.documents],
        )
