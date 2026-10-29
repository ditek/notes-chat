import json
import os
from functools import lru_cache
from pathlib import Path
from textwrap import dedent
from threading import Lock

import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import APIConnectionError, APIError, APIStatusError, OpenAI

load_dotenv()


class RAGError(RuntimeError):
    """An actionable error safe to display in any application interface."""

    def __init__(self, message, status_code=503):
        super().__init__(message)
        self.status_code = status_code


DEFAULT_NOTES_DIR = Path(os.getenv("NOTES_DIR", "notes"))
DEFAULT_CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
DEFAULT_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "notes")
DEFAULT_EMBED_MODEL = os.getenv(
    "HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
# No free Qwen endpoints were available in OpenRouter's 2026-10-29 catalog.
DEFAULT_LLM_MODEL = "google/gemma-4-31b-it:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
NOTES_REPO_CONTENTS_URL = os.getenv("NOTES_REPO_CONTENTS_URL", "")
_EMBEDDINGS_LOCK = Lock()
_EMBEDDING_METADATA_KEY = "notes_chat_embedding_config"


#################### Indexing flow ###################


def _load_notes(notes_dir):
    notes = []
    notes_path = Path(notes_dir)
    for note_file in notes_path.rglob("*.md"):
        with open(note_file, "r", encoding="utf-8") as f:
            notes.append({"content": f.read(), "metadata": {"source": str(note_file)}})
    print(f"Loaded {len(notes)} notes from {notes_dir}")
    return notes


def _chunk_notes(notes, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )
    chunks = []
    for note in notes:
        source = note["metadata"]["source"]
        content = note["content"]
        print(f"Chunking {source} with {len(content)} chars")
        split_texts = splitter.split_text(content)
        for chunk_id, chunk_text in enumerate(split_texts):
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        "source": source,
                        "chunk_id": chunk_id,
                    },
                }
            )
            print(f"\tchunk {chunk_id}: {len(chunk_text)} chars")
    return chunks


def _make_documents(chunks):
    return [
        Document(page_content=chunk["content"], metadata=chunk["metadata"])
        for chunk in chunks
    ]


def sync_notes_from_github(notes_dir=DEFAULT_NOTES_DIR):
    notes_path = Path(notes_dir)
    notes_path.mkdir(parents=True, exist_ok=True)

    response = requests.get(NOTES_REPO_CONTENTS_URL, timeout=30)
    response.raise_for_status()

    downloaded = 0
    for item in response.json():
        name = item.get("name", "")
        download_url = item.get("download_url")
        if (
            not name.startswith("Notes-")
            or not name.endswith(".md")
            or not download_url
        ):
            continue

        note_response = requests.get(download_url, timeout=30)
        note_response.raise_for_status()
        (notes_path / name).write_text(note_response.text, encoding="utf-8")
        downloaded += 1

    print(f"Downloaded {downloaded} note files to {notes_path}")
    return downloaded


def _build_index(
    documents, embeddings, reset=False, persist_directory=DEFAULT_CHROMA_DIR
):
    vector_store = _open_vector_store(embeddings, persist_directory, reset=reset)
    vector_store.add_documents(documents)
    return vector_store


def index_notes(
    notes_dir=DEFAULT_NOTES_DIR, reset=False, persist_directory=DEFAULT_CHROMA_DIR
):
    notes = _load_notes(notes_dir)
    if not notes:
        raise RAGError(f"No markdown notes found in {notes_dir}", status_code=400)
    chunks = _chunk_notes(notes)
    documents = _make_documents(chunks)
    embeddings = create_embeddings()
    vector_store = _build_index(
        documents,
        embeddings,
        reset=reset,
        persist_directory=persist_directory,
    )
    return vector_store


################### Asking flow ###################


@lru_cache(maxsize=1)
def _cached_embeddings():
    return HuggingFaceEmbeddings(
        model_name=DEFAULT_EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 16, "normalize_embeddings": False},
    )


def create_embeddings():
    # lru_cache alone can initialize twice during concurrent first requests.
    with _EMBEDDINGS_LOCK:
        try:
            return _cached_embeddings()
        except Exception as exc:
            raise RAGError(
                "Could not load the local CPU embedding model. Check HF_EMBED_MODEL, "
                "model download access, and write permissions for HF_HOME."
            ) from exc


def _embedding_signature(embeddings):
    return json.dumps(
        {
            "backend": "sentence-transformers-local-v1",
            "model": embeddings.model_name,
            "model_kwargs": embeddings.model_kwargs,
            "encode_kwargs": embeddings.encode_kwargs,
            "query_encode_kwargs": embeddings.query_encode_kwargs,
        },
        sort_keys=True,
    )


def _open_vector_store(embeddings, persist_directory, reset=False):
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    signature = _embedding_signature(embeddings)
    vector_store = Chroma(
        collection_name=DEFAULT_COLLECTION_NAME,
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_metadata={_EMBEDDING_METADATA_KEY: signature},
    )
    if reset:
        # Only explicit --reset/sidebar/admin actions may replace old vectors.
        vector_store.reset_collection()
    collection = vector_store._collection
    metadata = collection.metadata or {}
    if metadata.get(_EMBEDDING_METADATA_KEY) != signature:
        if collection.count():
            raise RAGError(
                "The Chroma collection has legacy or incompatible embeddings. "
                "Preserve it by setting CHROMA_COLLECTION=notes-local-minilm-v1 "
                "(or another unused name), then run `uv run main.py index`. "
                "See README.md for backup and rebuild instructions.",
                status_code=409,
            )
        collection.modify(metadata={**metadata, _EMBEDDING_METADATA_KEY: signature})
    return vector_store


def load_vector_store(embeddings):
    vector_store = _open_vector_store(embeddings, DEFAULT_CHROMA_DIR)
    if not vector_store._collection.count():
        raise RAGError(
            "The Chroma index is empty. Run `uv run main.py index` or use "
            "the Rebuild index sidebar control before asking questions."
        )
    return vector_store


def _retrieve_context(question: str, vector_store: Chroma, k=3):
    return vector_store.similarity_search(question, k=k)


def _format_context(docs):
    context_parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata["source"]
        chunk_id = doc.metadata["chunk_id"]

        context_parts.append(
            f"[{i}] Source: {source}, chunk {chunk_id}\n{doc.page_content}"
        )
    return "\n\n".join(context_parts)


def create_llm():
    token = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not token:
        raise RAGError(
            "Missing OPENROUTER_API_KEY. Set it in .env locally or as a "
            "Hugging Face Space secret. Indexing does not require this key."
        )
    _generation_model()
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=token,
        timeout=45.0,
        max_retries=0,
    )


def _generation_model():
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_LLM_MODEL).strip()
    if not model.endswith(":free"):
        raise RAGError("OPENROUTER_MODEL must select an explicit :free model.")
    return model


def _ask_llm(system_prompt, user_prompt, llm) -> str:
    model = _generation_model()
    try:
        response = llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=250,
            temperature=0.1,
            extra_body={
                "provider": {"max_price": {"prompt": 0, "completion": 0, "request": 0}},
                "reasoning": {"enabled": False},
            },
        )
    except APIStatusError as exc:
        if exc.status_code == 429:
            message = "OpenRouter free-model rate limit reached. Please retry later."
        elif exc.status_code in (401, 402, 403):
            message = (
                "OpenRouter rejected this request. Check OPENROUTER_API_KEY "
                "and the account's access to free models."
            )
        elif exc.status_code in (400, 404, 422, 502, 503):
            message = (
                f"OpenRouter model {model} is unavailable or rejected the request. "
                "Retry later or explicitly set OPENROUTER_MODEL to an available "
                ":free model. No paid fallback is used."
            )
        else:
            message = "OpenRouter could not complete the request. Please retry later."
        raise RAGError(message, 429 if exc.status_code == 429 else 503) from exc
    except APIConnectionError as exc:
        raise RAGError("Could not reach OpenRouter. Please retry later.") from exc
    except APIError as exc:
        raise RAGError(
            "OpenRouter returned an invalid response. Please retry later."
        ) from exc

    if not response.choices or not response.choices[0].message.content:
        raise RAGError(
            "OpenRouter returned no answer. Retry later or select another :free model."
        )
    content = response.choices[0].message.content.strip()
    if not content:
        raise RAGError("OpenRouter returned an empty answer. Please retry later.")
    return content


def _format_chat_history(chat_history, max_messages=6):
    recent = chat_history[-max_messages:]
    parts = []
    for message in recent:
        role = message["role"]
        content = message["content"]
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _rewrite_question(question: str, chat_history, llm) -> str:
    if not chat_history:
        return question
    chat_history_str = _format_chat_history(chat_history)

    system_prompt = dedent("""
        Rewrite the user's question to be self-contained based on the conversation history. Keep the meaning the same.
        Rules:
        - Use the conversation history only to resolve references like "it", "that", "this", or "they".
        - Do not answer the question.
        - Do not add extra explanation.
        - Return only the rewritten question.
    """).strip()
    user_prompt = dedent(f"""
        Conversation history:
        {chat_history_str}

        Current question:
        {question}

        Rewritten question:
    """).strip()
    rewritten = _ask_llm(system_prompt, user_prompt, llm)
    return rewritten.strip() if rewritten else question


def answer_question(
    question: str, vector_store: Chroma, llm, k: int = 3, chat_history=None
):
    if chat_history:
        retrieval_query = _rewrite_question(question, chat_history, llm)
    else:
        retrieval_query = question
    docs = _retrieve_context(retrieval_query, vector_store, k=k)
    context = _format_context(docs)
    chat_history_str = _format_chat_history(chat_history) if chat_history else ""

    system_prompt = dedent("""
        You answer questions using only the provided context from the user's notes.

        Rules:
        - Give the final answer only.
        - Do not explain how you found the answer.
        - Do not mention "the context", "the notes", or "the question" unless citing a source.
        - Synthesize all relevant context into one concise answer.
        - Do not answer separately for each source.
        - If multiple chunks contain the same fact, mention it once.
        - Do not use outside knowledge when possible.
        - Keep the answer to at most 4 sentences unless the question asks for steps or examples.
        - If the context does not answer the question, reply exactly: "I don't know from the notes."
    """).strip()

    user_prompt = dedent(f"""
        Conversation so far:
        {chat_history_str}

        Current question:
        {question}

        Context:
        {context}

    """).strip()

    answer = _ask_llm(system_prompt, user_prompt, llm)
    return answer, docs, retrieval_query
