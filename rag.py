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


DEFAULT_NOTES_DIR = Path(os.getenv("NOTES_DIR", "notes"))
DEFAULT_CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
DEFAULT_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "notes")
DEFAULT_EMBED_MODEL = os.getenv(
    "HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_LLM_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
NOTES_REPO_CONTENTS_URL = os.getenv("NOTES_REPO_CONTENTS_URL", "")
_EMBEDDINGS_LOCK = Lock()


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
        raise RuntimeError(f"No markdown notes found in {notes_dir}")
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
    )


def create_embeddings():
    # Prevent concurrent first requests from loading multiple copies of the model.
    with _EMBEDDINGS_LOCK:
        return _cached_embeddings()


def _open_vector_store(embeddings, persist_directory, reset=False):
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    configuration = f"local-cpu-v1:{embeddings.model_name}"
    metadata_key = "notes_chat_embeddings"
    vector_store = Chroma(
        collection_name=DEFAULT_COLLECTION_NAME,
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_metadata={metadata_key: configuration},
    )
    if reset:
        vector_store.reset_collection()
    collection = vector_store._collection
    metadata = collection.metadata or {}
    if metadata.get(metadata_key) != configuration:
        if collection.count():
            raise RuntimeError(
                "Chroma contains incompatible embeddings."
                "Set CHROMA_COLLECTION to an unused name and run `index` or use --reset"
            )
        collection.modify(metadata={**metadata, metadata_key: configuration})
    return vector_store


def load_vector_store(embeddings):
    return _open_vector_store(embeddings, DEFAULT_CHROMA_DIR)


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
        raise RuntimeError("Missing OPENROUTER_API_KEY")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=token,
        timeout=45.0,
        max_retries=0,
    )


def _ask_llm(system_prompt, user_prompt, llm) -> str:
    try:
        response = llm.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=250,
            temperature=0.1,
            extra_body={
                "provider": {"max_price": {"prompt": 0, "completion": 0, "request": 0}},
            },
        )
    except APIStatusError as exc:
        messages = {
            401: "OpenRouter rejected the API key. Check OPENROUTER_API_KEY.",
            402: "OpenRouter denied access to this model. Check account access to free models.",
            403: "OpenRouter denied access to this model. Check account access to free models.",
            404: "OpenRouter model is unavailable. Check OPENROUTER_MODEL.",
            429: "OpenRouter rate limit reached. Please retry later.",
        }
        raise RuntimeError(
            messages.get(
                exc.status_code,
                "OpenRouter request failed. Retry later or check OPENROUTER_MODEL.",
            )
        ) from exc
    except APIConnectionError as exc:
        raise RuntimeError("Could not reach OpenRouter. Please retry later.") from exc
    except APIError as exc:
        raise RuntimeError(
            "OpenRouter returned an invalid response. Please retry later."
        ) from exc
    answer = response.choices[0].message.content if response.choices else None
    if not answer or not answer.strip():
        raise RuntimeError("OpenRouter returned no answer. Please retry later.")
    return answer


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
