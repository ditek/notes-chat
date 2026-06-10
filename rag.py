import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


#################### Indexing flow ###################

def _load_notes(notes_dir):
    notes = []
    for note_file in Path(notes_dir).rglob('*.md'):
        with open(note_file, 'r', encoding='utf-8') as f:
            notes.append({
                'content': f.read(),
                'metadata': {'source': str(note_file)}
            })
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
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "source": source,
                    "chunk_id": chunk_id,
                },
            })
            print(f"\tchunk {chunk_id}: {len(chunk_text)} chars")
    return chunks


def _make_documents(chunks):
    return [
        Document(
            page_content=chunk['content'],
            metadata=chunk['metadata']
        ) for chunk in chunks
    ]


def _build_index(documents, embeddings, reset=False):
    vector_store = Chroma(
        collection_name="notes",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )
    if reset:
        print("Resetting vector store collection...")
        vector_store.reset_collection()
    vector_store.add_documents(documents)
    return vector_store


def index_notes(notes_dir, reset=False):
    notes = _load_notes(notes_dir)
    chunks = _chunk_notes(notes)
    documents = _make_documents(chunks)
    embeddings = create_embeddings()
    vector_store = _build_index(documents, embeddings, reset=reset)
    return vector_store


################### Asking flow ###################

def create_embeddings():
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
    )


def load_vector_store(embeddings):
    return Chroma(
        collection_name="notes",
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )


def _retrieve_context(question: str, vector_store: Chroma, k=3):
    return vector_store.similarity_search(question, k=k)


def _format_context(docs):
    context_parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata["source"]
        chunk_id = doc.metadata["chunk_id"]

        context_parts.append(
            f"[{i}] Source: {source}, chunk {chunk_id}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(context_parts)


def create_llm():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError("Missing HF_TOKEN or HUGGINGFACEHUB_API_TOKEN")
    return InferenceClient(
        model="Qwen/Qwen3-4B-Instruct-2507",
        provider="nscale",
        api_key=token,
    )


def _ask_llm(system_prompt, user_prompt, llm) -> str | None:
    response = llm.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=250,
        temperature=0.1,
    )
    return response.choices[0].message.content


def _format_chat_history(chat_history, max_messages=6):
    recent = chat_history[-max_messages:]
    parts = []
    for message in recent:
        role = message["role"]
        content = message["content"]
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def answer_question(question: str, vector_store: Chroma, llm, k: int = 3, chat_history=None):
    docs = _retrieve_context(question, vector_store, k=k)
    context = _format_context(docs)
    chat_history_str = _format_chat_history(chat_history) if chat_history else ""

    from textwrap import dedent
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

    return answer, docs
