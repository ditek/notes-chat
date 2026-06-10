import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient

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
    return notes


def _chunk_notes(notes, chunk_size=800, chunk_overlap=150):
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks = []
    for note in notes:
        content = note['content']
        print(f'Processing {note["metadata"]["source"]} with {len(content)} chars')
        content = note['content']
        chunk_id = 0
        step = chunk_size - chunk_overlap
        i = 0
        while i < len(content):
            end = min(i + chunk_size, len(content))
            chunk = content[i:end]
            chunks.append({
                'content': chunk,
                'metadata': {
                    'source': note['metadata']['source'],
                    'chunk_id': chunk_id,
                    'start': i,
                    'end': end
                }
            })
            chunk_id += 1
            print(f'\tchunk {chunk_id}, chars {i}-{end}')
            i += step
    return chunks


def _make_documents(chunks):
    return [
        Document(
            page_content=chunk['content'],
            metadata=chunk['metadata']
        ) for chunk in chunks
    ]


def _build_index(documents, embeddings):
    from langchain_chroma import Chroma
    vector_store = Chroma(
        collection_name="notes",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )
    vector_store.add_documents(documents)
    return vector_store


def index_notes(notes_dir):
    notes = _load_notes(notes_dir)
    chunks = _chunk_notes(notes)
    documents = _make_documents(chunks)
    embeddings = create_embeddings()
    vector_store = _build_index(documents, embeddings)
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
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        provider="featherless-ai",
        api_key=token,
    )


def _ask_llm(prompt, llm) -> str | None:
    response = llm.chat_completion(
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.2,
    )
    return response.choices[0].message.content


def answer_question(question: str, vector_store: Chroma, llm, k: int = 3):
    docs = _retrieve_context(question, vector_store, k=k)
    context = _format_context(docs)

    from textwrap import dedent
    prompt = dedent(f"""
        Answer the question using only the context below.
        If the context does not contain the answer, say you do not know.

        Question:
        {question}

        Context:
        {context}

        Answer:
    """).strip()

    answer = _ask_llm(prompt, llm)
    return answer, docs
