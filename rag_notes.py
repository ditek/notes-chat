from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


ROOT = Path(__file__).parent
NOTES_DIR = ROOT / "notes"
DB_DIR = ROOT / "work" / "chroma_notes"
COLLECTION_NAME = "markdown_notes"


def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing HF token. Copy .env.example to .env and set HF_TOKEN."
        )
    return token


def load_markdown_notes(notes_dir: Path = NOTES_DIR) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(notes_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path.relative_to(ROOT))},
                )
            )
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def make_embeddings() -> HuggingFaceEndpointEmbeddings:
    token = get_hf_token()
    model = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    provider = os.getenv("HF_PROVIDER", "auto")

    return HuggingFaceEndpointEmbeddings(
        model=model,
        task="feature-extraction",
        provider=provider,
        huggingfacehub_api_token=token,
    )


def make_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_DIR),
        embedding_function=make_embeddings(),
    )


def build_index(reset: bool = False) -> None:
    if reset and DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    docs = load_markdown_notes()
    if not docs:
        raise RuntimeError(f"No markdown files found in {NOTES_DIR}")

    chunks = split_documents(docs)
    vector_store = make_vector_store()
    vector_store.add_documents(chunks)

    print(f"Loaded {len(docs)} markdown file(s).")
    print(f"Indexed {len(chunks)} chunk(s) into {DB_DIR}.")


def make_chat_model() -> ChatHuggingFace:
    token = get_hf_token()
    repo_id = os.getenv("HF_LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    provider = os.getenv("HF_PROVIDER", "auto")

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        provider=provider,
        huggingfacehub_api_token=token,
        max_new_tokens=350,
        temperature=0.2,
        do_sample=False,
    )
    return ChatHuggingFace(llm=llm)


def format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] Source: {source}\n{doc.page_content}")
    return "\n\n".join(parts)


def answer_question(question: str, k: int = 4) -> str:
    vector_store = make_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(question)

    prompt = PromptTemplate.from_template(
        """You answer questions using the user's markdown notes.

Rules:
- Use only the context below.
- If the notes do not contain the answer, say that you do not know from the notes.
- Include brief source references like [1] or [2].

Question:
{question}

Context:
{context}

Answer:"""
    )

    chain = prompt | make_chat_model() | StrOutputParser()
    return chain.invoke(
        {
            "question": question,
            "context": format_context(retrieved_docs),
        }
    )


def search_notes(question: str, k: int = 4) -> None:
    vector_store = make_vector_store()
    docs = vector_store.similarity_search(question, k=k)
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content.replace("\n", " ")[:280]
        print(f"\n[{i}] {source}\n{preview}...")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="RAG over local markdown notes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Embed notes into Chroma.")
    index_parser.add_argument("--reset", action="store_true", help="Rebuild the DB.")

    ask_parser = subparsers.add_parser("ask", help="Ask a question over the notes.")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--k", type=int, default=4)

    search_parser = subparsers.add_parser(
        "search", help="Show retrieved chunks without asking the LLM."
    )
    search_parser.add_argument("question")
    search_parser.add_argument("--k", type=int, default=4)

    args = parser.parse_args()

    if args.command == "index":
        build_index(reset=args.reset)
    elif args.command == "ask":
        print(answer_question(args.question, k=args.k))
    elif args.command == "search":
        search_notes(args.question, k=args.k)


if __name__ == "__main__":
    main()
