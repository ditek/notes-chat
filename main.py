"""
This module implements a simple RAG system using LangChain and HuggingFace.
It includes two main flows:
1. Indexing flow: Load markdown notes, chunk them, create documents, generate embeddings, and build a vector store index.
2. Asking flow: Given a question, retrieve relevant chunks from the vector store, format the context, and ask an LLM to answer the question based on the retrieved context.
Usage:
1. To index notes (only need to do this once, or whenever notes change):
    main.py index
2. To ask questions:
    main.py ask "Your question here"
"""

import argparse

from rag import (
    index_notes,
    sync_notes_from_github,
    create_embeddings,
    load_vector_store,
    create_llm,
    answer_question,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("--notes-dir", default="notes")
    index_parser.add_argument("--reset", action="store_true", help="Reset the index by deleting existing vector store for the collection")

    sync_parser = subparsers.add_parser("sync-notes")
    sync_parser.add_argument("--notes-dir", default="notes")
    sync_parser.add_argument("--index", action="store_true", help="Rebuild the index after syncing notes")

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--k", type=int, default=3)

    args = parser.parse_args()

    if args.command == "index":
        index_notes(args.notes_dir, reset=args.reset)

    elif args.command == "sync-notes":
        count = sync_notes_from_github(args.notes_dir)
        print(f"Synced {count} notes")
        if args.index:
            index_notes(args.notes_dir, reset=True)

    elif args.command == "ask":
        embeddings = create_embeddings()
        vector_store = load_vector_store(embeddings)
        llm = create_llm()
        answer, docs, _ = answer_question(
            args.question,
            vector_store,
            llm,
            k=args.k,
        )
        print(answer)
        print("\nSources:")
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            chunk_id = doc.metadata.get("chunk_id", "?")
            start = doc.metadata.get("start", "?")
            end = doc.metadata.get("end", "?")
            print(f"[{i}] {source} chunk {chunk_id}")
            print(f"Content:\n{doc.page_content[:200]}\n{'-'*40}")
