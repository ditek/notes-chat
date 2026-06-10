"""
This module implements a simple RAG system using LangChain and HuggingFace.
It includes two main flows:
1. Indexing flow: Load markdown notes, chunk them, create documents, generate embeddings, and build a vector store index.
2. Asking flow: Given a question, retrieve relevant chunks from the vector store, format the context, and ask an LLM to answer the question based on the retrieved context.
Usage:
1. To index notes (only need to do this once, or whenever notes change):
    main.py index
2. To ask questions:
    main.py ask "Your question here
"""

import sys

from rag import (
    index_notes,
    create_embeddings,
    load_vector_store,
    create_llm,
    answer_question,
)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py [index|ask] [question]")
        sys.exit(1)
    command = sys.argv[1]
    if command == "index":
        index_notes('notes')
    elif command == "ask":
        if len(sys.argv) < 3:
            print("Usage: main.py ask [question]")
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        embeddings = create_embeddings()
        vector_store = load_vector_store(embeddings)
        llm = create_llm()
        answer, docs = answer_question(question, vector_store, llm)
        print(answer)
        print("\nSources:")
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            chunk_id = doc.metadata.get("chunk_id", "?")
            start = doc.metadata.get("start", "?")
            end = doc.metadata.get("end", "?")
            print(f"[{i}] {source} chunk {chunk_id} chars {start}-{end}")
    else:
        print("Unknown command. Use: index or ask")
        sys.exit(1)
