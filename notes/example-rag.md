# RAG Learning Notes

## What RAG Means

RAG stands for retrieval augmented generation. The basic idea is to retrieve
relevant information first, then give that information to a language model as
context for answering a question.

## Main Steps

1. Load documents.
2. Split them into chunks.
3. Create embeddings for each chunk.
4. Store the embeddings in a vector database.
5. Embed the user's question.
6. Retrieve similar chunks.
7. Ask the language model to answer using the retrieved chunks.

## Why Chunking Matters

Chunking makes documents easier to search. If a whole document is embedded as
one vector, the embedding may blur together many topics. Smaller chunks usually
make retrieval more precise, but chunks that are too small may lose context.

## What To Tune Later

Useful RAG settings include chunk size, chunk overlap, the embedding model, the
number of retrieved chunks, and the prompt that tells the model how to use the
retrieved context.
