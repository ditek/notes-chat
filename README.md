# Chatty Notes

[![Code checks](https://github.com/ditek/notes-chat/actions/workflows/ruff-check.yaml/badge.svg)](https://github.com/ditek/notes-chat/actions/workflows/ruff-check.yaml)
[![Live Demo](https://img.shields.io/badge/Live-Demo-blue?style=flat-square)](https://diaa.dev/notes)

Chatty Notes is a conversational Retrieval-Augmented Generation (RAG) application for querying text documents using natural language. I use it for my markdown notes, but it works with any text-based documents. PDF support is planned.

It combines semantic search with an LLM to retrieve relevant information, generate context-aware answers, and expose the source passages used.

<img src="docs/example-card.svg" alt="application architecture" width="720">

## Key Features

- **Conversational retrieval:** Rewrites follow-up questions into standalone search queries using conversation history.
- **Semantic search:** Runs Sentence Transformers embeddings locally on CPU and stores vectors in Chroma.
- **Document indexing:** Splits documents into overlapping chunks with source metadata.
- **Grounded generation:** Instructs the LLM to answer using retrieved context and acknowledge missing information.
- **Source inspection:** Displays retrieved passages, filenames, chunk IDs, and retrieval queries.
- **Multiple interfaces:** Provides a Streamlit UI, FastAPI endpoints, and CLI access.

## Operation
<img src="docs/architecture.svg" alt="application architecture" width="720">

The current flow is:

1. Sync markdown notes from `github.com/ditek/Notes` (can be changed in the `.env` file) or use local files in `notes/`.
2. Split notes into chunks with LangChain's text splitter.
3. Embed chunks locally on CPU with `sentence-transformers/all-MiniLM-L6-v2`.
4. Store vectors in Chroma.
5. Rewrite follow-up questions using chat history.
6. Retrieve relevant chunks with top-k similarity retrieval.
7. Generate an answer through Groq (default: `qwen/qwen3.8-27b`).

## Setup
After cloning, run these commands.

```bash
uv sync --locked
uv run pre-commit install
cp .env.example .env
```

Edit `.env` and set your Groq API key and chat model:

```text
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=qwen/qwen3.8-27b
```

## Usage
### Sync Notes

Download notes from a public git repo (specified in `NOTES_REPO_CONTENTS_URL`):

```bash
uv run main.py sync-notes
```

Download and rebuild the vector index:

```bash
uv run main.py sync-notes --index
```

`sync-notes --index` replaces vectors in the selected Chroma collection. Back up an existing index before rebuilding it.

You can also index whatever markdown files are already in `notes/`:

```bash
uv run main.py index
```

Indexing without `--reset` appends chunks. Use `--reset` for a full rebuild after
backing up the index.

### Ask From The CLI

```bash
uv run main.py ask "What is an Ansible inventory?" --k 3
```

### Run The Streamlit App

```bash
uv run streamlit run app.py
```

The app includes:

- chat history
- query rewriting for follow-up questions
- source chunk display
- a `k` slider for retrieved chunks
- buttons to sync notes, rebuild the index, reload the index, and clear chat

Sidebar controls appear when `ENABLE_SIDEBAR_CONTROLS=true`. The **Rebuild index**
button replaces vectors in the selected collection, so back up existing data first.

### Run The FastAPI App

```bash
uv run uvicorn api:app --reload --port 8000
```

Ask a question:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is an Ansible inventory?",
    "k": 3,
    "chat_history": []
  }'
```

The response includes:

- `answer`
- `retrieval_query`
- `sources`

Admin endpoints are disabled unless `API_ADMIN_TOKEN` is set. When enabled, call them with:

```bash
Authorization: Bearer your_admin_token
```

Available admin endpoints:

- `POST /admin/sync-notes`
- `POST /admin/index`

### Environment Variables

- Set `GROQ_API_KEY` for question rewriting and answer generation. Indexing does not require it.
- Set `GROQ_MODEL` to select the chat model (default: `qwen/qwen3.8-27b`).
- Set `HF_EMBED_MODEL` to select the local CPU embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`).
- Set `HF_HOME` if you need a different writable model cache location.
- Set `CHROMA_DIR=./chroma_db` for local development.
- Set `CHROMA_COLLECTION` to select the collection to build and query (default: `notes`).
- Set `API_ALLOWED_ORIGINS` to a comma-separated list of browser origins allowed to call the FastAPI app. It defaults to `*`.
- Set `API_ADMIN_TOKEN` only if you want to enable protected sync/index endpoints.
- Keep `ENABLE_SIDEBAR_CONTROLS=false` to hide the sidebar in public embeds.
- Set `DEFAULT_RETRIEVAL_K` to control how many chunks are retrieved when the sidebar is hidden.

### Deployments

Deployment-specific files live under `deploy/`.

On Hugging Face Spaces, keep the Docker SDK and select CPU Basic hardware. Set
`GROQ_API_KEY` as a **Space secret**, and `GROQ_MODEL`, `HF_EMBED_MODEL`,
and `CHROMA_COLLECTION` as **Space variables**. Existing HF Git credentials or
`HF_DEPLOY_TOKEN` are still used separately for deployment.
Switching the chat provider does not require rebuilding the Chroma index.

The Docker image caches the default MiniLM model during build. Its default notes,
Chroma, and model-cache directories are writable. Set `CHROMA_DIR=/data/chroma_db`
only when writable persistent storage is mounted at `/data`. Without persistent
storage, resync downloaded notes and rebuild the index after container restarts.

See the [Hugging Face deployment guide](deploy/huggingface/README.md) for the deploy commands.

### Tests

```bash
uv run --locked python -m unittest discover -s tests -v
```

Tests mock model loading and Groq calls, and use temporary Chroma databases.
They require no API keys or model downloads and consume no generation credits.
