# Chatty Notes

[![Code checks](https://github.com/ditek/notes-chat/actions/workflows/ruff-check.yaml/badge.svg)](https://github.com/ditek/notes-chat/actions/workflows/ruff-check.yaml)
[![Live Demo](https://img.shields.io/badge/Live-Demo-blue?style=flat-square)](https://diaa.dev/notes)

Chatty Notes is a conversational Retrieval-Augmented Generation (RAG) application for querying text documents using natural language. I use it for my markdown notes, but it works with any text-based documents. PDF support is planned.

It combines semantic search with an LLM to retrieve relevant information, generate context-aware answers, and expose the source passages used.

<img src="docs/example-card.svg" alt="application architecture" width="720">

## Key Features

- **Conversational retrieval:** Rewrites follow-up questions into standalone search queries using conversation history.
- **Semantic search:** Runs Sentence Transformers embeddings locally on CPU and stores them in Chroma.
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
5. Rewrite follow-up questions using chat history through OpenRouter.
6. Retrieve relevant chunks for each question with top-k similarity retrieval.
7. Generate a grounded answer through OpenRouter using an explicit free model.

## Setup
After cloning, run these commands.

```bash
uv sync --locked
uv run pre-commit install
cp .env.example .env
```

Edit `.env` and set:

```text
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_MODEL=google/gemma-4-31b-it:free
```

As of **2026-10-29**, OpenRouter's [model catalog](https://openrouter.ai/api/v1/models)
lists no free Qwen models, and [`qwen/qwen3.8-27b:free`](https://openrouter.ai/api/v1/models/qwen/qwen3.8-27b:free/endpoints)
has no serving endpoints. The default is the available `google/gemma-4-31b-it:free`.
Set `OPENROUTER_MODEL` explicitly if you prefer a different available free model or
if a free Qwen model becomes available. There is no automatic model fallback.
Paid model IDs are rejected, and requests cap prompt, completion, and request pricing
at zero. Free model availability and account rate limits can change.

No Hugging Face inference token or GPU is required. The public embedding model is
downloaded on first use and cached; indexing and retrieval then run locally. Only
question rewriting and answer generation use OpenRouter. Both retain the existing
250-token limit and temperature 0.1; optional reasoning is disabled for concise answers.

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

You can also index whatever markdown files are already in `notes/`:

```bash
uv run main.py index --reset
```

### Ask From The CLI

```bash
uv run main.py ask "What is an Ansible inventory?" --k 3
```

Build the index before asking. Missing credentials, empty or incompatible indexes,
OpenRouter rate limits, and unavailable models produce actionable errors in the CLI
and Streamlit. FastAPI returns a `detail` error with status 409 for incompatible
indexes, 429 for rate limits, and 503 for configuration/provider failures.

### Migrate an Existing Chroma Index

The hosted model name alone does not establish vector compatibility: the previous
provider's revision, pooling, and normalization settings were not recorded. Existing
nonempty collections without matching local configuration metadata are refused for
both querying and appending, even when dimensions match. New collections record
the backend, model name, model options, and document/query encoding options.

Stop running app/API processes while changing configuration or rebuilding. Keep your
existing `CHROMA_DIR` and markdown notes, and set an **unused** collection name in `.env`:

```text
CHROMA_COLLECTION=notes-local-minilm-v1
```

Then index the existing local files without resetting the old collection:

```bash
uv run main.py index --notes-dir notes
uv run main.py ask "What is an Ansible inventory?"
```

Restart Streamlit/FastAPI with the same configuration. This leaves the old `notes`
collection and all markdown files intact. Changing `HF_EMBED_MODEL` or embedding
settings later also requires a new collection or an explicit rebuild.

To reuse the same collection instead, first stop all writers and copy the entire
configured Chroma directory to a backup (example for the default local directory):

```bash
cp -a chroma_db chroma_db.before-local-embeddings
uv run main.py index --notes-dir notes --reset
```

`--reset`, the sidebar's **Rebuild index**, and admin indexing with `reset=true`
replace vectors in the selected collection. They never delete markdown notes.
An interrupted rebuild can be retried from those files. Avoid repeated indexing
without a reset in an existing compatible collection, as it appends chunks.

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

- Set `OPENROUTER_API_KEY` as a secret for generation; indexing requires no API key.
- Set `OPENROUTER_MODEL` to an available explicit `:free` model for rewriting and answering.
- Set `HF_EMBED_MODEL` to select the local CPU embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`).
- Set `HF_HOME` only if you need a different writable model download cache.
- Set `CHROMA_DIR=./chroma_db` for local development.
- Set `CHROMA_COLLECTION` to the collection to build and query; use a new name when migrating.
- Set `API_ALLOWED_ORIGINS` to a comma-separated list of browser origins allowed to call the FastAPI app. It defaults to `*`.
- Set `API_ADMIN_TOKEN` only if you want to enable protected sync/index endpoints.
- Keep `ENABLE_SIDEBAR_CONTROLS=false` to hide the sidebar in public embeds.
- Set `DEFAULT_RETRIEVAL_K` to control how many chunks are retrieved when the sidebar is hidden.

### Deployments

Deployment-specific files live under `deploy/`.

For deploying to Hugging Face Spaces, see:

[deployment instructions](deploy/huggingface/README.md).

### Verification

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -m 'not integration'
RUN_LOCAL_EMBEDDINGS_TEST=1 uv run pytest -m integration
```

Unit tests mock generation and model loading. The optional integration test downloads
the public model, verifies CPU inference and Chroma retrieval, then embeds a query
with network access blocked. No automated test consumes generation credits.
Linux installs CPU-only PyTorch wheels through uv; the Dockerfile does the same for pip.
