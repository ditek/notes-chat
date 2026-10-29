# Hugging Face Deployment

This folder contains the Hugging Face Space overlay.

The main repository stays generic. During deployment, `deploy.py` creates a temporary tree from the current committed state, applies these Space-specific files, and pushes that generated commit to the configured Hugging Face remote.

Generated deployment changes:

- prepends `README.header.yml` to the root `README.md`
- copies this folder's `Dockerfile` to the deployment root
- removes the `deploy/` folder from the generated Space commit

## Local One-Time Setup

Add your Space as a Git remote named `hf`:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_NAMESPACE/YOUR_SPACE
```

Authenticate with Hugging Face Git using your local credential manager:

```bash
hf auth login --add-to-git-credential
```

With this route, the deploy script reads the Space URL from your local Git remote and authentication from your local credential manager.

## CI Setup

For CI or another non-interactive environment, do not configure a local `hf` remote. Set these environment variables instead:

```text
HF_SPACE_REMOTE_URL=https://huggingface.co/spaces/YOUR_NAMESPACE/YOUR_SPACE
HF_DEPLOY_USERNAME=YOUR_HF_USERNAME
HF_DEPLOY_TOKEN=hf_your_write_token
```

When `HF_SPACE_REMOTE_URL` is set, the deploy script automatically uses this env-var route. The token is passed to Git through `GIT_ASKPASS` at runtime and is not written into the generated deployment commit.

## Space Setup

Keep the Space's **Docker** SDK and **CPU Basic** hardware. Streamlit still listens
on port 8501 (`README.header.yml` supplies `app_port: 8501`). No GPU, Hugging Face
Inference Providers token, or paid inference subscription is needed.

In **Settings → Variables and secrets**, add:

| Type | Name | Value |
| --- | --- | --- |
| Secret | `OPENROUTER_API_KEY` | Your OpenRouter API key |
| Variable | `OPENROUTER_MODEL` | `google/gemma-4-31b-it:free` |
| Variable | `HF_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| Variable | `CHROMA_COLLECTION` | `notes-local-minilm-v1` (an unused name for migration) |
| Variable | `NOTES_REPO_CONTENTS_URL` | `https://api.github.com/repos/ditek/Notes/contents` or your public notes repo |
| Variable | `ENABLE_SIDEBAR_CONTROLS` | Normally `false`; see initial indexing below |

The default free model was verified in OpenRouter's [catalog](https://openrouter.ai/api/v1/models)
and [endpoints](https://openrouter.ai/api/v1/models/google/gemma-4-31b-it:free/endpoints)
on **2026-10-29**. The requested `qwen/qwen3.8-27b:free` had no endpoints and no
free Qwen appeared in the catalog. Model IDs ending in `:free` are required and
requests limit provider prices to zero; the app never switches to a paid model.
If a free model becomes unavailable or rate limited, the UI displays a retry or
configuration message. Missing OpenRouter credentials do not stop Streamlit startup
or local indexing; asking a question reports the missing secret.

Remove obsolete Space variables/secrets `HF_PROVIDER`, `HF_LLM_MODEL`, and `HF_TOKEN`
if they were used for inference. **Keep deployment authentication** (`HF_DEPLOY_TOKEN`
in CI, or your local HF Git credentials): it has a separate purpose.

### Writable paths and persistence

The image runs as UID 1000 in `/home/user/app`. Its default `notes/` and `chroma_db/`
directories are writable, so CPU Basic works without a storage purchase. Chroma
persists across requests in that runtime, but the Space's default filesystem is
ephemeral: notes and vectors can disappear when the container restarts or rebuilds.
Resync and rebuild from the public notes repository when that happens.

If you already have a writable persistent volume at `/data`, set:

```text
CHROMA_DIR=/data/chroma_db
NOTES_DIR=/data/notes
```

Keep the existing Chroma directory when migrating and choose a new collection name
to preserve its legacy vectors. Do not point SQLite/Chroma at a read-only mount.
Validate SQLite locking/durability support on your storage backend before relying
on it; merely exposing object storage as files does not establish that support.
See Hugging Face's [storage documentation](https://huggingface.co/docs/hub/spaces-storage)
for the available mounts.

The Dockerfile preloads the public MiniLM embedding model into
`HF_HOME=/home/user/.cache/huggingface` at build time. Runtime reuses the image cache
without an HF token. The first build takes longer because it downloads PyTorch and
the model; later starts reuse these image layers. Changing `HF_EMBED_MODEL` downloads
that model on first use and requires an index rebuild. Keep `HF_HOME` writable;
mounting a different cache hides the preloaded files and triggers a new download.

CPU wheels avoid CUDA packages. The small encoder, batch size 16, and two CPU math
threads keep memory and CPU usage bounded for typical notes collections. One model
is shared per Python process, including concurrent requests; use a single app worker
on CPU Basic. Very large notes collections still need capacity monitoring.

### Initial indexing or migration in a Space

1. Pause app traffic while migrating. Keep existing markdown files and vectors.
2. Set `CHROMA_COLLECTION=notes-local-minilm-v1` (or another unused name). This is
   the non-destructive migration path; the legacy collection remains untouched.
3. Temporarily make the Space private and set `ENABLE_SIDEBAR_CONTROLS=true`.
   Those controls are intended for trusted users and have no separate admin login.
4. In Streamlit, click **Sync notes from GitHub** if the configured notes directory
   is empty, then **Rebuild index**. Both operations work without an OpenRouter key.
5. Ask a question and a follow-up, then inspect the displayed sources. Set
   `ENABLE_SIDEBAR_CONTROLS=false` before making the Space public again.

If running commands inside the container, the equivalent is:

```bash
python main.py sync-notes --notes-dir "${NOTES_DIR:-notes}"
python main.py index --notes-dir "${NOTES_DIR:-notes}"
```

These commands use the configured `CHROMA_DIR`, `CHROMA_COLLECTION`, and
`HF_EMBED_MODEL`. Indexing does not call OpenRouter. An empty, legacy, or mismatched
index produces instructions instead of querying or appending incompatible vectors.
To reuse a legacy collection name, stop all writers, back up the complete Chroma
directory, then explicitly run `python main.py index --notes-dir ... --reset`.
Only the selected collection's vectors are replaced; markdown notes are preserved.

## Deploy

From the repo root, after committing the feature branch (the script exports `HEAD`):

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -m 'not integration'
python deploy/huggingface/deploy.py --dry-run
python deploy/huggingface/deploy.py
```

The script requires a clean worktree and overlays the committed source onto the
Space's `main` branch. It does not upload `.env`, ignored notes, or local Chroma data.
Configure the Space secrets and variables separately in Settings before using chat.

To build and test the same Dockerfile locally when Docker is running:

```bash
docker build -f deploy/huggingface/Dockerfile -t notes-chat:cpu .
docker run --rm -p 8501:8501 --env-file .env notes-chat:cpu
```

The local `.dockerignore` excludes credentials, caches, notes, and Chroma data.
Initialize notes/index using the sidebar, or mount an existing compatible directory
with UID 1000 write permissions. The deploy script's existing Git credential handling
is unchanged and never passes deployment credentials to the application.
