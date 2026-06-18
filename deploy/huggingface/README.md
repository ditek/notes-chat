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

The Space should use persistent storage for Chroma. Mount a Storage Bucket at:

```text
/data
```

Then set this Space variable:

```text
CHROMA_DIR=/data/chroma_db
```

## Deploy

From the repo root:

```bash
python deploy/huggingface/deploy.py
```

Preview the generated changes without pushing:

```bash
python deploy/huggingface/deploy.py --dry-run
```
