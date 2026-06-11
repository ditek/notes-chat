# Notes RAG with Hugging Face

This is a small conversational RAG app for asking questions over markdown notes.

The current flow is:

1. Sync markdown notes from `ditek/Notes` or use local files in `notes/`.
2. Split notes into chunks with LangChain's text splitter.
3. Embed chunks with Hugging Face.
4. Store vectors locally in Chroma.
5. Retrieve relevant chunks for each question.
6. Rewrite follow-up questions using chat history.
7. Answer with a Hugging Face chat model.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set a Hugging Face token with this fine-grained permission:

```text
Make calls to Inference Providers
```

## Sync Notes

Download notes from the public Notes repo:

```bash
python main.py sync-notes
```

Download and rebuild the vector index:

```bash
python main.py sync-notes --index
```

You can also index whatever markdown files are already in `notes/`:

```bash
python main.py index --reset
```

## Ask From The CLI

```bash
python main.py ask "What is an Ansible inventory?" --k 3
```

## Run The Streamlit App

```bash
streamlit run app.py
```

The app includes:

- chat history
- query rewriting for follow-up questions
- source chunk display
- a `k` slider for retrieved chunks
- buttons to sync notes, rebuild the index, reload the index, and clear chat

## Environment Variables

- Set `HF_TOKEN` to allow communication with Hugging Face. Only `
Make calls to Inference Providers` permission is needed for that.
- Keep `ENABLE_SIDEBAR_CONTROLS=false` to hide the sidebar in public embeds.
- Set `DEFAULT_RETRIEVAL_K` to control how many chunks are retrieved when the sidebar is hidden.

