FROM python:3.10-slim

RUN useradd --create-home --uid 1000 user \
    && mkdir -p /home/user/app \
    && chown user:user /home/user/app

ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

COPY requirements.txt .
# Install CPU wheels first to avoid downloading CUDA libraries.
RUN pip install --no-cache-dir 'torch>=2.2' --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

USER user

# Cache the public default model during build to reduce startup latency.
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})"

COPY --chown=user:user . .
RUN mkdir -p notes chroma_db

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
