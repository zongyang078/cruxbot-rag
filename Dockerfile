# The index is not baked into the image. It is 300 MB, it changes on a
# different cadence than the code, and mounting it means rebuilding the image
# does not mean re-downloading it.
FROM python:3.12-slim

WORKDIR /app

# Model weights land here at first run. Named so a volume can persist them --
# otherwise every container start re-downloads ~500 MB of bi-encoder and
# cross-encoder.
ENV HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    CRUXBOT_CHROMA_PATH=/app/data/demo/chroma \
    CRUXBOT_BM25_CACHE=/app/data/demo/bm25.pkl

# Dependencies before source, so editing a module does not invalidate the
# layer that installs torch.
COPY pyproject.toml README.md ./
RUN mkdir -p src/cruxbot && touch src/cruxbot/__init__.py \
    && pip install --no-cache-dir -e ".[rag,serve]" \
    && rm -rf /root/.cache/pip

COPY src/ ./src/
COPY benchmarks/ ./benchmarks/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if 'ok' in urllib.request.urlopen('http://localhost:8080/health').read().decode() else 1)"

CMD ["uvicorn", "cruxbot.api:app", "--host", "0.0.0.0", "--port", "8080"]
