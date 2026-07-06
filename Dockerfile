FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./app /app/app

RUN mkdir -p /app/models

# baixa o modelo no build (usa token se existir)
ARG HF_MODEL_URL
ARG HF_TOKEN
RUN if [ -z "$HF_MODEL_URL" ]; then echo "HF_MODEL_URL não definido" && exit 1; fi && \
    if [ -n "$HF_TOKEN" ]; then \
      curl -L -H "Authorization: Bearer $HF_TOKEN" "$HF_MODEL_URL" -o /app/models/model.gguf; \
    else \
      curl -L "$HF_MODEL_URL" -o /app/models/model.gguf; \
    fi && \
    test -s /app/models/model.gguf

ENV LLAMA_MODEL_PATH=/app/models/model.gguf

EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
