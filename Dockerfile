FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Pasta para modelo (se usar llama)
RUN mkdir -p /app/models

ENV MODEL_PROVIDER=rules
ENV MODEL_PATH=/app/models/model.gguf
ENV N_THREADS=4
ENV N_CTX=4096

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
