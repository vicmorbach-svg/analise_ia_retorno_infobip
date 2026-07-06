FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
RUN mkdir -p /app/models

# Baixe um GGUF pequeno (exemplo)
# Substitua pela URL real do modelo escolhido
# RUN curl -L "URL_DO_MODELO_GGUF" -o /app/models/model.gguf

ENV MODEL_PATH=/app/models/model.gguf
ENV N_THREADS=4

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
