#!/usr/bin/env sh
set -eu

mkdir -p /data/models

if [ ! -s /data/models/model.gguf ]; then
  echo "Baixando GGUF (primeira vez)..."
  if [ -n "${HF_TOKEN:-}" ]; then
    curl -L -H "Authorization: Bearer $HF_TOKEN" "$HF_MODEL_URL" -o /data/models/model.gguf
  else
    curl -L "$HF_MODEL_URL" -o /data/models/model.gguf
  fi
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}"
