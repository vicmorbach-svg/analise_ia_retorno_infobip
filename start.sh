#!/usr/bin/env sh
set -eu

mkdir -p /data/models

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}"
