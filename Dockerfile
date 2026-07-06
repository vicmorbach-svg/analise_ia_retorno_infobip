FROM python:3.11-slim

WORKDIR /app

# Dependências de build para compilar llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    ninja-build \
    pkg-config \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Ajuda CMake a achar compiladores
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENV LLAMA_MODEL_PATH=/data/models/model.gguf

EXPOSE 8080
CMD ["/app/start.sh"]
