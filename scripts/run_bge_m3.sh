#!/usr/bin/env bash
# Start the API with the local bge-m3 embedding model served by Ollama.
# Requires: ollama serve, and `ollama pull qllama/bge-m3`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

export EMBEDDING_ID="ollama:qllama/bge-m3:latest"
export CHROMA_COLLECTION_VERSION="v1"

uv run python backend/local_rag_api.py
