#!/usr/bin/env bash
# Start the API with the BGE-large embedding model.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

export EMBEDDING_ID="hf:BAAI/bge-large-en-v1.5"
export CHROMA_COLLECTION_VERSION="v1"

uv run python backend/local_rag_api.py
