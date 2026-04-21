#!/usr/bin/env bash
# Start the API with the E5-large embedding model.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

export EMBEDDING_ID="hf:intfloat/e5-large-v2"
export CHROMA_COLLECTION_VERSION="v1"

uv run python backend/local_rag_api.py
