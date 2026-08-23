#!/usr/bin/env bash
# Start the API with the Qwen3 embedding model (local, via sentence-transformers).
# First run downloads ~1.2 GB from Hugging Face into ~/.cache/huggingface.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

export EMBEDDING_ID="hf:Qwen/Qwen3-Embedding-0.6B"
export CHROMA_COLLECTION_VERSION="v1"

uv run python backend/local_rag_api.py
