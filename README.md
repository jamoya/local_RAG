# Local RAG

A local retrieval-augmented generation system: a Flask API owning chunking, embeddings, a Chroma vector store and the LLM call, plus a folder watcher that keeps the store in sync with a directory on disk, plus CLI and Gradio clients.

Everything runs locally by default. With no API keys configured it falls back to TF-IDF embeddings and a mock LLM, so the stack starts and answers without any external service.

## Components

| Path | Role |
| --- | --- |
| `backend/local_rag_api.py` | Flask API, default `127.0.0.1:5050`. Ingest, retrieval, answering. |
| `backend/folder_watcher.py` | Watchdog daemon. Ingests new/changed files, deletes removed ones. |
| `frontend/launch.py` | CLI client. |
| `frontend/gradio_app.py` | Web UI (default `127.0.0.1:7860`) with Ask, Chat and Compare tabs. |
| `scripts/` | Launchers and env-var presets. |

Vectors persist to `CHROMA_DB_PATH` (default `./local_chroma_db`), which is gitignored local state rebuilt by re-ingesting.

## Setup

```bash
uv venv
uv pip install -r requirements.txt
```

Copy your API keys into a `.env` at the repo root (gitignored). Only the providers you actually use need keys:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

## Running

One-shot launcher (API plus watcher in the background, logs to `logs/`):

```bash
bash scripts/start-mac.sh
bash scripts/stop-mac.sh
```

`stop-mac.sh` stops the API and watcher and, if `start-mac.sh` had to start Ollama itself, that too (tracked via `logs/ollama.pid`). An Ollama you were already running -- including the desktop app -- is left alone. LM Studio is never started or stopped by these scripts; a model loaded there keeps its memory until you unload it in LM Studio.

Or start the pieces manually:

```bash
# API, with an embedding preset
bash scripts/run_bge.sh            # hf:BAAI/bge-large-en-v1.5
bash scripts/run_e5.sh             # hf:intfloat/e5-large-v2
bash scripts/run_bge_m3.sh         # ollama:qllama/bge-m3:latest (needs ollama serve)
bash scripts/run_qwen3_emb.sh      # hf:Qwen/Qwen3-Embedding-0.6B

# Watcher -- embedding_id and version must match the API
uv run python backend/folder_watcher.py \
    --watch "$(pwd)/watched_folder" \
    --embedding-id "ollama:qllama/bge-m3:latest" \
    --version v1 \
    --reconcile-interval 300

# Clients
uv run python frontend/launch.py --query "..." --llm-id openai:gpt-4o-mini
uv run python frontend/gradio_app.py
```

The Gradio UI probes `GET /config` at startup and aligns its embedding and version fields with the running backend.

## Adding documents

Supported extensions: `.pdf`, `.txt`, `.md`, `.docx`.

Drop files into `watched_folder/`. The watcher picks them up automatically; without it, trigger a one-shot scan:

```bash
curl -s -X POST http://127.0.0.1:5050/ingest_folder \
  -H 'Content-Type: application/json' \
  -d '{"folder":"'"$(pwd)/watched_folder"'"}'
```

Chunk IDs are deterministic, so re-ingesting a file overwrites its chunks rather than duplicating them.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health`, `GET /ready` | Liveness and readiness. |
| `GET /config` | Server-side defaults. Instantiates no embedding model. |
| `GET /sources` | List ingested sources. |
| `POST /ingest` | Upload a file (multipart). |
| `POST /ingest_folder` | Scan a folder and ingest new/changed files. |
| `POST /delete` | Remove a source's chunks. |
| `POST /retrieve` | Retrieval only, no LLM. |
| `POST /answer` | Retrieval plus generation. |

```bash
curl -s -X POST http://127.0.0.1:5050/answer \
  -H 'Content-Type: application/json' \
  -d '{"query":"hello","llm_id":"mock:any"}'
```

## Model identifiers

Both embeddings and LLMs use `provider:model` strings.

- `embedding_id`: `tfidf:local` (fallback, always works), `hf:<model>` such as `hf:BAAI/bge-large-en-v1.5`, `ollama:<model>` such as `ollama:qllama/bge-m3:latest` (needs a running Ollama server), `fake:<anything>` for tests.
- `reranker_id`: `none:` (default, disabled) or `ce:<hf-model>` such as `ce:BAAI/bge-reranker-v2-m3`. Re-scores retrieved chunks with a cross-encoder before generation. Overridable per request on `/retrieve` and `/answer`.
- `llm_id`: `anthropic:<model>`, `openai:<model>`, `gemini:<model>`, `ollama:<model>`, `lmstudio:<model>` (local LM Studio server), `mock:any` for tests. The key for the chosen provider must be present on the *server* process.

LM Studio models must be **loaded with a context long enough for RAG prompts**. LM Studio defaults to a small context (8192) even for models supporting far more; a reasoning model can then spend its whole budget before emitting any content and `/answer` returns an empty-answer notice. Raise the context length when loading the model in LM Studio.

The Chroma collection name embeds the version and the embedding id, so switching either creates a separate collection and vectors from different embedding models are never mixed. Keep `EMBEDDING_ID` and `CHROMA_COLLECTION_VERSION` consistent between API and watcher, or the watcher will write where `/answer` does not read.

## Configuration

All optional; defaults shown.

| Variable | Default | Notes |
| --- | --- | --- |
| `RAG_HOST` / `RAG_PORT` | `127.0.0.1` / `5050` | API bind address. |
| `CHROMA_DB_PATH` | `./local_chroma_db` | Vector store location. |
| `CHROMA_COLLECTION_VERSION` | `v1` | Part of the collection name. |
| `EMBEDDING_ID` | `tfidf:local` | |
| `LLM_ID` | `mock:any` | |
| `WATCHED_FOLDER` | `<cwd>/watched_folder` | Default for `/ingest_folder`. |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | Used by `lmstudio:` LLMs. |
| `RERANKER_ID` | `none:` | Cross-encoder reranking, e.g. `ce:BAAI/bge-reranker-v2-m3`. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1600` / `240` | |
| `TOP_K` / `FETCH_K` | `6` / `24` | |
| `SEARCH_TYPE` | `mmr` | `mmr` or `similarity`. |
| `MMR_LAMBDA` | `0.3` | |
| `MAX_CONTEXT_CHARS` | `18000` | |

## Tests

No network and no real Chroma required; the suite uses `fake:any` embeddings and a `mock:any` LLM.

```bash
uv run pytest backend frontend
```
