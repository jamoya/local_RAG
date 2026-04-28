# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

A local RAG (retrieval-augmented generation) system with four cooperating processes and a Chroma vector store on disk.

- `backend/local_rag_api.py` — Flask API (default `127.0.0.1:5000`). Endpoints: `/health`, `/ready`, `/config`, `/sources`, `/ingest` (multipart), `/ingest_folder`, `/delete`, `/retrieve`, `/answer`. Owns chunking, embeddings, vector store, retrieval and LLM call. Shared helpers `_ingest_documents` (chunk + upsert pre-loaded `Document`s) and `_ingest_local_file` (load a file from disk and ingest) are reused by both `/ingest` and `/ingest_folder`.
- `backend/folder_watcher.py` — watchdog-based daemon. Scans a folder, POSTs new/changed files to `/ingest`, calls `/delete` on removals. Keeps a per-`(version, embedding_id)` state file `.ingested_state__<version>__<slug>__<hash>.json` inside the watched folder. Optional periodic reconcile against `/sources`. Its helpers (`scan_disk`, `state_file_path`, `load_state`, `save_state`, `file_fingerprint`, `should_track`) are imported directly by `/ingest_folder` so server-driven ingests and the daemon stay in sync.
- `frontend/launch.py` — thin CLI client. Sends `{query, llm_id}` to `/answer`.
- `frontend/gradio_app.py` — Gradio web UI that POSTs to the same API. At startup it probes `GET /config` and auto-aligns the `embedding_id` / `version` textboxes with the running backend (falls back to env vars if the API is unreachable). The "Refresh sources" button now calls `POST /ingest_folder` against `<repo>/watched_folder` before re-listing `/sources`, so dropping a file into the watched folder and clicking refresh ingests it. Has a "launch-compatible" switch that, when on, sends only `{query, llm_id}` (matching `launch.py`); when off, also sends `embedding_id`, `version`, `top_k`, `fetch_k`, `search_type`, `mmr_lambda`, `max_context_chars`. Three tabs: **Ask** (single-shot), **Chat** (multi-turn with history), **Compare** (two-column side-by-side — same question runs against two independent configs in parallel via `compare_step`; each column has its own LLM/embeddings/retrieval knobs). The LLM dropdown includes `anthropic:claude-sonnet-4-6`, `anthropic:claude-haiku-4-5`, and `anthropic:claude-opus-4-7` alongside the existing OpenAI/Gemini/Ollama options.

Vector persistence: Chroma under `CHROMA_DB_PATH` (default `./local_chroma_db`). Chroma/LangChain imports are wrapped in try/except — if any are missing the API falls back to an in-memory `_TfidfStore`. Keep that dual path working when editing retrieval code.

### Provider-prefix convention

Both embeddings and LLMs use `provider:model` strings.

- `embedding_id`: `tfidf:local` (fallback, always works), `fake:<anything>` (tests), `hf:<hf-model>` (e.g. `hf:BAAI/bge-large-en-v1.5`, `hf:intfloat/e5-large-v2`). `openai:` / `gemini:` / `ollama:` prefixes are parsed but not wired to real embedding classes — `make_embeddings` raises for them.
- `llm_id`: `mock:any` (tests), `ollama:<model>`, `openai:<model>`, `gemini:<model>`, `anthropic:<model>` (e.g. `anthropic:claude-sonnet-4-5`). Requires the matching API key env var on the *server* process. For Anthropic, `ANTHROPIC_MAX_TOKENS` (default 4096) caps the response. When `llm_id` starts with `anthropic:`, `/answer` splits the user prompt and tags the system + retrieved-context blocks with `cache_control: {"type": "ephemeral"}` so re-running the same query against multiple Claude configs (the comparison flow) hits Anthropic's prompt cache. Per-provider cache stays warm for 5 minutes; minimum cacheable prefix is 2048 tokens (Sonnet) / 4096 tokens (Haiku/Opus).

### Collection naming (critical invariant)

Chroma collection name is `f"{BASE_COLLECTION}__{version}__{slug(embedding_id)}"` (see `_collection_name`). This means switching `embedding_id` or `version` creates a *new* collection — vectors from different embeddings are never mixed. The watcher's state filename encodes the same pair. Do not change this scheme without migrating existing data; stale collections under `./local_chroma_db/` will accumulate.

### Retrieval pipeline

`retrieve_docs` builds multiple query variants via `_build_query_variants` (regulatory-domain acronym expansions like BAT-AEL, PCDD/F, EAF; unit-token boosting; stopword-stripped keyword form), runs vector search on each, optionally unions MMR results, then re-ranks candidates with TF-IDF cosine blended 0.55 lexical / 0.45 vector. If you tune relevance, this blend and `_build_query_variants` are the main levers.

Chunk IDs are `sha256(f"{source_path}|{page}|{chunk}")[:24]` — deterministic, so re-ingesting the same file overwrites rather than duplicates (upsert is implemented as `delete_by_source` + re-add in `_ingest_documents`).

### Folder-ingest flow (`/ingest_folder`)

`POST /ingest_folder` (JSON body: `folder`, `embedding_id`, `version`, `extensions` — all optional) scans the folder, compares each file's mtime+size against the watcher's `.ingested_state__…json`, and calls `_ingest_local_file` on new/changed files. After a successful ingest it updates the same state file so the daemon (when next started) does not re-ingest the same content. The response returns `{scanned, ingested: [{source_path, chunks}], skipped_unchanged, errors}`. When `folder` is omitted, the server uses `WATCHED_FOLDER` env var or `<CWD>/watched_folder`.

## Commands

Running the stack (three terminals, typically):

```bash
# 0. One-shot launcher (API + watcher in background, see scripts/):
bash scripts/start-mac.sh
bash scripts/stop-mac.sh

# 1. Or start the API manually. Two preset env-var bundles exist:
bash scripts/run_bge.sh           # uses hf:BAAI/bge-large-en-v1.5
bash scripts/run_e5.sh            # uses hf:intfloat/e5-large-v2
# or load all env vars from .env (run from repo root):
source scripts/run_env.sh && uv run python backend/local_rag_api.py

# 2. Start the folder watcher (embedding_id must match the API):
uv run python backend/folder_watcher.py \
    --watch "$(pwd)/watched_folder" \
    --embedding-id "hf:BAAI/bge-large-en-v1.5" \
    --version v1 \
    --reconcile-interval 300

# 3. Query via CLI or UI:
uv run python frontend/launch.py --query "…" --llm-id openai:gpt-4o-mini
uv run python frontend/gradio_app.py   # auto-aligns with backend via /config

# 4. Inspect server-side defaults (no embedding model is instantiated):
curl -s http://127.0.0.1:5000/config

# 5. Trigger a one-shot folder ingest without the daemon:
curl -s -X POST http://127.0.0.1:5000/ingest_folder \
  -H 'Content-Type: application/json' \
  -d '{"folder":"'"$(pwd)/watched_folder"'"}'
```

Tests (use `fake:any` embeddings and `mock:any` LLM — no network, no real Chroma needed):

```bash
uv run pytest backend frontend                                              # full suite
uv run pytest backend/test_local_rag_api.py::test_ingest_pdf_page_range -v  # single test
```

The tests rely on `CHROMA_DB_PATH`, `EMBEDDING_ID`, `LLM_ID`, and related env vars being set *before* `local_rag_api` is imported — `_make_app` does this with `importlib.reload`. Preserve that ordering in new tests.

## Points of attention

- `.env` is gitignored and contains live API keys. Never echo its contents, never commit it, and do not copy values into code or other files.
- `watched_folder/` and `files_seg/` are gitignored (see recent commit `c75c5d4`). Don't re-add them.
- The `local_chroma_db/` directory is *not* gitignored but some UUID subdirs appear in `git status` as untracked — leave them alone; they are Chroma's per-collection storage.
- `CHROMA_COLLECTION_VERSION` and `EMBEDDING_ID` must be consistent between the API server and the watcher, otherwise the watcher writes into one collection and `/answer` reads from another. The Gradio UI mitigates this for itself by probing `GET /config` at startup, but the daemon still relies on its CLI args / env vars.
- `WATCHED_FOLDER` env var (read by `local_rag_api.py`) sets the default folder used by `/ingest_folder` and surfaced in `/config`. If unset it defaults to `<CWD>/watched_folder`.
- Supported file extensions for ingest are hardcoded in `SUPPORTED_EXTS = {".pdf", ".txt", ".md", ".docx"}`. Add new types in both `backend/local_rag_api.py` (`/ingest` dispatch and `_ingest_local_file`) and `backend/folder_watcher.py` (`DEFAULT_EXTENSIONS`).
- `/ingest_folder` imports helpers from `folder_watcher.py` at request time. Keep the watcher module importable (no side effects at import) so this cross-module reuse stays cheap.
