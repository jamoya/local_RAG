# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

A local RAG (retrieval-augmented generation) system: a Flask API that also watches the active folder in-process, a Gradio UI, a thin CLI, and a Chroma vector store on disk.

- `backend/local_rag_api.py` — Flask API (default `127.0.0.1:5050`). Endpoints: `/health`, `/ready`, `/config`, `/sources`, `/folders`, `/watched_folder`, `/ingest` (multipart), `/ingest_folder`, `/delete`, `/retrieve`, `/answer`. Owns chunking, embeddings, vector store, retrieval and LLM call. Shared helpers `_ingest_documents` (chunk + upsert pre-loaded `Document`s) and `_ingest_local_file` (load a file from disk and ingest) are reused by both `/ingest` and `/ingest_folder`. `sync_folder(folder, embedding_id, version)` reconciles one folder against the store and backs both `/ingest_folder` and `/watched_folder`; it **deletes** as well as adds, so `/ingest_folder` is no longer add-only. Also owns the in-process folder watcher (see `backend/watch_manager.py`) and `_INGEST_LOCK`, a `threading.RLock` guarding `_ingest_documents`, `delete_by_source` and `get_store`'s cache fill — reentrant because `_ingest_documents` calls the other two.
- `backend/watch_manager.py` — `WatchManager`, a single repointable watchdog `Observer` living inside the API process. Callbacks are injected so it never imports `local_rag_api` (that would be a cycle). Started only under `__main__`, so the test suite never spawns one; `app.run` passes `use_reloader=False` because the Werkzeug reloader would otherwise start two.
- `backend/folder_registry.py` — owns `<CHROMA_DB_PATH>/folders.json`, shape `{active, known}`. Every function takes `chroma_path` explicitly so the module stays pure. Note `known_folders(chroma_path, fallback)` injects the fallback only when the known list is empty.
- `backend/folder_watcher.py` — standalone watchdog daemon, **no longer started by `scripts/start-mac.sh`** (the API watches in-process now). Kept for headless use, and its helpers are still imported at request time by `sync_folder`. Scans a folder, POSTs new/changed files to `/ingest`, calls `/delete` on removals. Keeps a per-`(version, embedding_id)` state file `.ingested_state__<version>__<slug>__<hash>.json` inside the watched folder. Optional periodic reconcile against `/sources`. The helpers `sync_folder` borrows are `scan_disk`, `state_file_path`, `load_state`, `save_state` and `file_fingerprint`, so server-driven ingests and the daemon share one state format.
- `frontend/launch.py` — thin CLI client. Sends `{query, llm_id}` to `/answer`.
- `frontend/gradio_app.py` — Gradio web UI that POSTs to the same API. At startup it probes `GET /config` and auto-aligns the `embedding_id` / `version` textboxes with the running backend (falls back to env vars if the API is unreachable). A **Folders panel** lists every folder with documents in the active collection (per-folder file counts, active one marked) from `GET /folders`; its dropdown accepts any absolute path, and "Use this folder" POSTs to `/watched_folder` to switch, reconcile and refresh — using the long `RAG_INGEST_TIMEOUT`, not the short answer timeout. The Sources list is scoped to the active folder via `/sources?folder=`. The "Refresh sources" button reconciles the **selected** folder via `POST /ingest_folder` before re-listing, so it now removes as well as adds. A "restrict retrieval to the selected folder" checkbox (**default on**) adds `folder` to the request body; `ask_scoped` and `chat_scoped` collapse that checkbox and the dropdown into one `folder` argument, because Gradio binds inputs positionally. It says *selected*, not *active*, on purpose: the dropdown takes typed paths and only "Use this folder" commits a switch, so the two can differ. Compare stays unscoped by design. Has a "launch-compatible" switch (**defaults to off** so the per-model retrieval preset below actually reaches the server) that, when on, sends only `{query, llm_id}` (matching `launch.py`); when off, also sends `embedding_id`, `version`, `top_k`, `fetch_k`, `search_type`, `mmr_lambda`, `max_context_chars`. Three tabs: **Ask** (single-shot), **Chat** (multi-turn with history), **Compare** (two-column side-by-side — same question runs against two independent configs in parallel via `compare_step`; each column has its own LLM/embeddings/retrieval knobs). The LLM dropdown includes the Anthropic models (`claude-sonnet-4-6`, `claude-haiku-4-5`, `claude-opus-4-7`), the OpenAI GPT-5.6 family (`openai:gpt-5.6-sol`, `openai:gpt-5.6-terra`, `openai:gpt-5.6-luna`) alongside the older GPT-4o/4.1 entries, the Gemini option, and the local generators: `ollama:qwen3:30b`, `ollama:gpt-oss:latest`, `ollama:gemma4:31b-mlx`, plus `lmstudio:google/gemma-4-12b-qat`, `lmstudio:qwen/qwen3.6-27b`, `lmstudio:mistralai/magistral-small-2509` (these need LM Studio's server running). `frontend/test_gradio_app.py` asserts every dropdown entry is a provider `make_llm` dispatches on. The default `llm_id` is `openai:gpt-5.6-luna`. Selecting a model applies a retrieval preset (`context_preset`): any `gpt-5.6` model gets `top_k=20` / `max_context_chars=50000`, everything else gets `top_k=6` / `max_context_chars=18000`. `top_k` moves with the cap because the cap alone is not binding -- `top_k` chunks of `CHUNK_SIZE` is what actually bounds the context. The preset is wired to all three model dropdowns (Ask/Chat share one, Compare has A and B) and always overwrites the sliders on model change. The preset only reaches the server while the "launch.py-compatible" switch is off (its default); turning it on sends just `{query, llm_id}` and the backend's own `TOP_K` / `MAX_CONTEXT_CHARS` env defaults apply instead.

Vector persistence: Chroma under `CHROMA_DB_PATH` (default `./local_chroma_db`). Chroma/LangChain imports are wrapped in try/except — if any are missing the API falls back to an in-memory `_TfidfStore`. Keep that dual path working when editing retrieval code.

### Provider-prefix convention

Both embeddings and LLMs use `provider:model` strings.

- `embedding_id`: `tfidf:local` (fallback, always works), `fake:<anything>` (tests), `hf:<hf-model>` (e.g. `hf:BAAI/bge-large-en-v1.5`, `hf:intfloat/e5-large-v2`), `ollama:<model>` (e.g. `ollama:qllama/bge-m3:latest`, `ollama:nomic-embed-text:v1.5` — needs a running Ollama server). `openai:` / `gemini:` prefixes are parsed but not wired to real embedding classes — `make_embeddings` raises for them.
- `llm_id`: `mock:any` (tests), `ollama:<model>`, `openai:<model>`, `gemini:<model>`, `anthropic:<model>`, `lmstudio:<model>` (OpenAI-compatible LM Studio server, base URL from `LMSTUDIO_BASE_URL`, default `http://localhost:1234/v1`) (e.g. `anthropic:claude-sonnet-4-5`). Requires the matching API key env var on the *server* process. For Anthropic, `ANTHROPIC_MAX_TOKENS` (default 4096) caps the response. When `llm_id` starts with `anthropic:`, `/answer` splits the user prompt and tags the system + retrieved-context blocks with `cache_control: {"type": "ephemeral"}` so re-running the same query against multiple Claude configs (the comparison flow) hits Anthropic's prompt cache. Per-provider cache stays warm for 5 minutes; minimum cacheable prefix is 2048 tokens (Sonnet) / 4096 tokens (Haiku/Opus).

LM Studio models must be **loaded with a context long enough for RAG prompts**. LM Studio defaults to a small context (8192) even for models supporting far more; a reasoning model can then spend its whole budget before emitting any content and `/answer` returns an empty-answer notice. Raise the context length when loading the model in LM Studio.

### Collection naming (critical invariant)

Chroma collection name is `f"{BASE_COLLECTION}__{version}__{slug(embedding_id)}"` (see `_collection_name`). This means switching `embedding_id` or `version` creates a *new* collection — vectors from different embeddings are never mixed. Folders are **not** part of the collection name: a `folder` metadata key on every chunk is what separates them, so switching folders needs no re-embedding and is reversible. Chunks written before that key existed are backfilled on first store access by `_backfill_folder_metadata`, using Chroma's metadata-merging `update()` with no embeddings argument — so the backfill costs nothing. This matters because a `where={"folder": ...}` filter silently excludes rows lacking the key rather than erroring. It pages **and writes** at `store._client.get_max_batch_size()` (5461 on this build): Chroma refuses any single `update()` above that, and since the backfill runs inside `get_store` before the cache is filled, one oversized write makes *every* store-touching endpoint (`/sources`, `/folders`, `/ingest_folder`, `/watched_folder`) return 500. Never hardcode that page size. The watcher's state filename encodes the same pair. Do not change this scheme without migrating existing data; stale collections under `./local_chroma_db/` will accumulate.

### Retrieval pipeline

`retrieve_docs` builds multiple query variants via `_build_query_variants` (regulatory-domain acronym expansions like BAT-AEL, PCDD/F, EAF; unit-token boosting; stopword-stripped keyword form), runs vector search on each, optionally unions MMR results, then re-ranks candidates with TF-IDF cosine blended 0.55 lexical / 0.45 vector. If you tune relevance, this blend and `_build_query_variants` are the main levers.

When `RERANKER_ID` (or a per-request `reranker_id`) is set to `ce:<hf-model>`, `retrieve_docs` feeds the top `fetch_k` hybrid candidates to a `sentence-transformers` CrossEncoder (`backend/reranker.py`) which makes the final `top_k` cut. Default is `none:`, which leaves retrieval byte-identical to before. The cross-encoder is cached per `reranker_id`; `ce:BAAI/bge-reranker-v2-m3` costs ~6.5 s on first load then ~0.6 s per query at `fetch_k=24`. Note that rerankers are not available through Ollama -- they need the HF model.

Chunk IDs are `sha256(f"{source_path}|{page}|{chunk}")[:24]` — deterministic, so re-ingesting the same file overwrites rather than duplicates (upsert is implemented as `delete_by_source` + re-add in `_ingest_documents`).

### Folder reconciliation (`sync_folder`, `/ingest_folder`, `/watched_folder`)

`sync_folder(folder, embedding_id, version, extensions=None)` reconciles one folder against the store using **three**
inputs, and the ordering rule matters:

- the **disk** (`scan_disk`, always `recursive=False` — top level only)
- the **store** (`_sources_in(store, folder)`) — decides whether a file is ingested *at all*
- the watcher's `.ingested_state__…json` — decides only whether an already-present file *changed*

A file is ingested when it is absent from the store, or present but with a changed mtime+size; it is skipped only when
both unchanged **and** present. Anything in the store for that folder but no longer on disk is deleted. Consulting the
store rather than trusting the state file is the point: a stale state file beside an emptied collection used to make a
sync a silent no-op.

Deletion is scoped by the `folder` metadata key, so syncing folder A can never touch folder B's documents. Note the
consequence: syncing an **empty** folder removes that folder's vectors, by design.

`POST /ingest_folder` (JSON body: `folder`, `embedding_id`, `version`, `extensions` — all optional) validates the folder
then delegates, returning `{folder, scanned, ingested: [{source_path, chunks}], deleted, skipped_unchanged, errors}`.
It is **no longer add-only**. When `folder` is omitted the server uses the registry's active folder, falling back to
`WATCHED_FOLDER` or `<CWD>/watched_folder`.

`POST /watched_folder` (body `{folder, embedding_id?, version?}`) additionally persists the choice via `folder_registry`,
updates `_ACTIVE`, and repoints the in-process watcher. 400 when the path is not a directory — `sync_folder` itself does
not validate, so that check is load-bearing.

## Commands

Running the stack (three terminals, typically):

```bash
# 0. One-shot launcher (API + Gradio UI in background, see scripts/). The API
#    watches the active folder in-process -- there is no separate watcher process.
#    UI on http://127.0.0.1:7860, API on http://127.0.0.1:5050 (the API has no
#    `/` route -- browsing port 5050 returns 404). Override the UI address with
#    GRADIO_SERVER_NAME / GRADIO_SERVER_PORT.
bash scripts/start-mac.sh
bash scripts/stop-mac.sh

# 1. Or start the API manually. Preset env-var bundles:
bash scripts/run_bge.sh           # uses hf:BAAI/bge-large-en-v1.5
bash scripts/run_e5.sh            # uses hf:intfloat/e5-large-v2
bash scripts/run_bge_m3.sh        # uses ollama:qllama/bge-m3:latest (needs ollama serve)
bash scripts/run_qwen3_emb.sh     # uses hf:Qwen/Qwen3-Embedding-0.6B
# or load all env vars from .env (run from repo root):
source scripts/run_env.sh && uv run python backend/local_rag_api.py

# 2. Optional, headless only: the standalone watcher daemon. start-mac.sh no
#    longer runs it -- the API watches in-process. If you do run it, its
#    embedding_id must match the API's or the two write to different collections.
uv run python backend/folder_watcher.py \
    --watch "$(pwd)/watched_folder" \
    --embedding-id "ollama:qllama/bge-m3:latest" \
    --version v1 \
    --reconcile-interval 300

# 3. Query via CLI or UI:
uv run python frontend/launch.py --query "…" --llm-id openai:gpt-4o-mini
uv run python frontend/gradio_app.py   # auto-aligns with backend via /config
                                       # (start-mac.sh already runs this)

# 4. Inspect server-side defaults (no embedding model is instantiated):
curl -s http://127.0.0.1:5050/config

# 5. Switch the active watched folder (validates, reconciles, repoints the watcher):
curl -s -X POST http://127.0.0.1:5050/watched_folder \
  -H 'Content-Type: application/json' \
  -d '{"folder":"'"$HOME/Documents/reports"'"}'

# 6. List folders that have documents in the active collection:
curl -s http://127.0.0.1:5050/folders

# 7. Reconcile a folder in one shot (ingests new/changed, deletes removed):
curl -s -X POST http://127.0.0.1:5050/ingest_folder \
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
- `watched_folder/`, `files_seg/`, `local_chroma_db/` and `__pycache__/` are gitignored. Don't re-add them.
- `local_chroma_db/` is purely local state — Chroma's per-collection storage, rebuilt by re-ingesting. It is not versioned, so never rely on it being present in a fresh clone.
- The API process now owns folder, `embedding_id` and `version` together (`_ACTIVE`), so the old config-drift hazard between server and watcher is gone for the default setup. It returns only if you run the standalone `folder_watcher.py` daemon by hand — then its CLI args must match the API, or it writes into one collection while `/answer` reads another.
- `scripts/start-mac.sh` runs the API with `python -u`. The watcher's ingest/removal lines are printed from the API process now, and without `-u` they sit in a block buffer instead of reaching `logs/api.log`.
- `WATCHED_FOLDER` env var (read by `local_rag_api.py`) sets the default folder used by `/ingest_folder` and surfaced in `/config`. If unset it defaults to `<CWD>/watched_folder`.
- `_watch_upsert` / `_watch_delete` read and write the same `.ingested_state__…json` that `sync_folder` uses. That is
  load-bearing twice over: watchdog emits both `created` and `modified` for a single write, and the fingerprint check is
  what stops the second one re-embedding the file; and without the write, every later sync would treat watcher-ingested
  files as new and re-embed the whole folder.
- Known limitation: `_Handler._settled_upsert` waits for a file's size to stop changing on watchdog's single dispatch
  thread, up to 4 s per file. Copying many files in at once serializes -- events queue rather than being lost, but the
  last file starts late. Moving the settle-and-ingest off the dispatch thread would fix it.
- `extensions` on `/ingest_folder` narrows what a sync **ingests and deletes**. The delete set is filtered by the same
  suffixes, so a `.pdf`-scoped sync cannot remove the folder's `.txt` vectors.
- Supported file extensions for ingest are hardcoded in `SUPPORTED_EXTS = {".pdf", ".txt", ".md", ".docx"}`. Add new types in both `backend/local_rag_api.py` (`/ingest` dispatch and `_ingest_local_file`) and `backend/folder_watcher.py` (`DEFAULT_EXTENSIONS`).
- `sync_folder` imports helpers from `folder_watcher.py` at request time. Keep the watcher module importable (no side effects at import) so this cross-module reuse stays cheap.
