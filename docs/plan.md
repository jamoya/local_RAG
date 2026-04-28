2# Project Reorganization Plan

Goal: split the flat repo into `backend/`, `frontend/`, `scripts/`, `docs/`
without breaking the running stack or test suite.

## Mapping

### `backend/` (Flask API + watcher daemon + tests)

| Current path                | New path                            |
|-----------------------------|-------------------------------------|
| `local_rag_api.py`          | `backend/local_rag_api.py`          |
| `folder_watcher.py`         | `backend/folder_watcher.py`         |
| `test_local_rag_api.py`     | `backend/test_local_rag_api.py`     |

The test file already does `sys.path.insert(0, parent_of_this_file)`,
so it keeps importing `local_rag_api` cleanly when both files live in `backend/`.

### `frontend/` (clients/UI)

| Current path     | New path                  |
|------------------|---------------------------|
| `launch.py`      | `frontend/launch.py`      |
| `gradio_app.py`  | `frontend/gradio_app.py`  |

Both files only reach the API over HTTP; no Python import paths to update.

### `scripts/` (launchers and env helpers)

| Current path     | New path                |
|------------------|-------------------------|
| `run_bge.sh`     | `scripts/run_bge.sh`    |
| `run_e5.sh`      | `scripts/run_e5.sh`     |
| `run_env.sh`     | `scripts/run_env.sh`    |
| (new)            | `scripts/start-mac.sh`  |
| (new)            | `scripts/stop-mac.sh`   |

`run_bge.sh` / `run_e5.sh` currently call `python local_rag_api.py`.
After the move the command must point at `backend/local_rag_api.py`
and use `uv run` per repo convention.

### `docs/` (markdown notes and RAG outputs)

| Current path                                | New path                                            |
|---------------------------------------------|-----------------------------------------------------|
| `RAG_excange_products_industries.md`        | `docs/RAG_excange_products_industries.md`           |
| `rag_exchange.md`                           | `docs/rag_exchange.md`                              |
| `rag_exchange.json`                         | `docs/rag_exchange.json`                            |
| `RAG_exchanges/`                            | `docs/RAG_exchanges/`                               |
| (new)                                       | `docs/plan.md` (this file)                          |

### Files that stay at the repo root

- `CLAUDE.md` — Claude Code auto-loads project instructions only from the
  repository root; moving it would silently disable that behavior.
- `.env`, `.gitignore`, `requirements.txt` — standard root locations.
- `local_chroma_db/`, `chroma_db/`, `watched_folder/`, `files_seg/`,
  `uploads/` — runtime data; folder paths are referenced by env vars and
  by users from the project root.

## New scripts

### `scripts/start-mac.sh`
1. `cd` to the repo root (resolved from the script's own location).
2. Source `.env` if present, otherwise default to `EMBEDDING_ID=tfidf:local`
   so the API still boots without HF/OpenAI keys.
3. Start `backend/local_rag_api.py` in the background, log to
   `logs/api.log`, write the pid to `logs/api.pid`.
4. Wait for `GET /health` to return 200.
5. Start `backend/folder_watcher.py` against `./watched_folder`,
   log to `logs/watcher.log`, pid in `logs/watcher.pid`.
6. Print URLs the user can hit (API base + Gradio command).

### `scripts/stop-mac.sh`
1. For each pid file under `logs/*.pid`, send SIGTERM, wait briefly,
   SIGKILL if still alive, then remove the pid file.
2. As a fallback, kill any leftover `local_rag_api.py`, `folder_watcher.py`,
   or `gradio_app.py` processes started by the user.

## Validation

After the moves:
1. `uv run pytest backend/test_local_rag_api.py -q` to confirm the test
   suite still imports the API correctly from the new location.
2. `bash scripts/start-mac.sh` then a dummy `POST /answer` with
   `llm_id=mock:any` to prove the API is up and routes work without
   needing live model credentials.
3. `bash scripts/stop-mac.sh` to confirm clean shutdown.

## Out of scope

- Renaming modules or splitting `local_rag_api.py` further.
- Changing the Chroma collection naming scheme.
- Touching `.env`, secrets, or any data directory.
