# Selectable watched folder

Date: 2026-08-23
Status: approved design, not yet implemented

## Problem

The watched folder is fixed at process start. `WATCHED_FOLDER` is read from the
environment into a module-level constant (`backend/local_rag_api.py:153`), and the
watcher daemon takes it as a CLI argument (`scripts/start-mac.sh:144`). Nothing can
change it at runtime.

The user wants to point the system at any local folder from the Gradio UI. On
switching, the system must verify that every file in the new folder is present in the
vector store and ingest whatever is missing; from then on, additions and removals in
that folder must be reflected in the store. The UI must also show which folders have
documents in the store, and list only the active folder's files.

Four gaps block this today:

1. `_collection_name` (`:165`) keys collections on `(version, embedding_id)` only.
   Documents from every folder land in the same collection with nothing to tell them
   apart.
2. `/ingest_folder` (`:762`) only adds. It never deletes vectors for files that have
   disappeared, and it decides what to ingest from the `.ingested_state__*.json`
   fingerprint file alone — so a stale state file beside an empty database makes it
   skip everything.
3. `/sources` (`:613`) returns every `source_path` in the collection, unfiltered.
4. The watcher is a separate process with a fixed `--watch` path and its own
   `--embedding-id` / `--version` flags, which must be kept in sync with the API by
   hand. CLAUDE.md lists this drift as a standing hazard.

## Decisions

Settled with the user before design. Each was chosen over the alternatives listed.

| Decision | Chosen | Rejected |
|---|---|---|
| Isolation | One collection, `folder` metadata tag | Collection per folder (forces full re-embed) |
| Query scope | Filtered to active folder, checkbox to widen | Always scoped; never scoped |
| Watching | Observer inside the API process | External daemon polling `/config`; sync-on-demand only |
| Recursion | Top level only | Recursive; per-folder checkbox |
| Folder picker | Dropdown of known folders, custom paths allowed | `gr.FileExplorer`; plain textbox |
| New panel | Folders that have documents in the store | Chroma directories on disk; both |
| Daemon | `start-mac.sh` stops launching it, module stays | Keep launching; delete the module |

Two further calls, made during design:

- The registry file lives next to the Chroma database, not inside each watched
  folder. It describes the store, and a watched folder may be read-only or unplugged.
- The `folder` backfill runs lazily on first store access, not from a manual
  endpoint, so no upgrade step can be forgotten.

## Verified assumptions

Checked against the installed libraries, not assumed:

- `langchain_chroma.Chroma.similarity_search_with_score` and
  `max_marginal_relevance_search` both accept `filter: dict | None`.
- `chromadb` `Collection.update(ids, metadatas)` accepts metadatas with no
  embeddings, so backfilling a metadata key costs no re-embedding.
- `_TfidfStore` (`:311`) has no filter support; the fallback path needs it added.
- `app.run(..., debug=True)` (`:1008`) runs the Werkzeug reloader, which imports the
  module twice and would start two Observers.

## Architecture

```text
Gradio UI                      Flask API process
  Folders panel  --POST /watched_folder-->  folder_registry  (folders.json)
                 --GET  /folders -------->  (counts per active collection)
  Sources panel  --GET  /sources?folder=->  sync_folder()
  Ask/Chat       --POST /answer {folder}->  retrieve_docs(folder=)
                                            watch_manager   (Observer, repointable)
                                                  |
                                            Chroma collection
                                            chunk.metadata.folder
```

One collection per `(version, embedding_id)` as today. The `folder` metadata key is
the axis everything else turns on: filtering sources, scoping retrieval, scoping
deletion, and counting files per folder.

## Components

### `folder` metadata key

`_ingest_documents` stamps each chunk with `folder = str(Path(source_path).parent)`
alongside the existing `chunk` and `id`.

`_backfill_folder_metadata(store)` brings old chunks up to date: page through
`collection.get(include=["metadatas"])`, derive `folder` from `source_path` for any
metadata lacking it, write back via `collection.update(ids, metadatas)` with no
embeddings argument. Called from `get_store()` and guarded by a module-level
`_BACKFILLED: set[str]` of collection names, so it runs once per collection per
process. It is idempotent: a run interrupted halfway is completed by the next one.

Paging matters. `/sources` already pages `collection.get()` at 10000 rows because
Chroma's SQLite backend binds one variable per row and fails past roughly 32k chunks.
The backfill uses the same page size.

### `backend/folder_registry.py`

Small module owning `<CHROMA_PATH>/folders.json`:

```json
{
  "active": "/Users/jm/docs",
  "known": ["/Users/jm/docs", "/Users/jm/PythonProjects/_WORK/Local_RAG/watched_folder"]
}
```

Functions: `load_registry()`, `save_registry(reg)`, `set_active(folder)`,
`known_folders()`, `active_folder()`. `active_folder()` falls back to the existing
`WATCHED_FOLDER` value when the file is absent, so a fresh install behaves as it does
now. Paths are stored resolved and absolute.

### `sync_folder(folder, embedding_id, version)`

Three-way reconciliation. This is the component that satisfies the "check everything
is ingested" requirement.

| Input | Answers |
| --- | --- |
| `scan_disk(folder, exts, recursive=False)` | what is on disk now |
| store, `where={"folder": folder}` | what is actually ingested |
| `.ingested_state__*.json` | mtime and size, to skip unchanged files |

Rules, per file:

- on disk, absent from the store → ingest
- on disk, in the store, fingerprint differs → re-ingest
- in the store, absent from disk → `delete_by_source`
- otherwise → skip, count as unchanged

Consulting the store rather than trusting the state file is the point: it is what
catches a populated folder whose vectors were never written, or were wiped.

Returns `{scanned, ingested: [{source_path, chunks}], deleted: [source_path],
skipped_unchanged, errors}`.

`scan_disk`, `state_file_path`, `load_state`, `save_state` and `file_fingerprint` are
imported from `folder_watcher.py` at call time, preserving the cross-module reuse
CLAUDE.md requires (and therefore the rule that the watcher module has no import-time
side effects).

The state file stays inside the watched folder under its current naming scheme, so
the standalone daemon remains compatible.

### `backend/watch_manager.py`

`WatchManager` owns one `watchdog` `Observer` and one scheduled watch.

- `start(folder, embedding_id, version)` — schedule and start
- `repoint(folder)` — `unschedule_all()` then `schedule(handler, folder, recursive=False)`
- `stop()`

The handler calls `_ingest_local_file` and `delete_by_source` directly rather than
over HTTP, and reuses the size-settle loop from `WatchHandler._ingest`
(`folder_watcher.py:136`) so large copies are not ingested mid-write.

Two consequences:

- `app.run` gains `use_reloader=False`. A background thread and the auto-reloader do
  not mix; without this the Observer starts twice.
- The observer thread writes to Chroma concurrently with request threads. A
  module-level `threading.Lock` guards three places: `_ingest_documents`,
  `delete_by_source`, and the cache-population branch of `get_store` (where two
  threads could otherwise both construct a store for the same key and one would win
  the `_STORE_CACHE` write).

## API changes

| Endpoint | Change |
| --- | --- |
| `POST /watched_folder` | New. Body `{folder, embedding_id?, version?}`. Validates the directory, sets it active in the registry, runs `sync_folder`, repoints the observer, returns the sync summary plus `active_folder`. 400 if not a directory. |
| `GET /folders` | New. Accepts optional `?embedding_id=` and `?version=` like `/sources`, defaulting to the server's, and reports against that one collection — a folder may hold documents under several embedding models, and mixing counts across collections would be misleading. Returns `{active, chroma_path, collection, folders: [{path, exists, file_count}]}` — the union of registry `known` and folders present in that collection, so a folder with zero documents in it still appears. |
| `GET /sources` | Optional `?folder=` filters by the `folder` metadata key. Omitted means all, as today. |
| `POST /answer`, `POST /retrieve` | Optional `folder` in the body, passed to `retrieve_docs`. |
| `GET /config` | Gains `active_folder`. |
| `POST /ingest_folder` | Delegates to `sync_folder`, so it now deletes as well as adds. Response gains `deleted`. |

`retrieve_docs` grows a `folder: Optional[str] = None` parameter and builds
`flt = {"folder": folder} if folder else None`, passed to `_vector_candidates` and to
`max_marginal_relevance_search`. `_vector_candidates` grows a matching parameter.

`_TfidfStore.similarity_search_with_score`, `similarity_search` and
`max_marginal_relevance_search` each gain `filter: Optional[dict] = None`, applied as
metadata equality over `self._docs` before scoring. This keeps the no-Chroma fallback
working, which CLAUDE.md requires.

## UI changes

The left column becomes two panels:

```text
Folders                             1 DB · ./local_chroma_db
[ /Users/jm/.../watched_folder   v ]
[ Use this folder ]
  * /Users/jm/.../watched_folder   12 files   ACTIVE
    /Users/jm/Documents/reports    47 files
Sources                                       connected
  Ingested 3 · deleted 1 · 8 unchanged
  PDF  CBAM Final Report.pdf
```

- Folder dropdown: `gr.Dropdown(allow_custom_value=True)` populated from `/folders`,
  value set to `active`. Any absolute path can be typed.
- "Use this folder" calls `POST /watched_folder`, then refreshes both panels. The
  sync summary renders in the existing banner style, extended with the delete count.
- The sources panel calls `/sources?folder=<active>`, so it lists only the active
  folder's files.
- A "restrict retrieval to active folder" checkbox, default on, in "Connection &
  model". `build_request_body` adds `folder` when it is ticked and
  launch-compatible is off. The existing rule holds: launch-compatible on sends only
  `{query, llm_id}`.
- `DEFAULT_WATCHED_FOLDER` remains the fallback when `/config` carries no
  `active_folder`.

The folder switch reuses the ingest timeout (`RAG_INGEST_TIMEOUT`, default 1800s),
not the answer timeout — a first sync of a large folder takes minutes.

## Error handling

- Non-existent or non-directory path → 400, message surfaced in the UI banner, active
  folder unchanged.
- Per-file ingest failures → collected in `errors` and reported in the banner; the
  rest of the sync continues. This matches how `/ingest_folder` behaves now.
- Unreadable or read-only watched folder → `save_state` raises and the request fails
  loudly. Accepted limitation, not worked around.
- API unreachable from the UI → existing offline pill, folders panel shows the last
  known state.

## Testing

Tests use `fake:any` embeddings and `mock:any` LLM, with env vars set before
`local_rag_api` is imported via `importlib.reload` in `_make_app`, as the existing
suite does.

Backend:

- state file present, store empty → files are re-ingested (the core requirement)
- file unchanged → skipped
- file deleted from disk → its vectors are removed on sync
- deletion is scoped: a file from another folder is untouched by a sync
- backfill adds `folder` to pre-existing chunks and does not call the embedder
  (assert call count on the fake)
- `/sources?folder=` filters
- `/answer` and `/retrieve` with `folder` filter retrieval
- `_TfidfStore` honours `filter`
- `/folders` shape, active marking, zero-document folder still listed
- `WatchManager.repoint` unschedules the previous directory — fake observer, no real
  filesystem timing

Frontend:

- `build_request_body` includes `folder` when the restrict checkbox is on and
  launch-compatible is off, and omits it otherwise
- folders panel HTML marks the active folder
- "Use this folder" posts to `/watched_folder` with the typed path (mocked)

## Migration

- Collection names are unchanged. Nothing is re-embedded.
- The first store access after upgrade backfills `folder` for existing chunks.
- `scripts/start-mac.sh` drops the watcher block; `scripts/stop-mac.sh` drops the
  matching teardown.
- `backend/folder_watcher.py` stays runnable standalone and importable.
- CLAUDE.md updated: the new endpoints, the `folder` metadata key, the registry file,
  the in-process watcher, and the fact that `/ingest_folder` now deletes.

## Risks

- `/ingest_folder` becomes destructive. Mitigated by scoping deletion to
  `folder == <that folder>`, so documents from other folders can never be removed by
  a sync of one folder.
- A failed backfill leaves some chunks invisible under a folder filter. Re-running is
  idempotent and happens on the next process start.
- Concurrent writes from the observer thread and request threads. Mitigated by the
  ingest lock.

## Out of scope

Recursive folder trees, multiple simultaneously active folders, per-folder embedding
models, moving vectors between collections, and any change to chunking, reranking or
the retrieval blend.
