# Selectable Watched Folder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user pick any local folder as the watched folder from the Gradio UI, with the vector store reconciled to that folder's contents and kept in sync from then on.

**Architecture:** One Chroma collection per `(version, embedding_id)` as today, with a new `folder` metadata key on every chunk as the axis for filtering sources, scoping retrieval, and scoping deletion. A registry file beside the Chroma DB holds the active folder. A `watchdog` Observer moves inside the Flask process so folder, `embedding_id` and `version` have a single owner.

**Tech Stack:** Python 3.12, Flask, LangChain (`langchain-chroma`), ChromaDB, watchdog, Gradio, pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-08-23-selectable-watched-folder-design.md`

## Global Constraints

- Package manager is `uv`. Always `uv run …`, never `python3 …`. Always `uv add …`, never `pip install …`.
- No emojis anywhere — not in code, print statements, or logging.
- Do not overengineer, do not program defensively. Exception handlers only where a failure is expected and handled.
- Sparing comments outside docstrings. Favour short modules and short functions.
- Collection naming stays `f"{BASE_COLLECTION}__{version}__{_slug(embedding_id)}"`. Do not change it.
- The watcher state file stays inside the watched folder under its current name, `.ingested_state__<version>__<slug>__<hash>.json`.
- Nothing may be re-embedded. Metadata backfill uses `collection.update(ids, metadatas)` with no embeddings argument.
- Folder handling is **top level only** — `recursive=False` everywhere.
- The no-Chroma `_TfidfStore` fallback path must keep working. Every store-facing change gets a matching fallback change.
- `backend/folder_watcher.py` must stay importable with no import-time side effects; `/ingest_folder` imports its helpers at request time.
- Tests run offline with `EMBEDDING_ID=fake:any` and `LLM_ID=mock:any`, and must set env vars **before** `local_rag_api` is imported (`_make_app` does this with `importlib.reload`).
- Full suite command: `uv run pytest backend frontend`.

## Verified Library Behaviour

These were checked against the installed versions. Do not re-derive them; do not write code that contradicts them.

- `langchain_chroma.Chroma.similarity_search_with_score(query, k, filter=None, …)` and `max_marginal_relevance_search(query, k, fetch_k, lambda_mult, filter=None, …)` both accept a `filter` dict.
- `chromadb` `Collection.update(ids=…, metadatas=…)` **merges** metadata into the existing dict rather than replacing it, and preserves embeddings and documents untouched. So the backfill passes only `{"folder": …}`, and a test **cannot** simulate legacy rows by updating with a dict that omits `folder` — it must `add()` rows without the key in the first place.
- A `where={"folder": …}` filter silently excludes rows that lack the `folder` key. No error is raised. This is exactly why the backfill exists.
- `app.run(..., debug=True)` at `backend/local_rag_api.py:1008` runs the Werkzeug reloader, which imports the module twice.

---

### Task 1: Stamp and backfill the `folder` metadata key

**Files:**
- Modify: `backend/local_rag_api.py` (`_ingest_documents` at :647, `get_store` at :374)
- Test: `backend/test_local_rag_api.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces:
  - `_folder_of(source_path: str) -> str` — resolved absolute parent directory
  - `_backfill_folder_metadata(store: Any, cname: str) -> int` — returns rows updated
  - `_BACKFILLED: set[str]` — collection names already backfilled this process
  - every chunk written by `_ingest_documents` now carries `metadata["folder"]`

- [ ] **Step 1: Write the failing test for stamping**

Add to `backend/test_local_rag_api.py`:

```python
def test_ingest_stamps_folder_metadata(client, tmp_path):
    docdir = tmp_path / "docs"
    docdir.mkdir()
    src = docdir / "a.txt"
    src.write_text("BAT-AEL for electric arc furnace dust is 5 mg per normal cubic metre.")

    r = client.post(
        "/ingest",
        data={
            "embedding_id": "fake:any",
            "version": "vtest",
            "source_path": str(src),
            "file": (io.BytesIO(src.read_bytes()), "a.txt"),
        },
        content_type="multipart/form-data",
    )
    assert r.status_code == 200, r.data

    import local_rag_api as mod
    store, _, _, _ = mod.get_store("fake:any", "vtest")
    metas = store._collection.get(include=["metadatas"])["metadatas"]
    assert metas
    assert all(m["folder"] == str(docdir.resolve()) for m in metas)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest backend/test_local_rag_api.py::test_ingest_stamps_folder_metadata -v`
Expected: FAIL with `KeyError: 'folder'`.

- [ ] **Step 3: Implement stamping**

Add near `_collection_name` (`backend/local_rag_api.py:165`):

```python
def _folder_of(source_path: str) -> str:
    """Absolute parent directory of a source file, the key retrieval is scoped by."""
    return str(Path(source_path).resolve().parent)
```

In `_ingest_documents` (`:652-657`), extend the metadata loop:

```python
    folder = _folder_of(source_path)
    chunks = split_documents(docs)
    for idx, d in enumerate(chunks):
        md = dict(d.metadata or {})
        md["chunk"] = idx
        md["folder"] = folder
        md["id"] = doc_id(source_path, md.get("page"), idx)
        d.metadata = md
```

- [ ] **Step 4: Run it to confirm it passes**

Run: `uv run pytest backend/test_local_rag_api.py::test_ingest_stamps_folder_metadata -v`
Expected: PASS.

- [ ] **Step 5: Write the failing test for the backfill**

Legacy rows are created by `add()`ing directly without a `folder` key — `update()` merges, so it cannot strip one.

```python
def test_backfill_adds_folder_without_re_embedding(tmp_path):
    app = _make_app(tmp_path)  # noqa: F841
    import local_rag_api as mod

    store, _, cname, _ = mod.get_store("fake:any", "vtest")
    coll = store._collection
    coll.add(
        ids=["legacy-1"],
        embeddings=[[0.25] * 128],
        documents=["legacy chunk written before the folder key existed"],
        metadatas=[{"source_path": "/data/reports/old.pdf", "page": 1, "chunk": 0}],
    )
    before = list(coll.get(ids=["legacy-1"], include=["embeddings"])["embeddings"][0])

    updated = mod._backfill_folder_metadata(store, cname)

    assert updated == 1
    md = coll.get(ids=["legacy-1"], include=["metadatas"])["metadatas"][0]
    assert md["folder"] == "/data/reports"
    assert md["page"] == 1
    after = list(coll.get(ids=["legacy-1"], include=["embeddings"])["embeddings"][0])
    assert after == before


def test_backfill_is_idempotent(tmp_path):
    app = _make_app(tmp_path)  # noqa: F841
    import local_rag_api as mod

    store, _, cname, _ = mod.get_store("fake:any", "vtest")
    store._collection.add(
        ids=["legacy-2"],
        embeddings=[[0.5] * 128],
        documents=["another legacy chunk"],
        metadatas=[{"source_path": "/data/reports/old2.pdf", "chunk": 0}],
    )
    assert mod._backfill_folder_metadata(store, cname) == 1
    assert mod._backfill_folder_metadata(store, cname) == 0
```

- [ ] **Step 6: Run them to confirm they fail**

Run: `uv run pytest backend/test_local_rag_api.py -k backfill -v`
Expected: FAIL with `AttributeError: module 'local_rag_api' has no attribute '_backfill_folder_metadata'`.

- [ ] **Step 7: Implement the backfill**

Add above `get_store` (`:374`). Page at 10000 for the same reason `/sources` does — Chroma's SQLite backend binds one variable per row and fails past roughly 32k.

```python
_BACKFILLED: set = set()


def _backfill_folder_metadata(store: Any, cname: str) -> int:
    """Add the folder key to chunks written before it existed. Returns rows updated.

    Chroma merges metadata on update and leaves embeddings untouched, so this
    costs no re-embedding. Rows without the key are invisible to a folder
    filter, which is why this runs before any filtered read.
    """
    coll = getattr(store, "_collection", None)
    if coll is None:
        return 0

    updated = 0
    offset, page = 0, 10000
    while True:
        got = coll.get(include=["metadatas"], limit=page, offset=offset)
        ids = got.get("ids") or []
        metas = got.get("metadatas") or []
        stale_ids, stale_metas = [], []
        for cid, md in zip(ids, metas):
            md = md or {}
            if not md.get("folder") and md.get("source_path"):
                stale_ids.append(cid)
                stale_metas.append({"folder": _folder_of(md["source_path"])})
        if stale_ids:
            coll.update(ids=stale_ids, metadatas=stale_metas)
            updated += len(stale_ids)
        if len(ids) < page:
            break
        offset += page
    _BACKFILLED.add(cname)
    return updated
```

- [ ] **Step 8: Run them to confirm they pass**

Run: `uv run pytest backend/test_local_rag_api.py -k backfill -v`
Expected: PASS (2 tests).

- [ ] **Step 9: Call the backfill from `get_store`**

Replace the Chroma branch of `get_store` (`:382-386`) so a collection is backfilled once per process on first access:

```python
    if Chroma is not None:
        emb = make_embeddings(embedding_id)
        store = Chroma(collection_name=cname, persist_directory=CHROMA_PATH, embedding_function=emb)
        _STORE_CACHE[key] = (store, emb, cname, "chroma")
        if cname not in _BACKFILLED:
            _backfill_folder_metadata(store, cname)
        return _STORE_CACHE[key]
```

- [ ] **Step 10: Run the full suite**

Run: `uv run pytest backend frontend`
Expected: all green, including the 98 pre-existing tests.

- [ ] **Step 11: Commit**

```bash
git add backend/local_rag_api.py backend/test_local_rag_api.py
git commit -m "Tag every chunk with its source folder, and backfill old ones"
```

---

### Task 2: Folder-scoped retrieval

**Files:**
- Modify: `backend/local_rag_api.py` (`_TfidfStore` at :311, `_vector_candidates` at :448, `retrieve_docs` at :480, `/retrieve` at :821, `/answer` at :932)
- Test: `backend/test_local_rag_api.py`

**Interfaces:**
- Consumes: `metadata["folder"]` from Task 1
- Produces:
  - `_vector_candidates(store, query: str, k: int, flt: Optional[Dict[str, Any]] = None)`
  - `retrieve_docs(store, query, top_k, fetch_k, search_type, mmr_lambda, reranker_id=None, folder: Optional[str] = None)`
  - `_TfidfStore.similarity_search_with_score(query, k, filter=None)`, `.similarity_search(query, k, filter=None)`, `.max_marginal_relevance_search(query, k, fetch_k, lambda_mult, filter=None)`
  - `/answer` and `/retrieve` accept an optional `folder` in the JSON body

- [ ] **Step 1: Write the failing tests**

```python
def _ingest_text(client, path: Path, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    r = client.post(
        "/ingest",
        data={
            "embedding_id": "fake:any",
            "version": "vtest",
            "source_path": str(path),
            "file": (io.BytesIO(body.encode("utf-8")), path.name),
        },
        content_type="multipart/form-data",
    )
    assert r.status_code == 200, r.data


def test_answer_folder_filter_excludes_other_folders(client, tmp_path):
    a = tmp_path / "alpha" / "a.txt"
    b = tmp_path / "beta" / "b.txt"
    _ingest_text(client, a, "The alpha limit for dust is 5 mg per normal cubic metre.")
    _ingest_text(client, b, "The beta limit for dust is 9 mg per normal cubic metre.")

    r = client.post("/retrieve", json={
        "query": "limit for dust",
        "embedding_id": "fake:any",
        "version": "vtest",
        "folder": str((tmp_path / "alpha").resolve()),
    })
    assert r.status_code == 200, r.data
    names = {s["source_path"] for s in r.get_json()["sources"]}
    assert names == {str(a.resolve())}


def test_retrieve_without_folder_spans_all_folders(client, tmp_path):
    a = tmp_path / "alpha" / "a.txt"
    b = tmp_path / "beta" / "b.txt"
    _ingest_text(client, a, "The alpha limit for dust is 5 mg per normal cubic metre.")
    _ingest_text(client, b, "The beta limit for dust is 9 mg per normal cubic metre.")

    r = client.post("/retrieve", json={
        "query": "limit for dust", "embedding_id": "fake:any", "version": "vtest",
    })
    names = {s["source_path"] for s in r.get_json()["sources"]}
    assert names == {str(a.resolve()), str(b.resolve())}


def test_tfidf_fallback_store_honours_filter(tmp_path):
    app = _make_app(tmp_path)  # noqa: F841
    import local_rag_api as mod
    from langchain_core.documents import Document

    store = mod._TfidfStore()
    store.add_documents([
        Document(page_content="alpha dust limit", metadata={"folder": "/alpha", "source_path": "/alpha/a.txt"}),
        Document(page_content="beta dust limit", metadata={"folder": "/beta", "source_path": "/beta/b.txt"}),
    ])

    hits = store.similarity_search("dust limit", k=5, filter={"folder": "/alpha"})
    assert [d.metadata["source_path"] for d in hits] == ["/alpha/a.txt"]

    mmr = store.max_marginal_relevance_search("dust limit", k=5, fetch_k=5, lambda_mult=0.3, filter={"folder": "/beta"})
    assert [d.metadata["source_path"] for d in mmr] == ["/beta/b.txt"]
```

Check the `Document` import path used at the top of `backend/local_rag_api.py` and mirror it in the test rather than assuming.

- [ ] **Step 2: Run them to confirm they fail**

Run: `uv run pytest backend/test_local_rag_api.py -k "folder_filter or spans_all_folders or fallback_store_honours" -v`
Expected: FAIL — folder is ignored, so both folders come back; the `_TfidfStore` test fails with `TypeError: … got an unexpected keyword argument 'filter'`.

- [ ] **Step 3: Add filter support to `_TfidfStore`**

Add a helper above the class and thread `filter` through its three search methods. Restrict the candidate index set before scoring:

```python
def _md_matches(md: Optional[Dict[str, Any]], flt: Optional[Dict[str, Any]]) -> bool:
    if not flt:
        return True
    md = md or {}
    return all(md.get(k) == v for k, v in flt.items())
```

In `_TfidfStore`, change the three signatures and gate the candidate rows:

```python
    def similarity_search_with_score(self, query: str, k: int, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        if not self._docs or self._X is None:
            return []
        qv = self._vectorizer.transform([query])
        sims = cosine_similarity(qv, self._X).ravel()
        allowed = [i for i, d in enumerate(self._docs) if _md_matches(d.metadata, filter)]
        allowed.sort(key=lambda i: sims[i], reverse=True)
        return [(self._docs[i], float(1.0 - sims[i])) for i in allowed[:k]]

    def similarity_search(self, query: str, k: int, filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        return [d for d, _ in self.similarity_search_with_score(query, k, filter)]
```

For `max_marginal_relevance_search`, add `filter: Optional[Dict[str, Any]] = None` to the signature and replace the candidate line

```python
        candidates = sims.argsort()[::-1][:fetch_k].tolist()
```

with

```python
        allowed = [i for i, d in enumerate(self._docs) if _md_matches(d.metadata, filter)]
        candidates = sorted(allowed, key=lambda i: sims[i], reverse=True)[:fetch_k]
```

Leave the rest of the MMR loop as it is.

- [ ] **Step 4: Thread the filter through retrieval**

`_vector_candidates` (`:448`):

```python
def _vector_candidates(store: Any, query: str, k: int, flt: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
    if hasattr(store, "similarity_search_with_score"):
        try:
            return store.similarity_search_with_score(query, k=k, filter=flt)
        except Exception:
            pass
    if hasattr(store, "similarity_search"):
        docs = store.similarity_search(query, k=k, filter=flt)
        return [(d, 1.0) for d in docs]
    raise RuntimeError("Store does not support retrieval")
```

`retrieve_docs` (`:480`) gains the parameter and builds the filter once:

```python
def retrieve_docs(store: Any, query: str, top_k: int, fetch_k: int, search_type: str, mmr_lambda: float,
                  reranker_id: Optional[str] = None, folder: Optional[str] = None) -> List[Document]:
    flt = {"folder": folder} if folder else None
    variants = _build_query_variants(query)
    cand: List[Tuple[Document, float]] = []
    per_q = max(5, min(fetch_k, 40))
    for v in variants[:4]:
        cand.extend(_vector_candidates(store, v, k=per_q, flt=flt))
```

and passes it to MMR (`:497`):

```python
            mmr_docs = store.max_marginal_relevance_search(query, k=top_k, fetch_k=max(fetch_k, top_k*4), lambda_mult=mmr_lambda, filter=flt)
```

The `except Exception: pass` around the MMR call already exists — leave it.

- [ ] **Step 5: Accept `folder` on both routes**

In `/retrieve` (`:821`) and `/answer` (`:932`), read it next to the other knobs:

```python
    folder = payload.get("folder") or None
```

and pass `folder=folder` into the `retrieve_docs(...)` call in each route.

- [ ] **Step 6: Run the tests to confirm they pass**

Run: `uv run pytest backend/test_local_rag_api.py -k "folder_filter or spans_all_folders or fallback_store_honours" -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest backend frontend`
Expected: all green. The retrieval blend is untouched when `folder` is absent, so the existing relevance tests must not move.

- [ ] **Step 8: Commit**

```bash
git add backend/local_rag_api.py backend/test_local_rag_api.py
git commit -m "Scope retrieval to a folder via a metadata filter"
```

---

### Task 3: Folder-scoped `/sources`

**Files:**
- Modify: `backend/local_rag_api.py` (`/sources` at :613)
- Test: `backend/test_local_rag_api.py`

**Interfaces:**
- Consumes: `metadata["folder"]` from Task 1
- Produces: `_sources_in(store: Any, folder: Optional[str] = None) -> List[str]` — sorted distinct source paths, optionally restricted to one folder. **Task 5 depends on this function.** `/sources` accepts `?folder=`.

- [ ] **Step 1: Write the failing test**

```python
def test_sources_filters_by_folder(client, tmp_path):
    a = tmp_path / "alpha" / "a.txt"
    b = tmp_path / "beta" / "b.txt"
    _ingest_text(client, a, "alpha content about dust limits")
    _ingest_text(client, b, "beta content about dust limits")

    r = client.get(f"/sources?embedding_id=fake:any&version=vtest&folder={(tmp_path / 'alpha').resolve()}")
    assert r.status_code == 200, r.data
    assert r.get_json()["sources"] == [str(a.resolve())]

    r = client.get("/sources?embedding_id=fake:any&version=vtest")
    assert set(r.get_json()["sources"]) == {str(a.resolve()), str(b.resolve())}
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest backend/test_local_rag_api.py::test_sources_filters_by_folder -v`
Expected: FAIL — both paths come back from the filtered call.

- [ ] **Step 3: Extract `_sources_in` and use it in the route**

Replace the body of `sources()` (`:613-633`). The paging behaviour and the fallback branch move into the helper unchanged; only the `where` argument is new.

```python
def _sources_in(store: Any, folder: Optional[str] = None) -> List[str]:
    """Distinct source paths in the store, optionally restricted to one folder."""
    where = {"folder": folder} if folder else None
    if hasattr(store, "_collection"):
        # Chroma's SQLite backend binds one variable per row, so a single large
        # get() raises "too many SQL variables" past ~32k chunks. Page instead.
        found = set()
        offset, page = 0, 10000
        while True:
            got = store._collection.get(include=["metadatas"], limit=page, offset=offset, where=where)
            metas = got.get("metadatas") or []
            found.update((m or {}).get("source_path") for m in metas if (m or {}).get("source_path"))
            if len(metas) < page:
                break
            offset += page
        return sorted(found)
    docs = [d for d in getattr(store, "_docs", []) if _md_matches(d.metadata, where)]
    return sorted({(d.metadata or {}).get("source_path") for d in docs if (d.metadata or {}).get("source_path")})


@app.get("/sources")
def sources():
    embedding_id = request.args.get("embedding_id", DEFAULT_EMBEDDING_ID)
    version = request.args.get("version", DEFAULT_VERSION)
    folder = request.args.get("folder") or None
    store, _, cname, backend = get_store(embedding_id, version)
    srcs = _sources_in(store, folder)
    return jsonify({"sources": srcs, "collection": cname, "backend": backend,
                    "embedding_id": embedding_id, "version": version, "folder": folder})
```

- [ ] **Step 4: Run it to confirm it passes**

Run: `uv run pytest backend/test_local_rag_api.py::test_sources_filters_by_folder -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest backend frontend`
Expected: all green. `test_sources_pages_large_collections` (`backend/test_local_rag_api.py:265`) covers the paging path that just moved — it must still pass. If its fake collection's `get()` does not accept a `where` keyword, add `where=None` to that fake's signature in the same commit.

- [ ] **Step 6: Commit**

```bash
git add backend/local_rag_api.py backend/test_local_rag_api.py
git commit -m "Filter /sources by folder, extracting a reusable _sources_in"
```

---

### Task 4: Folder registry

**Files:**
- Create: `backend/folder_registry.py`
- Create: `backend/test_folder_registry.py`
- Modify: `backend/local_rag_api.py` (`/config` at :748)

**Interfaces:**
- Consumes: nothing
- Produces, all taking `chroma_path` explicitly so the module stays pure and importable without env setup:
  - `registry_path(chroma_path: str) -> Path`
  - `load_registry(chroma_path: str) -> Dict[str, Any]` — always `{"active": Optional[str], "known": List[str]}`
  - `save_registry(chroma_path: str, reg: Dict[str, Any]) -> None`
  - `active_folder(chroma_path: str, fallback: str) -> str`
  - `known_folders(chroma_path: str, fallback: str) -> List[str]`
  - `set_active(chroma_path: str, folder: str) -> Dict[str, Any]` — resolves, adds to `known`, returns the registry
  - `/config` gains `active_folder`

- [ ] **Step 1: Write the failing tests**

Create `backend/test_folder_registry.py`:

```python
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))

from pathlib import Path

import folder_registry as fr


def test_active_falls_back_when_no_file(tmp_path):
    assert fr.active_folder(str(tmp_path), "/default/watched") == "/default/watched"
    assert fr.known_folders(str(tmp_path), "/default/watched") == ["/default/watched"]


def test_set_active_persists_and_accumulates_known(tmp_path):
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()

    fr.set_active(str(tmp_path), str(one))
    assert fr.active_folder(str(tmp_path), "/unused") == str(one.resolve())

    fr.set_active(str(tmp_path), str(two))
    assert fr.active_folder(str(tmp_path), "/unused") == str(two.resolve())
    assert set(fr.known_folders(str(tmp_path), "/unused")) == {str(one.resolve()), str(two.resolve())}


def test_set_active_twice_does_not_duplicate_known(tmp_path):
    one = tmp_path / "one"
    one.mkdir()
    fr.set_active(str(tmp_path), str(one))
    fr.set_active(str(tmp_path), str(one))
    assert fr.known_folders(str(tmp_path), "/unused") == [str(one.resolve())]


def test_corrupt_registry_falls_back(tmp_path):
    fr.registry_path(str(tmp_path)).write_text("{not json")
    assert fr.active_folder(str(tmp_path), "/default/watched") == "/default/watched"
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `uv run pytest backend/test_folder_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'folder_registry'`.

- [ ] **Step 3: Write the module**

Create `backend/folder_registry.py`. `load_registry` tolerates a missing or corrupt file the way `folder_watcher.load_state` does — that is a genuinely expected failure, not defensive padding.

```python
"""Which local folder is the active watched folder, and which ones we have seen.

Stored beside the Chroma database rather than inside a watched folder: it
describes the store, and a watched folder may be read-only or unplugged.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

REGISTRY_NAME = "folders.json"


def registry_path(chroma_path: str) -> Path:
    p = Path(chroma_path)
    p.mkdir(parents=True, exist_ok=True)
    return p / REGISTRY_NAME


def load_registry(chroma_path: str) -> Dict[str, Any]:
    path = registry_path(chroma_path)
    default: Dict[str, Any] = {"active": None, "known": []}
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text() or "{}")
    except json.JSONDecodeError:
        return default
    if not isinstance(data, dict):
        return default
    active = data.get("active")
    known = data.get("known")
    return {
        "active": active if isinstance(active, str) else None,
        "known": [k for k in known if isinstance(k, str)] if isinstance(known, list) else [],
    }


def save_registry(chroma_path: str, reg: Dict[str, Any]) -> None:
    registry_path(chroma_path).write_text(json.dumps(reg, indent=2, sort_keys=True))


def active_folder(chroma_path: str, fallback: str) -> str:
    return load_registry(chroma_path)["active"] or fallback


def known_folders(chroma_path: str, fallback: str) -> List[str]:
    known = list(load_registry(chroma_path)["known"])
    if fallback and fallback not in known:
        known.append(fallback)
    return sorted(known)


def set_active(chroma_path: str, folder: str) -> Dict[str, Any]:
    resolved = str(Path(folder).expanduser().resolve())
    reg = load_registry(chroma_path)
    reg["active"] = resolved
    if resolved not in reg["known"]:
        reg["known"].append(resolved)
    save_registry(chroma_path, reg)
    return reg
```

- [ ] **Step 4: Run them to confirm they pass**

Run: `uv run pytest backend/test_folder_registry.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Write the failing test for `/config`**

```python
def test_config_reports_active_folder(client, tmp_path):
    js = client.get("/config").get_json()
    assert js["active_folder"] == js["watched_folder"]
```

- [ ] **Step 6: Run it to confirm it fails**

Run: `uv run pytest backend/test_local_rag_api.py::test_config_reports_active_folder -v`
Expected: FAIL with `KeyError: 'active_folder'`.

- [ ] **Step 7: Surface it in `/config`**

Add the import beside the other backend imports in `backend/local_rag_api.py`:

```python
import folder_registry
```

and add one line to the `/config` payload (`:751`):

```python
        "active_folder": folder_registry.active_folder(CHROMA_PATH, WATCHED_FOLDER),
```

- [ ] **Step 8: Run it, then the full suite**

Run: `uv run pytest backend/test_local_rag_api.py::test_config_reports_active_folder -v`
Expected: PASS.
Run: `uv run pytest backend frontend`
Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add backend/folder_registry.py backend/test_folder_registry.py backend/local_rag_api.py backend/test_local_rag_api.py
git commit -m "Add a folder registry beside the Chroma DB and expose the active folder"
```

---

### Task 5: `sync_folder` — three-way reconciliation

**Files:**
- Modify: `backend/local_rag_api.py` (`/ingest_folder` at :761)
- Test: `backend/test_local_rag_api.py`

**Interfaces:**
- Consumes: `_sources_in` (Task 3), `_folder_of` (Task 1), `delete_by_source` (:393), `_ingest_local_file` (:681)
- Produces: `sync_folder(folder: str, embedding_id: str, version: str, extensions: Optional[List[str]] = None) -> Dict[str, Any]` returning `{"folder", "scanned", "ingested": [{"source_path", "chunks"}], "deleted": [str], "skipped_unchanged": int, "errors": [{"source_path", "error"}]}`. `/ingest_folder` delegates to it and its response gains `deleted`.

- [ ] **Step 1: Write the failing tests**

The first is the core requirement: a state file that claims everything is ingested must not stop a sync when the store is actually empty.

```python
def test_sync_ingests_when_state_file_lies_about_an_empty_store(tmp_path):
    app = _make_app(tmp_path)
    import local_rag_api as mod
    from folder_watcher import state_file_path, save_state, file_fingerprint

    folder = tmp_path / "docs"
    folder.mkdir()
    f = folder / "a.txt"
    f.write_text("dust limit is 5 mg per normal cubic metre")

    # A stale state file: fingerprints recorded, but nothing was ever stored.
    sp = state_file_path(folder, "vtest", "fake:any")
    save_state(sp, {"files": {str(f.resolve()): file_fingerprint(f)}})

    res = mod.sync_folder(str(folder), "fake:any", "vtest")

    assert [i["source_path"] for i in res["ingested"]] == [str(f.resolve())]
    assert res["skipped_unchanged"] == 0
    store, _, _, _ = mod.get_store("fake:any", "vtest")
    assert mod._sources_in(store, str(folder.resolve())) == [str(f.resolve())]


def test_sync_skips_unchanged_on_second_run(tmp_path):
    app = _make_app(tmp_path)
    import local_rag_api as mod

    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("dust limit is 5 mg per normal cubic metre")

    first = mod.sync_folder(str(folder), "fake:any", "vtest")
    second = mod.sync_folder(str(folder), "fake:any", "vtest")

    assert len(first["ingested"]) == 1
    assert second["ingested"] == []
    assert second["skipped_unchanged"] == 1


def test_sync_deletes_vectors_for_files_removed_from_disk(tmp_path):
    app = _make_app(tmp_path)
    import local_rag_api as mod

    folder = tmp_path / "docs"
    folder.mkdir()
    keep = folder / "keep.txt"
    gone = folder / "gone.txt"
    keep.write_text("keep this content about dust")
    gone.write_text("delete this content about dust")
    mod.sync_folder(str(folder), "fake:any", "vtest")

    gone.unlink()
    res = mod.sync_folder(str(folder), "fake:any", "vtest")

    assert res["deleted"] == [str(gone.resolve())]
    store, _, _, _ = mod.get_store("fake:any", "vtest")
    assert mod._sources_in(store, str(folder.resolve())) == [str(keep.resolve())]


def test_sync_never_touches_another_folders_documents(tmp_path):
    app = _make_app(tmp_path)
    import local_rag_api as mod

    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    (alpha / "a.txt").write_text("alpha content about dust")
    (beta / "b.txt").write_text("beta content about dust")
    mod.sync_folder(str(alpha), "fake:any", "vtest")
    mod.sync_folder(str(beta), "fake:any", "vtest")

    # Empty alpha entirely, then sync it. Beta must survive untouched.
    (alpha / "a.txt").unlink()
    res = mod.sync_folder(str(alpha), "fake:any", "vtest")

    assert res["deleted"] == [str((alpha / "a.txt").resolve())]
    store, _, _, _ = mod.get_store("fake:any", "vtest")
    assert mod._sources_in(store, str(beta.resolve())) == [str((beta / "b.txt").resolve())]


def test_ingest_folder_route_reports_deletions(client, tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    doomed = folder / "doomed.txt"
    doomed.write_text("content about dust that will be removed")

    r = client.post("/ingest_folder", json={"folder": str(folder), "embedding_id": "fake:any", "version": "vtest"})
    assert r.status_code == 200, r.data
    assert len(r.get_json()["ingested"]) == 1

    doomed.unlink()
    r = client.post("/ingest_folder", json={"folder": str(folder), "embedding_id": "fake:any", "version": "vtest"})
    assert r.get_json()["deleted"] == [str(doomed.resolve())]
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `uv run pytest backend/test_local_rag_api.py -k sync -v`
Expected: FAIL with `AttributeError: module 'local_rag_api' has no attribute 'sync_folder'`.

- [ ] **Step 3: Implement `sync_folder`**

Add above the `/ingest_folder` route. Note it consults **both** the store and the state file: the store decides whether a file is present at all, the fingerprint only decides whether a present file changed.

```python
def sync_folder(folder: str, embedding_id: str, version: str,
                extensions: Optional[List[str]] = None) -> Dict[str, Any]:
    """Reconcile one folder against the store: ingest what is missing or changed,
    delete vectors for files that are gone.

    Three inputs: the disk, the store, and the watcher's fingerprint state file.
    The store is what decides whether a file is ingested at all -- a stale state
    file beside an emptied collection must not make a sync a no-op.
    """
    from folder_watcher import (
        scan_disk, state_file_path, load_state, save_state, file_fingerprint,
    )

    folder_path = Path(folder).expanduser().resolve()
    exts = extensions or sorted(SUPPORTED_EXTS)

    state_path = state_file_path(folder_path, version, embedding_id)
    state = load_state(state_path)
    files_state = state.setdefault("files", {})

    disk = scan_disk(folder_path, exts, recursive=False)
    store, _, _, _ = get_store(embedding_id, version)
    indexed = set(_sources_in(store, str(folder_path)))

    ingested: List[Dict[str, Any]] = []
    deleted: List[str] = []
    skipped_unchanged = 0
    errors: List[Dict[str, str]] = []

    for path_str in sorted(disk):
        p = Path(path_str)
        try:
            fp = file_fingerprint(p)
        except FileNotFoundError:
            continue
        prev = files_state.get(path_str)
        unchanged = prev and prev.get("mtime") == fp["mtime"] and prev.get("size") == fp["size"]
        if unchanged and path_str in indexed:
            skipped_unchanged += 1
            continue
        try:
            r = _ingest_local_file(path_str, embedding_id, version)
            ingested.append({"source_path": r["source_path"], "chunks": r["chunks_added"]})
            files_state[path_str] = fp
            save_state(state_path, state)
            print(f"file {p.name} added to the database")
        except Exception as e:
            errors.append({"source_path": path_str, "error": f"{type(e).__name__}: {e}"})

    for path_str in sorted(indexed - disk):
        delete_by_source(store, path_str)
        files_state.pop(path_str, None)
        deleted.append(path_str)
        print(f"file {Path(path_str).name} removed from the database")
    if deleted:
        save_state(state_path, state)

    return {
        "folder": str(folder_path),
        "scanned": len(disk),
        "ingested": ingested,
        "deleted": deleted,
        "skipped_unchanged": skipped_unchanged,
        "errors": errors,
    }
```

- [ ] **Step 4: Make `/ingest_folder` delegate**

Replace the body of `ingest_folder()` (`:762-819`) with:

```python
@app.post("/ingest_folder")
def ingest_folder():
    """Scan a folder and reconcile it with the store: ingest new/changed, delete removed."""
    payload = request.get_json(silent=True) or {}
    folder = payload.get("folder") or folder_registry.active_folder(CHROMA_PATH, WATCHED_FOLDER)
    embedding_id = payload.get("embedding_id", DEFAULT_EMBEDDING_ID)
    version = payload.get("version", DEFAULT_VERSION)
    extensions = payload.get("extensions") or sorted(SUPPORTED_EXTS)

    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.is_dir():
        return jsonify({"error": f"folder not found or not a directory: {folder}"}), 400

    res = sync_folder(str(folder_path), embedding_id, version, extensions)
    _, _, cname, backend = get_store(embedding_id, version)
    return jsonify({
        "status": "ok",
        "embedding_id": embedding_id,
        "version": version,
        "collection": cname,
        "backend": backend,
        **res,
    })
```

- [ ] **Step 5: Run the tests to confirm they pass**

Run: `uv run pytest backend/test_local_rag_api.py -k "sync or ingest_folder" -v`
Expected: PASS (5 new tests plus the existing `/ingest_folder` tests).

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest backend frontend`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add backend/local_rag_api.py backend/test_local_rag_api.py
git commit -m "Reconcile folders against the store, not just the fingerprint file"
```

---

### Task 6: `GET /folders`

**Files:**
- Modify: `backend/local_rag_api.py`
- Test: `backend/test_local_rag_api.py`

**Interfaces:**
- Consumes: `_sources_in` (Task 3), `folder_registry` (Task 4)
- Produces: `GET /folders` returning `{"active": str, "chroma_path": str, "collection": str, "embedding_id": str, "version": str, "folders": [{"path": str, "exists": bool, "file_count": int}]}`, sorted by path. Accepts optional `?embedding_id=` and `?version=`.

- [ ] **Step 1: Write the failing test**

The active folder is set through `folder_registry` directly rather than through
`POST /watched_folder`, which does not exist until Task 7. This task must stand on
its own.

```python
def test_folders_lists_counts_and_marks_active(client, tmp_path):
    import local_rag_api as mod
    import folder_registry as fr

    alpha = tmp_path / "alpha"
    _ingest_text(client, alpha / "a.txt", "alpha content about dust limits")
    _ingest_text(client, alpha / "a2.txt", "more alpha content about dust limits")
    _ingest_text(client, tmp_path / "beta" / "b.txt", "beta content about dust limits")
    fr.set_active(mod.CHROMA_PATH, str(alpha))

    js = client.get("/folders?embedding_id=fake:any&version=vtest").get_json()
    assert js["active"] == str(alpha.resolve())
    by_path = {f["path"]: f for f in js["folders"]}
    assert by_path[str(alpha.resolve())]["file_count"] == 2
    assert by_path[str((tmp_path / "beta").resolve())]["file_count"] == 1
    assert by_path[str(alpha.resolve())]["exists"] is True


def test_folders_includes_a_known_folder_with_no_documents(client, tmp_path):
    import local_rag_api as mod
    import folder_registry as fr

    empty = tmp_path / "empty"
    empty.mkdir()
    fr.set_active(mod.CHROMA_PATH, str(empty))

    js = client.get("/folders?embedding_id=fake:any&version=vtest").get_json()
    by_path = {f["path"]: f for f in js["folders"]}
    assert by_path[str(empty.resolve())]["file_count"] == 0
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `uv run pytest backend/test_local_rag_api.py -k folders_lists -v`
Expected: FAIL with 404 on `/folders`.

- [ ] **Step 3: Implement the route**

```python
@app.get("/folders")
def folders():
    """Folders that have documents in the active collection, plus every known folder.

    Counts are per collection: the same folder may hold documents under several
    embedding models, and mixing counts across collections would mislead.
    """
    embedding_id = request.args.get("embedding_id", DEFAULT_EMBEDDING_ID)
    version = request.args.get("version", DEFAULT_VERSION)
    store, _, cname, _ = get_store(embedding_id, version)

    counts: Dict[str, int] = {}
    for src in _sources_in(store):
        counts[_folder_of(src)] = counts.get(_folder_of(src), 0) + 1

    active = folder_registry.active_folder(CHROMA_PATH, WATCHED_FOLDER)
    paths = set(counts) | set(folder_registry.known_folders(CHROMA_PATH, WATCHED_FOLDER)) | {active}

    return jsonify({
        "active": active,
        "chroma_path": CHROMA_PATH,
        "collection": cname,
        "embedding_id": embedding_id,
        "version": version,
        "folders": [
            {"path": p, "exists": Path(p).is_dir(), "file_count": counts.get(p, 0)}
            for p in sorted(paths)
        ],
    })
```

- [ ] **Step 4: Run them to confirm they pass**

Run: `uv run pytest backend/test_local_rag_api.py -k folders_ -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full suite and commit**

Run: `uv run pytest backend frontend`
Expected: all green.

```bash
git add backend/local_rag_api.py backend/test_local_rag_api.py
git commit -m "Add GET /folders with per-folder document counts"
```

---

### Task 7: `POST /watched_folder`

**Files:**
- Modify: `backend/local_rag_api.py`
- Test: `backend/test_local_rag_api.py`

**Interfaces:**
- Consumes: `sync_folder` (Task 5), `folder_registry.set_active` (Task 4)
- Produces:
  - `_ACTIVE: Dict[str, str]` — module-level `{"folder", "embedding_id", "version"}`, read by the watcher callbacks in Task 8
  - `POST /watched_folder` with body `{folder, embedding_id?, version?}` returning `{"status": "ok", "active_folder": str, **sync_result}`; 400 when the path is not a directory

- [ ] **Step 1: Write the failing tests**

```python
def test_watched_folder_sets_active_and_syncs(client, tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("dust limit is 5 mg per normal cubic metre")

    r = client.post("/watched_folder", json={"folder": str(folder), "embedding_id": "fake:any", "version": "vtest"})
    assert r.status_code == 200, r.data
    js = r.get_json()
    assert js["active_folder"] == str(folder.resolve())
    assert len(js["ingested"]) == 1
    assert client.get("/config").get_json()["active_folder"] == str(folder.resolve())


def test_watched_folder_rejects_a_non_directory(client, tmp_path):
    missing = tmp_path / "nope"
    r = client.post("/watched_folder", json={"folder": str(missing)})
    assert r.status_code == 400
    assert "not a directory" in r.get_json()["error"]


def test_watched_folder_rejects_a_file_path(client, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("not a folder")
    r = client.post("/watched_folder", json={"folder": str(f)})
    assert r.status_code == 400
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `uv run pytest backend/test_local_rag_api.py -k watched_folder -v`
Expected: FAIL with 404.

- [ ] **Step 3: Implement the route**

Add `_ACTIVE` next to the other module state, initialised from the registry:

```python
_ACTIVE: Dict[str, str] = {
    "folder": folder_registry.active_folder(CHROMA_PATH, WATCHED_FOLDER),
    "embedding_id": DEFAULT_EMBEDDING_ID,
    "version": DEFAULT_VERSION,
}
```

and the route:

```python
@app.post("/watched_folder")
def watched_folder():
    """Make a folder the active one: validate, persist, reconcile, repoint the watcher."""
    payload = request.get_json(silent=True) or {}
    folder = payload.get("folder")
    if not folder:
        return jsonify({"error": "folder is required"}), 400

    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.is_dir():
        return jsonify({"error": f"folder not found or not a directory: {folder}"}), 400

    embedding_id = payload.get("embedding_id", DEFAULT_EMBEDDING_ID)
    version = payload.get("version", DEFAULT_VERSION)

    folder_registry.set_active(CHROMA_PATH, str(folder_path))
    _ACTIVE.update({"folder": str(folder_path), "embedding_id": embedding_id, "version": version})

    res = sync_folder(str(folder_path), embedding_id, version)
    return jsonify({"status": "ok", "active_folder": str(folder_path),
                    "embedding_id": embedding_id, "version": version, **res})
```

The watcher repoint call is added in Task 8 — leave it out for now rather than referencing a module that does not exist yet.

- [ ] **Step 4: Run them to confirm they pass**

Run: `uv run pytest backend/test_local_rag_api.py -k watched_folder -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full suite and commit**

Run: `uv run pytest backend frontend`
Expected: all green — including the Task 6 `/folders` tests, which need this route.

```bash
git add backend/local_rag_api.py backend/test_local_rag_api.py
git commit -m "Add POST /watched_folder to switch and reconcile the active folder"
```

---

### Task 8: In-process watcher

**Files:**
- Create: `backend/watch_manager.py`
- Create: `backend/test_watch_manager.py`
- Modify: `backend/local_rag_api.py` (`_ingest_documents`, `delete_by_source`, `get_store`, `/watched_folder`, the `__main__` block at :1006)

**Interfaces:**
- Consumes: `_ACTIVE` (Task 7), `_ingest_local_file` (:681), `delete_by_source` (:393)
- Produces:
  - `WatchManager(on_upsert: Callable[[str], None], on_delete: Callable[[str], None], extensions: List[str], observer_factory: Callable[[], Any] = Observer)`
  - `.start(folder: str) -> None`, `.repoint(folder: str) -> None`, `.stop() -> None`, `.folder` attribute
  - `_INGEST_LOCK: threading.Lock` in `local_rag_api`

Callbacks are injected so this module never imports `local_rag_api` — that would be a cycle.

- [ ] **Step 1: Write the failing tests**

Create `backend/test_watch_manager.py`. A fake observer keeps real filesystem timing out of the suite.

```python
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))

from pathlib import Path

from watch_manager import WatchManager


class FakeObserver:
    def __init__(self):
        self.scheduled = []
        self.unschedule_calls = 0
        self.started = False
        self.stopped = False

    def schedule(self, handler, path, recursive=False):
        self.scheduled.append((path, recursive))

    def unschedule_all(self):
        self.unschedule_calls += 1
        self.scheduled = []

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def join(self, timeout=None):
        pass


def test_start_schedules_the_folder_non_recursively(tmp_path):
    fake = FakeObserver()
    wm = WatchManager(on_upsert=lambda p: None, on_delete=lambda p: None,
                      extensions=[".txt"], observer_factory=lambda: fake)
    wm.start(str(tmp_path))

    assert fake.started is True
    assert fake.scheduled == [(str(tmp_path), False)]
    assert wm.folder == str(tmp_path)


def test_repoint_unschedules_the_previous_folder(tmp_path):
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    fake = FakeObserver()
    wm = WatchManager(on_upsert=lambda p: None, on_delete=lambda p: None,
                      extensions=[".txt"], observer_factory=lambda: fake)
    wm.start(str(one))
    wm.repoint(str(two))

    assert fake.unschedule_calls == 1
    assert fake.scheduled == [(str(two), False)]
    assert wm.folder == str(two)


def test_handler_ignores_unsupported_extensions(tmp_path):
    seen = []
    fake = FakeObserver()
    wm = WatchManager(on_upsert=seen.append, on_delete=lambda p: None,
                      extensions=[".txt"], observer_factory=lambda: fake)
    wm.start(str(tmp_path))

    img = tmp_path / "photo.png"
    img.write_bytes(b"not a document")
    wm._handler.on_created(_FakeEvent(str(img)))
    assert seen == []

    txt = tmp_path / "note.txt"
    txt.write_text("a document")
    wm._handler.on_created(_FakeEvent(str(txt)))
    assert seen == [str(txt.resolve())]


class _FakeEvent:
    def __init__(self, src_path):
        self.src_path = src_path
        self.is_directory = False
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `uv run pytest backend/test_watch_manager.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'watch_manager'`.

- [ ] **Step 3: Write the module**

Create `backend/watch_manager.py`:

```python
"""One watchdog Observer, repointable at runtime.

Callbacks are injected rather than imported so this module never depends on
local_rag_api, which imports it.
"""

import time
from pathlib import Path
from typing import Any, Callable, List, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

SETTLE_TRIES = 10
SETTLE_SLEEP_S = 0.4


class _Handler(FileSystemEventHandler):
    def __init__(self, on_upsert: Callable[[str], None], on_delete: Callable[[str], None],
                 extensions: List[str]):
        self._on_upsert = on_upsert
        self._on_delete = on_delete
        self._exts = [e.lower() for e in extensions]

    def _tracked(self, p: Path) -> bool:
        return p.suffix.lower() in self._exts

    def _settled_upsert(self, p: Path):
        """Wait for a copy to finish before reading, then ingest."""
        if not self._tracked(p):
            return
        try:
            size = p.stat().st_size
        except FileNotFoundError:
            return
        for _ in range(SETTLE_TRIES):
            time.sleep(SETTLE_SLEEP_S)
            try:
                current = p.stat().st_size
            except FileNotFoundError:
                return
            if current == size:
                break
            size = current
        self._on_upsert(str(p.resolve()))

    def on_created(self, event):
        if not event.is_directory:
            self._settled_upsert(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory:
            self._settled_upsert(Path(event.src_path))

    def on_deleted(self, event):
        if not event.is_directory:
            p = Path(event.src_path)
            if self._tracked(p):
                self._on_delete(str(p.resolve()))

    def on_moved(self, event):
        if event.is_directory:
            return
        old = Path(event.src_path)
        if self._tracked(old):
            self._on_delete(str(old.resolve()))
        self._settled_upsert(Path(event.dest_path))


class WatchManager:
    """Owns a single Observer and the one folder it currently watches."""

    def __init__(self, on_upsert: Callable[[str], None], on_delete: Callable[[str], None],
                 extensions: List[str], observer_factory: Callable[[], Any] = Observer):
        self._handler = _Handler(on_upsert, on_delete, extensions)
        self._observer = observer_factory()
        self.folder: Optional[str] = None

    def start(self, folder: str) -> None:
        self._observer.schedule(self._handler, folder, recursive=False)
        self._observer.start()
        self.folder = folder
        print(f"watching {folder}")

    def repoint(self, folder: str) -> None:
        self._observer.unschedule_all()
        self._observer.schedule(self._handler, folder, recursive=False)
        self.folder = folder
        print(f"watching {folder}")

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join(timeout=5)
```

The settle loop in the test path costs `SETTLE_TRIES * SETTLE_SLEEP_S` only when sizes keep changing; for a written-and-closed file the first comparison matches and it breaks after one 0.4 s sleep.

- [ ] **Step 4: Run them to confirm they pass**

Run: `uv run pytest backend/test_watch_manager.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Add the ingest lock**

The observer thread writes to Chroma alongside request threads. In `backend/local_rag_api.py`, add `import threading` and:

```python
_INGEST_LOCK = threading.Lock()
```

Wrap three bodies. In `_ingest_documents` (`:647`), take the lock around everything after the docstring. In `delete_by_source` (`:393`), take it around the whole body. In `get_store` (`:374`), take it around the cache-population branch — two threads could otherwise both build a store for the same key:

```python
def get_store(embedding_id: str, version: str):
    key = f"{embedding_id}::{version}"
    if key in _STORE_CACHE:
        return _STORE_CACHE[key]
    with _INGEST_LOCK:
        if key in _STORE_CACHE:
            return _STORE_CACHE[key]
        cname = _collection_name(embedding_id, version)
        ...
```

Keep the existing body inside; only the guard and indentation change.

- [ ] **Step 6: Wire the watcher into the app**

Add below `_ACTIVE`:

```python
def _watch_upsert(source_path: str):
    try:
        _ingest_local_file(source_path, _ACTIVE["embedding_id"], _ACTIVE["version"])
        print(f"file {Path(source_path).name} added to the database")
    except Exception as e:
        print(f"[error] watch ingest failed for {source_path}: {type(e).__name__}: {e}")


def _watch_delete(source_path: str):
    store, _, _, _ = get_store(_ACTIVE["embedding_id"], _ACTIVE["version"])
    delete_by_source(store, source_path)
    print(f"file {Path(source_path).name} removed from the database")


_WATCHER: Optional[Any] = None


def _start_watcher():
    global _WATCHER
    from watch_manager import WatchManager
    _WATCHER = WatchManager(_watch_upsert, _watch_delete, sorted(SUPPORTED_EXTS))
    _WATCHER.start(_ACTIVE["folder"])
```

The exception handler on `_watch_upsert` is deliberate: an unhandled exception in the observer thread would kill watching silently for every later file.

In `/watched_folder`, after `_ACTIVE.update(...)`:

```python
    if _WATCHER is not None:
        _WATCHER.repoint(str(folder_path))
```

And in the `__main__` block (`:1006`), start it and disable the reloader — a background thread and the auto-reloader do not mix, and the reloader would otherwise start two Observers:

```python
if __name__ == "__main__":
    os.makedirs(CHROMA_PATH, exist_ok=True)
    _start_watcher()
    app.run(host=HOST, port=PORT, debug=True, use_reloader=False)
```

Because the watcher starts only under `__main__`, the test suite never starts an Observer.

- [ ] **Step 7: Write the failing test for repointing on switch**

```python
def test_watched_folder_repoints_the_watcher(client, tmp_path):
    import local_rag_api as mod

    class Spy:
        def __init__(self):
            self.folder = None

        def repoint(self, folder):
            self.folder = folder

    spy = Spy()
    mod._WATCHER = spy
    try:
        folder = tmp_path / "docs"
        folder.mkdir()
        r = client.post("/watched_folder", json={"folder": str(folder), "embedding_id": "fake:any", "version": "vtest"})
        assert r.status_code == 200, r.data
        assert spy.folder == str(folder.resolve())
    finally:
        mod._WATCHER = None
```

- [ ] **Step 8: Run it, then the full suite**

Run: `uv run pytest backend/test_local_rag_api.py::test_watched_folder_repoints_the_watcher -v`
Expected: PASS.
Run: `uv run pytest backend frontend`
Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add backend/watch_manager.py backend/test_watch_manager.py backend/local_rag_api.py backend/test_local_rag_api.py
git commit -m "Move folder watching into the API process, repointable at runtime"
```

---

### Task 9: Gradio folders panel

**Files:**
- Modify: `frontend/gradio_app.py` (`fetch_sources` at :459, `refresh_sources_panel` at :484, `_format_ingest_banner` at :539, `build_ui` sources panel at :827)
- Test: `frontend/test_gradio_app.py`

**Interfaces:**
- Consumes: `GET /folders` (Task 6), `POST /watched_folder` (Task 7), `/sources?folder=` (Task 3)
- Produces:
  - `fetch_folders(api_base: str, embedding_id: str, version: str, timeout_s: int) -> Tuple[Dict[str, Any], Optional[str]]`
  - `render_folders_html(payload: Dict[str, Any]) -> str`
  - `switch_watched_folder(api_base, folder, embedding_id, version, ingest_timeout_s) -> Tuple[Optional[Dict[str, Any]], Optional[str]]`
  - `fetch_sources(...)` gains a `folder: str = ""` parameter, forwarded as the `folder` query param
  - `_format_ingest_banner` reports deletions

- [ ] **Step 1: Write the failing tests**

```python
def test_render_folders_html_marks_the_active_one():
    html_out = ga.render_folders_html({
        "active": "/data/alpha",
        "chroma_path": "/db",
        "folders": [
            {"path": "/data/alpha", "exists": True, "file_count": 2},
            {"path": "/data/beta", "exists": False, "file_count": 7},
        ],
    })
    assert "/data/alpha" in html_out
    assert "ACTIVE" in html_out
    assert "2 files" in html_out
    assert "7 files" in html_out
    assert "/db" in html_out


def test_fetch_sources_forwards_the_folder_param():
    captured = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"sources": [], "collection": "c"}

    def _get(url, params=None, timeout=None):
        captured["params"] = params
        return _Resp()

    with patch("requests.get", _get):
        ga.fetch_sources("http://x", "fake:any", "v1", 5, folder="/data/alpha")
    assert captured["params"]["folder"] == "/data/alpha"


def test_switch_watched_folder_posts_the_path():
    captured = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"status": "ok", "active_folder": "/data/alpha", "ingested": [], "deleted": []}

    def _post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _Resp()

    with patch("requests.post", _post):
        summary, err = ga.switch_watched_folder("http://x", "/data/alpha", "fake:any", "v1", 60)

    assert err is None
    assert captured["url"] == "http://x/watched_folder"
    assert captured["json"]["folder"] == "/data/alpha"
    assert summary["active_folder"] == "/data/alpha"


def test_switch_watched_folder_reports_a_rejected_path():
    class _Resp:
        status_code = 400

        def raise_for_status(self):
            raise ValueError("400 Client Error")

        def json(self):
            return {"error": "not a directory"}

    with patch("requests.post", lambda *a, **k: _Resp()):
        summary, err = ga.switch_watched_folder("http://x", "/nope", "fake:any", "v1", 60)

    assert summary is None
    assert "ValueError" in err


def test_ingest_banner_reports_deletions():
    out = ga._format_ingest_banner({"ingested": [{"chunks": 3}], "deleted": ["/a.txt"], "skipped_unchanged": 2}, None)
    assert "1 new file" in out
    assert "1 removed" in out
    assert "2 unchanged" in out
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `uv run pytest frontend/test_gradio_app.py -k "folders_html or folder_param or switch_watched or banner_reports" -v`
Expected: FAIL — `render_folders_html`, `switch_watched_folder` do not exist and `fetch_sources` takes no `folder`.

- [ ] **Step 3: Add the fetch and render helpers**

In `frontend/gradio_app.py`, beside `fetch_sources`:

```python
def fetch_folders(
    api_base: str,
    embedding_id: str,
    version: str,
    timeout_s: int = 30,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Call GET /folders and return (payload, error)."""
    api_base = (api_base or "").strip().rstrip("/")
    if not api_base:
        return {}, "API base URL is empty."
    params = {"embedding_id": (embedding_id or "").strip(), "version": (version or "").strip()}
    params = {k: v for k, v in params.items() if v}
    try:
        r = requests.get(f"{api_base}/folders", params=params, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {}, f"{type(e).__name__}: {e}"
    return (data if isinstance(data, dict) else {}), None


def render_folders_html(payload: Dict[str, Any]) -> str:
    """Folder list with per-folder document counts, active one marked."""
    folders = payload.get("folders") or []
    active = payload.get("active") or ""
    chroma_path = payload.get("chroma_path") or ""
    if not folders:
        return '<div class="nb-empty">No folders indexed yet.</div>'

    bits = [f'<div class="nb-counts">store: {html.escape(chroma_path)}</div>',
            '<div class="nb-sources-wrap">']
    for f in folders:
        path = f.get("path") or ""
        count = int(f.get("file_count") or 0)
        is_active = path == active
        dot = "nb-status-ok" if f.get("exists") else "nb-status-missing"
        mark = '<span class="nb-status-pill ok">ACTIVE</span>' if is_active else ""
        bits.append(
            '<div class="nb-source">'
            '  <div class="nb-source-body">'
            f'    <div class="nb-source-name">{html.escape(path)}</div>'
            f'    <div class="nb-counts">{count} files {mark}</div>'
            '  </div>'
            f'  <div class="nb-status-dot {dot}"></div>'
            '</div>'
        )
    bits.append("</div>")
    return "\n".join(bits)


def switch_watched_folder(
    api_base: str,
    folder: str,
    embedding_id: str,
    version: str,
    timeout_s: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Call POST /watched_folder. Returns (summary, error)."""
    api_base = (api_base or "").strip().rstrip("/")
    if not api_base:
        return None, "API base URL is empty."
    body = {"folder": (folder or "").strip(),
            "embedding_id": (embedding_id or "").strip(),
            "version": (version or "").strip()}
    body = {k: v for k, v in body.items() if v}
    try:
        r = requests.post(f"{api_base}/watched_folder", json=body, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    return (data if isinstance(data, dict) else {}), None
```

- [ ] **Step 4: Add `folder` to `fetch_sources` and `refresh_sources_panel`**

`fetch_sources` gains `folder: str = ""` as its last parameter and includes it in `params` (the existing `{k: v for k, v in params.items() if v}` already drops it when blank). `refresh_sources_panel` gains the same parameter and forwards it.

- [ ] **Step 5: Report deletions in the banner**

In `_format_ingest_banner` (`:539`), after the `ingested`/`skipped` lines:

```python
    deleted = summary.get("deleted") or []
    base = html.escape(
        f"Ingested {len(ingested)} new file{plural} · {total_chunks} chunks · "
        f"{len(deleted)} removed · {skipped} unchanged skipped"
    )
```

- [ ] **Step 6: Build the panel**

In `build_ui` (`:827`), above the existing Sources group:

```python
                with gr.Group(elem_classes=["nb-side"]):
                    gr.HTML('<div class="nb-section-title">Folders</div>')
                    folder_choice = gr.Dropdown(
                        label="Active folder",
                        choices=initial_folders,
                        value=initial_active,
                        allow_custom_value=True,
                        interactive=True,
                    )
                    use_folder_btn = gr.Button("Use this folder", variant="primary", size="sm")
                    folders_html = gr.HTML('<div class="nb-empty">Loading…</div>')
```

Add above the `with gr.Blocks(...)` line, beside the other `initial_*` values (`:813`):

```python
    initial_active = srv.get("active_folder") or DEFAULT_WATCHED_FOLDER
    _folders_payload, _ = fetch_folders(DEFAULT_API_BASE, initial_embedding_id, initial_version)
    initial_folders = [f["path"] for f in (_folders_payload.get("folders") or [])] or [initial_active]
```

Then the callback and wiring, near the existing `refresh_btn.click` (`:1093`):

```python
        def on_use_folder(api_base_v, folder_v, embedding_id_v, version_v, timeout_v):
            summary, err = switch_watched_folder(
                api_base_v, folder_v, embedding_id_v, version_v, DEFAULT_INGEST_TIMEOUT_S,
            )
            payload, _ = fetch_folders(api_base_v, embedding_id_v, version_v, timeout_v)
            active = (summary or {}).get("active_folder") or folder_v
            panel_html, pill = refresh_sources_panel(
                api_base_v, embedding_id_v, version_v, timeout_v, folder=active,
            )
            banner = _format_ingest_banner(summary, err)
            return (
                render_folders_html(payload),
                (banner + panel_html) if banner else panel_html,
                pill,
                gr.update(choices=[f["path"] for f in (payload.get("folders") or [])], value=active),
            )

        use_folder_btn.click(
            fn=on_use_folder,
            inputs=[api_base, folder_choice, embedding_id, version, timeout_s],
            outputs=[folders_html, sources_html, status_pill, folder_choice],
        )
```

Change `refresh_panel_with_ingest` (`:564`) to take the folder rather than reading the
module constant. Its new signature and body:

```python
def refresh_panel_with_ingest(
    api_base: str,
    embedding_id: str,
    version: str,
    timeout_s: int,
    folder: str = "",
    ingest_timeout_s: Optional[int] = None,
) -> Tuple[str, str]:
    """Reconcile the active folder then re-list /sources for it."""
    ingest_to = int(ingest_timeout_s) if ingest_timeout_s else DEFAULT_INGEST_TIMEOUT_S
    target = (folder or "").strip() or DEFAULT_WATCHED_FOLDER
    summary, err = trigger_folder_ingest(
        api_base=api_base,
        folder=target,
        embedding_id=embedding_id,
        version=version,
        timeout_s=ingest_to,
    )
    panel_html, pill = refresh_sources_panel(api_base, embedding_id, version, timeout_s, folder=target)
    banner = _format_ingest_banner(summary, err)
    return (banner + panel_html) if banner else panel_html, pill
```

Then replace the existing `refresh_btn.click` and `demo.load` wiring (`:1093-1103`) with:

```python
        refresh_btn.click(
            fn=refresh_panel_with_ingest,
            inputs=[api_base, embedding_id, version, timeout_s, folder_choice],
            outputs=[sources_html, status_pill],
        )

        def on_load(api_base_v, embedding_id_v, version_v, timeout_v, folder_v):
            payload, _ = fetch_folders(api_base_v, embedding_id_v, version_v, timeout_v)
            active = payload.get("active") or folder_v
            panel_html, pill = refresh_sources_panel(
                api_base_v, embedding_id_v, version_v, timeout_v, folder=active,
            )
            choices = [f["path"] for f in (payload.get("folders") or [])] or [active]
            return render_folders_html(payload), panel_html, pill, gr.update(choices=choices, value=active)

        demo.load(
            fn=on_load,
            inputs=[api_base, embedding_id, version, timeout_s, folder_choice],
            outputs=[folders_html, sources_html, status_pill, folder_choice],
        )
```

`demo.load` no longer calls `refresh_sources_panel` directly, so the sources list is
folder-scoped from the first paint.

- [ ] **Step 7: Run the tests to confirm they pass**

Run: `uv run pytest frontend/test_gradio_app.py -k "folders_html or folder_param or switch_watched or banner_reports" -v`
Expected: PASS (5 tests).

- [ ] **Step 8: Run the full suite**

Run: `uv run pytest backend frontend`
Expected: all green. `test_launch_compatible_defaults_off_so_the_preset_reaches_the_server` (`frontend/test_gradio_app.py:751`) calls `build_ui()` with `requests.get` patched to raise — the new `fetch_folders` call at build time must tolerate that, which it does by returning `({}, err)`.

- [ ] **Step 9: Commit**

```bash
git add frontend/gradio_app.py frontend/test_gradio_app.py
git commit -m "Add a folders panel and scope the sources list to the active folder"
```

---

### Task 10: Restrict-retrieval checkbox

**Files:**
- Modify: `frontend/gradio_app.py` (`build_request_body` at :324, `ask_api` at :587, `build_ui`)
- Test: `frontend/test_gradio_app.py`

**Interfaces:**
- Consumes: `folder` on `/answer` (Task 2)
- Produces: `build_request_body(..., folder: str = "")` — includes `folder` only when non-empty and `launch_compatible` is False

- [ ] **Step 1: Write the failing tests**

```python
def test_build_request_body_includes_folder():
    body = ga.build_request_body(
        question="q", llm_id="mock:any", embedding_id="fake:any", version="v1",
        top_k=6, fetch_k=24, search_type="mmr", mmr_lambda=0.3,
        max_context_chars=18000, launch_compatible=False, folder="/data/alpha",
    )
    assert body["folder"] == "/data/alpha"


def test_build_request_body_omits_blank_folder():
    body = ga.build_request_body(
        question="q", llm_id="mock:any", embedding_id="fake:any", version="v1",
        top_k=6, fetch_k=24, search_type="mmr", mmr_lambda=0.3,
        max_context_chars=18000, launch_compatible=False, folder="",
    )
    assert "folder" not in body


def test_launch_compatible_still_drops_folder():
    body = ga.build_request_body(
        question="q", llm_id="mock:any", embedding_id="fake:any", version="v1",
        top_k=6, fetch_k=24, search_type="mmr", mmr_lambda=0.3,
        max_context_chars=18000, launch_compatible=True, folder="/data/alpha",
    )
    assert body == {"query": "q", "llm_id": "mock:any"}
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `uv run pytest frontend/test_gradio_app.py -k "includes_folder or blank_folder or still_drops_folder" -v`
Expected: FAIL with `TypeError: build_request_body() got an unexpected keyword argument 'folder'`.

- [ ] **Step 3: Implement**

Add `folder: str = ""` as the last parameter of `build_request_body` and one entry to the non-compatible body dict (`:351`):

```python
        "folder": (folder or "").strip(),
```

The existing trailing `{k: v for k, v in body.items() if v not in ("", None, [])}` drops it when blank. The `launch_compatible` early return already excludes it.

- [ ] **Step 4: Wire the checkbox**

In `build_ui`, beside `launch_compatible` (`:863`):

```python
                            restrict_to_folder = gr.Checkbox(
                                label="restrict retrieval to active folder",
                                value=True,
                            )
```

`ask_api` (`:590`) and `chat_step` (`:701`) each gain a `folder` parameter. Placement
matters: `ask_api`'s existing comment at `:604` notes that `reranker_id` precedes
`history` so the Ask tab's **positional** Gradio inputs bind correctly. Put `folder`
after `reranker_id` and before `history`, so it becomes the next positional slot:

```python
    reranker_id: str = "",
    folder: str = "",
    history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, str, str, str, str]:
```

In `ask_api`'s `build_request_body(...)` call, add `folder=folder`. Give `chat_step`
the same trailing `folder: str = ""` parameter and forward it in its `ask_api(...)`
call alongside `reranker_id=reranker_id`.

The checkbox blanks the folder rather than the handlers branching on it. Add this
helper next to `on_use_folder`:

```python
        def _scoped_folder(restrict_v, folder_v):
            """The folder to scope retrieval by: blank means search everything."""
            return (folder_v or "") if restrict_v else ""
```

Because Gradio binds inputs positionally, the checkbox cannot be passed straight
through — wrap each handler so the two UI values collapse into one argument:

```python
        def ask_scoped(*args):
            *head, restrict_v, folder_v = args
            return ask_api(*head, _scoped_folder(restrict_v, folder_v))

        def chat_scoped(*args):
            *head, restrict_v, folder_v = args
            return chat_step(*head, _scoped_folder(restrict_v, folder_v))
```

Point the Ask and Chat `.click`/`.submit` handlers at `ask_scoped` / `chat_scoped`,
and append `restrict_to_folder, folder_choice` to the end of each handler's existing
`inputs` list. Leave the Compare tab unscoped — it exists to compare retrieval
configurations, and a third variable would muddy that.

- [ ] **Step 5: Run the tests, then the full suite**

Run: `uv run pytest frontend/test_gradio_app.py -k folder -v`
Expected: PASS.
Run: `uv run pytest backend frontend`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add frontend/gradio_app.py frontend/test_gradio_app.py
git commit -m "Add a restrict-retrieval-to-active-folder switch"
```

---

### Task 11: Scripts and documentation

**Files:**
- Modify: `scripts/start-mac.sh:135-155`, `scripts/stop-mac.sh`, `CLAUDE.md`

**Interfaces:**
- Consumes: everything above
- Produces: no code interfaces; the launcher no longer starts a second watcher

- [ ] **Step 1: Read the current watcher blocks**

Run: `sed -n '130,160p' scripts/start-mac.sh` and `grep -n "watcher\|WATCH" scripts/stop-mac.sh`
Note the exact line ranges before editing.

- [ ] **Step 2: Remove the watcher block from `start-mac.sh`**

Delete the block starting at the `# --- Watcher (only if watched_folder exists) ---` comment through the `fi` that closes it, and the `[[ -f "$WATCH_PID_FILE" ]] && echo …` summary line near the end. Replace the block with a single comment so the change reads clearly:

```bash
# --- Watcher ---
# The API process now owns folder watching (see backend/watch_manager.py) and
# repoints itself when the UI switches folders. No separate watcher process.
```

- [ ] **Step 3: Remove the matching teardown from `stop-mac.sh`**

Delete the lines that kill the watcher pid and remove `watcher.pid`, keeping the API and Gradio teardown as is.

- [ ] **Step 4: Verify the stack still starts**

Run: `bash scripts/start-mac.sh`
Expected: API healthy, UI reachable on `http://127.0.0.1:7860`, and `logs/api.log` contains a `watching <folder>` line and no watcher.pid message.
Then run: `bash scripts/stop-mac.sh`
Expected: both processes stop, no error about a missing watcher pid file.

- [ ] **Step 5: Update `CLAUDE.md`**

Edit these places:

- The `backend/local_rag_api.py` bullet: add `/folders` and `/watched_folder` to the endpoint list; note that `sync_folder` reconciles against the store and that `/ingest_folder` now deletes as well as adds.
- The `backend/folder_watcher.py` bullet: say the API owns watching in-process via `backend/watch_manager.py`, and the daemon remains for headless use but is no longer started by `scripts/start-mac.sh`.
- Add a `backend/folder_registry.py` bullet describing `<CHROMA_DB_PATH>/folders.json` and its `{active, known}` shape.
- The `frontend/gradio_app.py` bullet: the Folders panel, the active-folder dropdown, the "restrict retrieval to active folder" checkbox, and that the sources list is folder-scoped.
- The "Collection naming (critical invariant)" section: add that `folder` metadata, not the collection name, separates folders, and that chunks written before this feature are backfilled on first store access with no re-embedding.
- The "Folder-ingest flow" section: rewrite around `sync_folder`'s three inputs and the fact that the store, not the state file, decides whether a file is ingested.
- "Points of attention": replace the `CHROMA_COLLECTION_VERSION` / `EMBEDDING_ID` watcher-drift warning, since one process now owns both.
- The Commands section: drop step 2 (starting the watcher) from the normal flow, keeping it as an optional headless note.

- [ ] **Step 6: Run the full suite one last time**

Run: `uv run pytest backend frontend`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add scripts/start-mac.sh scripts/stop-mac.sh CLAUDE.md
git commit -m "Retire the separate watcher process and document the folder feature"
```

---

## Manual Verification

After Task 11, confirm the end-to-end behaviour the spec asked for. Automated tests cover the pieces; this covers the wiring.

- [ ] Start the stack: `bash scripts/start-mac.sh`, open `http://127.0.0.1:7860`.
- [ ] The Folders panel lists the default watched folder, marked ACTIVE, with a file count.
- [ ] Type a new absolute path of a folder holding one PDF into the dropdown, click "Use this folder". The banner reports the ingest, the Sources list shows only that folder's file, and the Folders panel marks the new folder ACTIVE.
- [ ] Copy another PDF into that folder. Within a few seconds `logs/api.log` shows `file <name> added to the database`, and clicking "Refresh sources" lists it.
- [ ] Delete a file from that folder. `logs/api.log` shows `file <name> removed from the database` and it disappears from Sources.
- [ ] Switch back to the original folder. Its documents reappear without re-ingesting (the banner reports 0 new, N unchanged) — this is the check that switching away is not destructive.
- [ ] Ask a question whose answer only exists in the non-active folder, with "restrict retrieval to active folder" ticked. The answer should be "Not found in the provided context." Untick it and ask again — the answer should now be found.
- [ ] `curl -s http://127.0.0.1:5050/folders | jq` shows both folders with correct counts.
