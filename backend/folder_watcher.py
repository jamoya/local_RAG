from __future__ import annotations

"""
Folder watcher (no n8n) that:
- Ingests new/changed files to the local RAG API (/ingest)
- Deletes vectors when files are removed (/delete)
- Reconcile mode: periodically removes vectors for files that no longer exist on disk,
  even if filesystem events were missed (uses API /sources + disk scan).

Usage:
  uv run python backend/folder_watcher.py \
    --watch "/Users/jm/PythonProjects/_WORK/Local_RAG/watched_folder" \
    --api "http://127.0.0.1:5000" \
    --embedding-id "hf:BAAI/bge-large-en-v1.5" \
    --version "v1" \
    --reconcile-interval 300

Requires: watchdog, requests
"""

import argparse
import json
import os
import hashlib
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Set

import requests
from watchdog.events import (
    FileSystemEventHandler,
    FileCreatedEvent,
    FileModifiedEvent,
    FileDeletedEvent,
    FileMovedEvent,
)
from watchdog.observers import Observer


DEFAULT_EXTENSIONS = [".pdf", ".txt", ".md", ".docx"]


def now() -> float:
    return time.time()


def load_state(path: Path) -> Dict[str, Any]:
    """Load state from disk, ensuring required keys exist.

    The state file may come from older versions, be empty, or be manually edited.
    We defensively normalize it to: {"files": {<abs_path>: {"mtime": .., "size": ..}}}.
    """
    default: Dict[str, Any] = {"files": {}}
    if not path.exists():
        return default
    try:
        raw = path.read_text().strip()
        data = json.loads(raw) if raw else {}
    except Exception:
        return default
    if not isinstance(data, dict):
        return default
    if "files" not in data or not isinstance(data.get("files"), dict):
        data["files"] = {}
    return data

def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.write_text(json.dumps(state, indent=2, sort_keys=True))

def _slug(s: str, max_len: int = 48) -> str:
    s2 = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s2[:max_len] if len(s2) > max_len else s2


def state_file_path(watch_dir: Path, version: str, embedding_id: str) -> Path:
    # Separate state per (version, embedding_id) so switching models doesn't reuse stale state.
    key = f"{version}|{embedding_id}".encode("utf-8")
    h = hashlib.sha1(key).hexdigest()[:10]
    return watch_dir / f".ingested_state__{version}__{_slug(embedding_id)}__{h}.json"


def file_fingerprint(p: Path) -> Dict[str, Any]:
    st = p.stat()
    return {"mtime": st.st_mtime, "size": st.st_size}


def should_track(p: Path, exts: List[str]) -> bool:
    return p.is_file() and p.suffix.lower() in [e.lower() for e in exts]


def api_post_multipart(url: str, file_path: Path, embedding_id: str, version: str) -> requests.Response:
    with file_path.open("rb") as f:
        files = {"file": (file_path.name, f)}
        data = {"embedding_id": embedding_id, "version": version, "source_path": str(file_path.resolve())}
        return requests.post(url, files=files, data=data, timeout=600)


def api_delete(url: str, source_path: str, embedding_id: str, version: str) -> requests.Response:
    payload = {"source_path": source_path, "embedding_id": embedding_id, "version": version}
    return requests.post(url, json=payload, timeout=60)


def api_sources(url: str, embedding_id: str, version: str) -> Set[str]:
    params = {"embedding_id": embedding_id, "version": version}
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()
    return set(data.get("source_paths", []) or [])


class WatchHandler(FileSystemEventHandler):
    def __init__(self, cfg, state_path: Path):
        self.cfg = cfg
        self.state_path = state_path
        self.state = load_state(state_path)
        # Backward/forward compatibility: guarantee required keys
        self.state.setdefault("files", {})

    def _ingest(self, p: Path):
        if not should_track(p, self.cfg.extensions):
            return

        fp = file_fingerprint(p)
        key = str(p.resolve())
        prev = self.state["files"].get(key)

        if prev and prev.get("mtime") == fp["mtime"] and prev.get("size") == fp["size"]:
            return  # unchanged

        ingest_url = self.cfg.api.rstrip("/") + "/ingest"
        print(f"[ingest] {p.name} -> {ingest_url} (embedding_id={self.cfg.embedding_id}, version={self.cfg.version})")

        # Wait for file to settle (helpful for large copies)
        for _ in range(10):
            time.sleep(0.4)
            try:
                fp2 = file_fingerprint(p)
                if fp2["size"] == fp["size"]:
                    break
                fp = fp2
            except FileNotFoundError:
                return

        try:
            r = api_post_multipart(ingest_url, p, self.cfg.embedding_id, self.cfg.version)
            if r.status_code >= 400:
                print(f"[error] ingest failed for {p.name}: API error {r.status_code}: {r.text}")
                return
            self.state["files"][key] = fp
            save_state(self.state_path, self.state)
        except requests.RequestException as e:
            print(f"[error] ingest failed for {p.name}: {e}")

    def _delete(self, key: str):
        delete_url = self.cfg.api.rstrip("/") + "/delete"
        print(f"[delete] {key} -> {delete_url} (embedding_id={self.cfg.embedding_id}, version={self.cfg.version})")
        try:
            r = api_delete(delete_url, key, self.cfg.embedding_id, self.cfg.version)
            if r.status_code >= 400:
                print(f"[error] delete failed for {key}: API error {r.status_code}: {r.text}")
                return
        except requests.RequestException as e:
            print(f"[error] delete failed for {key}: {e}")

        # Remove from local state regardless; reconcile can clean up later if API was down.
        self.state["files"].pop(key, None)
        save_state(self.state_path, self.state)

    def on_created(self, event):
        if isinstance(event, FileCreatedEvent) and not event.is_directory:
            self._ingest(Path(event.src_path))

    def on_modified(self, event):
        if isinstance(event, FileModifiedEvent) and not event.is_directory:
            self._ingest(Path(event.src_path))

    def on_deleted(self, event):
        if isinstance(event, FileDeletedEvent) and not event.is_directory:
            self._delete(str(Path(event.src_path).resolve()))

    def on_moved(self, event):
        if isinstance(event, FileMovedEvent) and not event.is_directory:
            # treat as delete old + ingest new
            old_key = str(Path(event.src_path).resolve())
            self._delete(old_key)
            self._ingest(Path(event.dest_path))


def scan_disk(watch_dir: Path, exts: List[str], recursive: bool) -> Set[str]:
    paths: Set[str] = set()
    if recursive:
        it = watch_dir.rglob("*")
    else:
        it = watch_dir.glob("*")
    for p in it:
        if should_track(p, exts):
            paths.add(str(p.resolve()))
    return paths


def reconcile(cfg, handler: WatchHandler, watch_dir: Path):
    """
    Reconcile database/state with disk.
    - Ask API for all source_paths in DB (preferred)
    - Compare with current files on disk
    - Delete DB entries not on disk
    """
    disk = scan_disk(watch_dir, cfg.extensions, cfg.recursive)

    # union of "known" sources (local state + API)
    known: Set[str] = set(handler.state.get("files", {}).keys())

    if cfg.reconcile_from_api:
        try:
            src_url = cfg.api.rstrip("/") + "/sources"
            known |= api_sources(src_url, cfg.embedding_id, cfg.version)
        except Exception as e:
            print(f"[reconcile] warning: could not fetch /sources: {e}")

    stale = sorted(known - disk)
    if stale:
        print(f"[reconcile] stale sources found: {len(stale)}. Deleting...")
    for key in stale:
        handler._delete(key)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", required=True, help="Folder to watch")
    ap.add_argument("--api", default=os.environ.get("RAG_API", os.environ.get("RAG_API_BASE", "http://127.0.0.1:5000")), help="Base URL of local RAG API (or env RAG_API / RAG_API_BASE)")
    ap.add_argument("--embedding-id", default=os.environ.get("EMBEDDING_ID", "hf:BAAI/bge-large-en-v1.5"))
    ap.add_argument("--version", default=os.environ.get("CHROMA_COLLECTION_VERSION", os.environ.get("VERSION", "v1")))
    ap.add_argument("--extensions", nargs="*", default=DEFAULT_EXTENSIONS)
    ap.add_argument("--recursive", action="store_true", help="Watch subfolders too")

    ap.add_argument("--reconcile-interval", type=int, default=0, help="Seconds between reconcile runs (0 disables)")
    ap.add_argument("--reconcile-from-api", action="store_true", default=True, help="Use API /sources in reconcile (recommended)")
    ap.add_argument("--no-reconcile-from-api", dest="reconcile_from_api", action="store_false")

    return ap.parse_args()


def main():
    cfg = parse_args()
    watch_dir = Path(cfg.watch).expanduser().resolve()
    if not watch_dir.exists() or not watch_dir.is_dir():
        print(f"Watch directory does not exist: {watch_dir}", file=sys.stderr)
        sys.exit(2)

    state_path = state_file_path(watch_dir, cfg.version, cfg.embedding_id)

    print(f"Watching: {watch_dir} (recursive={cfg.recursive})")
    print(f"API: {cfg.api}")
    print(f"Embedding: {cfg.embedding_id}")
    print(f"Version: {cfg.version}")
    print(f"Extensions: {cfg.extensions}")
    print(f"State: {state_path}")
    if cfg.reconcile_interval:
        print(f"Reconcile: every {cfg.reconcile_interval}s (from_api={cfg.reconcile_from_api})")

    handler = WatchHandler(cfg, state_path)

    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=cfg.recursive)
    observer.start()

    # Initial scan: ingest any new/changed files
    for p_str in sorted(scan_disk(watch_dir, cfg.extensions, cfg.recursive)):
        handler._ingest(Path(p_str))

    last_reconcile = 0.0
    try:
        while True:
            time.sleep(1.0)
            if cfg.reconcile_interval and (now() - last_reconcile) >= cfg.reconcile_interval:
                reconcile(cfg, handler, watch_dir)
                last_reconcile = now()
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()