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
