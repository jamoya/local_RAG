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
