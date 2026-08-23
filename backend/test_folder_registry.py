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
