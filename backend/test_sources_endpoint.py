"""Backend tests for the /sources endpoint and the on-disk-existence guarantee.

These tests guarantee that anything the Gradio Sources panel renders is
backed by a real file on disk at the moment of ingest.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import io
import os
import importlib

import pytest


def _make_app(tmp_path):
    os.environ["CHROMA_DB_PATH"] = str(tmp_path / "chroma")
    os.environ["CHROMA_BASE_COLLECTION_NAME"] = "test_kb_sources"
    os.environ["CHROMA_COLLECTION_VERSION"] = "vsrc"
    os.environ["EMBEDDING_ID"] = "fake:any"
    os.environ["FAKE_EMB_DIM"] = "64"
    os.environ["LLM_ID"] = "mock:any"
    mod = importlib.import_module("local_rag_api")
    importlib.reload(mod)
    return mod.app


@pytest.fixture()
def client(tmp_path):
    app = _make_app(tmp_path)
    app.config["TESTING"] = True
    return app.test_client()


def _ingest(client, tmp_path: Path, name: str, body: str) -> str:
    """Write a tmp .txt file, ingest it, return its absolute path."""
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    data = {
        "embedding_id": "fake:any",
        "version": "vsrc",
        "source_path": str(p),
        "file": (io.BytesIO(body.encode("utf-8")), name),
    }
    r = client.post("/ingest", data=data, content_type="multipart/form-data")
    assert r.status_code == 200, r.data
    return str(p)


def test_sources_empty_collection(client):
    r = client.get("/sources?embedding_id=fake:any&version=vsrc")
    assert r.status_code == 200
    js = r.get_json()
    assert "sources" in js
    assert js["sources"] == []
    assert js["embedding_id"] == "fake:any"
    assert js["version"] == "vsrc"


def test_sources_lists_ingested_files(client, tmp_path):
    p1 = _ingest(client, tmp_path, "doc-a.txt", "The first document is about steel BAT.")
    p2 = _ingest(client, tmp_path, "doc-b.txt", "The second document is about cement.")

    r = client.get("/sources?embedding_id=fake:any&version=vsrc")
    assert r.status_code == 200
    js = r.get_json()
    assert set(js["sources"]) == {p1, p2}


def test_sources_paths_exist_on_disk(client, tmp_path):
    """Every path returned by /sources at this moment must exist on disk."""
    _ingest(client, tmp_path, "alpha.txt", "alpha content")
    _ingest(client, tmp_path, "beta.txt", "beta content")
    _ingest(client, tmp_path, "gamma.txt", "gamma content")

    r = client.get("/sources?embedding_id=fake:any&version=vsrc")
    assert r.status_code == 200
    paths = r.get_json()["sources"]
    assert paths, "expected at least one source"
    for p in paths:
        assert Path(p).is_file(), f"Source path returned by /sources does not exist: {p}"


def test_sources_drops_deleted_file(client, tmp_path):
    """After /delete the source must disappear from /sources."""
    p = _ingest(client, tmp_path, "to_delete.txt", "transient content")
    r = client.post("/delete", json={
        "embedding_id": "fake:any", "version": "vsrc", "source_path": p,
    })
    assert r.status_code == 200
    r = client.get("/sources?embedding_id=fake:any&version=vsrc")
    assert r.status_code == 200
    assert p not in r.get_json()["sources"]


def test_sources_response_shape(client, tmp_path):
    """Contract: the keys the frontend relies on must be present."""
    _ingest(client, tmp_path, "shape.txt", "shape content")
    r = client.get("/sources?embedding_id=fake:any&version=vsrc")
    js = r.get_json()
    for key in ("sources", "collection", "backend", "embedding_id", "version"):
        assert key in js, f"missing key {key!r} in /sources response"
    assert isinstance(js["sources"], list)
    assert all(isinstance(s, str) for s in js["sources"])
