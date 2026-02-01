import os
import requests
import pytest

API = os.environ.get("RAG_API", "http://127.0.0.1:5000")
EMBEDDING_ID = os.environ.get("EMBEDDING_ID", "hf:BAAI/bge-large-en-v1.5")
VERSION = os.environ.get("VERSION", "v1")


def test_sources_endpoint():
    r = requests.get(
        f"{API}/sources",
        params={"embedding_id": EMBEDDING_ID, "version": VERSION},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    assert isinstance(data, dict)
    assert "source_paths" in data
    assert isinstance(data["source_paths"], list)


def test_ingest_then_delete_roundtrip(tmp_path):
    p = tmp_path / "rag_test.txt"
    p.write_text("hello from pytest\n", encoding="utf-8")
    source_path = str(p.resolve())

    # ingest
    with p.open("rb") as f:
        files = {"file": (p.name, f)}
        data = {
            "embedding_id": EMBEDDING_ID,
            "version": VERSION,
            "source_path": source_path,
        }
        r = requests.post(f"{API}/ingest", files=files, data=data, timeout=120)
    r.raise_for_status()

    # verify appears
    r = requests.get(
        f"{API}/sources",
        params={"embedding_id": EMBEDDING_ID, "version": VERSION},
        timeout=30,
    )
    r.raise_for_status()
    sources = set(r.json().get("source_paths", []))
    assert source_path in sources

    # delete
    payload = {"source_path": source_path, "embedding_id": EMBEDDING_ID, "version": VERSION}
    r = requests.post(f"{API}/delete", json=payload, timeout=30)
    r.raise_for_status()

    # verify removed
    r = requests.get(
        f"{API}/sources",
        params={"embedding_id": EMBEDDING_ID, "version": VERSION},
        timeout=30,
    )
    r.raise_for_status()
    sources = set(r.json().get("source_paths", []))
    assert source_path not in sources


def test_ingest_missing_required_fields_returns_400(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("bad\n", encoding="utf-8")

    with p.open("rb") as f:
        files = {"file": (p.name, f)}
        # Missing embedding_id/version/source_path on purpose
        try:
            r = requests.post(f"{API}/ingest", files=files, data={}, timeout=10)
        except requests.exceptions.ReadTimeout:
            pytest.fail("Server timed out instead of returning 4xx for missing required fields")

    assert 400 <= r.status_code < 500
    data = r.json()
    assert data.get("error")
    assert "missing" in data
    assert set(data["missing"]) == {"embedding_id", "version", "source_path"}
