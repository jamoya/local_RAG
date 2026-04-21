import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))

import os
import io
import importlib
from pathlib import Path

import pytest

def _make_app(tmp_path: Path):
    # Isolate Chroma DB per test run
    os.environ["CHROMA_DB_PATH"] = str(tmp_path / "chroma")
    os.environ["CHROMA_BASE_COLLECTION_NAME"] = "test_kb"
    os.environ["CHROMA_COLLECTION_VERSION"] = "vtest"
    # Use offline embeddings/LLM
    os.environ["EMBEDDING_ID"] = "fake:any"
    os.environ["FAKE_EMB_DIM"] = "128"
    os.environ["LLM_ID"] = "mock:any"
    # Import after env is set
    mod = importlib.import_module("local_rag_api")
    importlib.reload(mod)
    return mod.app

@pytest.fixture()
def client(tmp_path):
    app = _make_app(tmp_path)
    app.config["TESTING"] = True
    return app.test_client()

def test_health_and_ready(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"

    r = client.get("/ready?embedding_id=fake:any&version=vtest")
    assert r.status_code == 200
    js = r.get_json()
    assert js["status"] == "ready"
    assert js["embedding_id"] == "fake:any"
    assert js["version"] == "vtest"

def test_ingest_retrieve_answer_txt(client, tmp_path):
    content = (
        "The BREF covers smitheries with hammers where the energy exceeds 50 kilojoule per hammer, "
        "where the calorific power used exceeds 20 MW."
    ).encode("utf-8")
    data = {
        "embedding_id": "fake:any",
        "version": "vtest",
        "source_path": str(tmp_path / "doc.txt"),
    }
    r = client.post("/ingest", data={**data, "file": (io.BytesIO(content), "doc.txt")}, content_type="multipart/form-data")
    assert r.status_code == 200, r.data
    js = r.get_json()
    assert js["chunks_added"] >= 1

    q = {"query": "What are the specific energy and power thresholds that define the coverage of smitheries?",
         "embedding_id": "fake:any", "version": "vtest", "top_k": 3}
    r = client.post("/retrieve", json=q)
    assert r.status_code == 200
    ctx = r.get_json()["context"]
    assert "50 kilojoule" in ctx or "50 kilojoules" in ctx

    r = client.post("/answer", json={**q, "llm_id": "mock:any"})
    assert r.status_code == 200
    ans = r.get_json()["answer"]
    assert "MOCK_ANSWER" in ans

def test_ingest_pdf_page_range(client, tmp_path):
    # Create a tiny PDF with 2 pages and ingest only page 2.
    from reportlab.pdfgen import canvas
    pdf_path = tmp_path / "mini.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(72, 720, "PAGE ONE: ignore me")
    c.showPage()
    c.drawString(72, 720, "PAGE TWO: AOX 0.1 – 1 mg/l applies to wet scrubbing of cupola off-gases.")
    c.save()

    data = {
        "embedding_id": "fake:any",
        "version": "vtest",
        "source_path": str(pdf_path),
        "page_start": "2",
        "page_end": "2",
    }
    r = client.post("/ingest", data={**data, "file": (pdf_path.open("rb"), "mini.pdf")}, content_type="multipart/form-data")
    assert r.status_code == 200, r.data

    q = {"query": "What is the BAT-AEL for AOX for direct discharges and which waste water stream does it apply to?",
         "embedding_id": "fake:any", "version": "vtest", "top_k": 5}
    r = client.post("/retrieve", json=q)
    assert r.status_code == 200
    ctx = r.get_json()["context"]
    assert "AOX" in ctx
    assert "0.1" in ctx or "1 mg/l" in ctx
