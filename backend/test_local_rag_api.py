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

def test_answer_accepts_history_and_forwards_to_llm(client, tmp_path):
    content = b"Smitheries covered by the BREF have hammers exceeding 50 kilojoule."
    r = client.post(
        "/ingest",
        data={
            "embedding_id": "fake:any", "version": "vtest",
            "source_path": str(tmp_path / "doc.txt"),
            "file": (io.BytesIO(content), "doc.txt"),
        },
        content_type="multipart/form-data",
    )
    assert r.status_code == 200, r.data

    history = [
        {"role": "user", "content": "my name is Jose"},
        {"role": "assistant", "content": "Nice to meet you, Jose."},
    ]
    r = client.post("/answer", json={
        "query": "what is my name?",
        "embedding_id": "fake:any", "version": "vtest", "llm_id": "mock:any",
        "history": history,
    })
    assert r.status_code == 200, r.data
    ans = r.get_json()["answer"]
    assert "MOCK_ANSWER" in ans
    # Mock LLM echoes the last message content; history is passed between
    # system + final user, so the final message is the RAG user turn.
    assert "what is my name?" in ans


def test_answer_ignores_malformed_history(client, tmp_path):
    content = b"Some content to ingest."
    r = client.post(
        "/ingest",
        data={
            "embedding_id": "fake:any", "version": "vtest",
            "source_path": str(tmp_path / "doc.txt"),
            "file": (io.BytesIO(content), "doc.txt"),
        },
        content_type="multipart/form-data",
    )
    assert r.status_code == 200

    # history is not a list -> ignored, not 500
    r = client.post("/answer", json={
        "query": "hi", "embedding_id": "fake:any", "version": "vtest",
        "llm_id": "mock:any", "history": "not-a-list",
    })
    assert r.status_code == 200
    assert "MOCK_ANSWER" in r.get_json()["answer"]


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


# ----------------------------
# /config
# ----------------------------

def test_config_returns_server_defaults(client):
    r = client.get("/config")
    assert r.status_code == 200
    js = r.get_json()
    assert js["embedding_id"] == "fake:any"
    assert js["version"] == "vtest"
    assert js["llm_id"] == "mock:any"
    assert isinstance(js["watched_folder"], str)
    assert set(js["supported_extensions"]) == {".pdf", ".txt", ".md", ".docx"}


# ----------------------------
# /ingest_folder
# ----------------------------

def test_ingest_folder_missing_folder_returns_400(client, tmp_path):
    r = client.post("/ingest_folder", json={"folder": str(tmp_path / "does-not-exist")})
    assert r.status_code == 400
    assert "folder not found" in r.get_json()["error"].lower()


def test_ingest_folder_ingests_new_files(client, tmp_path):
    (tmp_path / "a.txt").write_text("alpha content one", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta markdown two", encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"binary ignored")  # unsupported ext

    r = client.post("/ingest_folder", json={
        "folder": str(tmp_path),
        "embedding_id": "fake:any",
        "version": "vtest",
    })
    assert r.status_code == 200, r.data
    js = r.get_json()
    assert js["status"] == "ok"
    assert js["scanned"] == 2  # .bin is not tracked
    assert len(js["ingested"]) == 2
    assert js["skipped_unchanged"] == 0
    assert js["errors"] == []

    r2 = client.get("/sources?embedding_id=fake:any&version=vtest")
    srcs = set(r2.get_json()["sources"])
    assert str((tmp_path / "a.txt").resolve()) in srcs
    assert str((tmp_path / "b.md").resolve()) in srcs


# ----------------------------
# Anthropic provider + cache_control plumbing
# ----------------------------

def test_build_chat_messages_emits_cache_control_blocks(tmp_path):
    app = _make_app(tmp_path)  # noqa: F841 - ensures module is loaded
    import local_rag_api as mod

    msgs = mod._build_chat_messages(
        "system prompt",
        history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        user_text="QUESTION ...",
        cacheable_user_prefix="CONTEXT block ...",
    )
    # System block is structured with cache_control
    assert isinstance(msgs[0].content, list)
    assert msgs[0].content[0]["cache_control"] == {"type": "ephemeral"}
    assert msgs[0].content[0]["text"] == "system prompt"
    # Final user message has two blocks: cached prefix + uncached question
    final = msgs[-1].content
    assert isinstance(final, list)
    assert len(final) == 2
    assert final[0]["text"] == "CONTEXT block ..."
    assert final[0]["cache_control"] == {"type": "ephemeral"}
    assert final[1]["text"] == "QUESTION ..."
    assert "cache_control" not in final[1]


def test_make_llm_anthropic_returns_chat_anthropic(tmp_path):
    app = _make_app(tmp_path)  # noqa: F841
    import local_rag_api as mod

    if mod.ChatAnthropic is None:
        pytest.skip("langchain-anthropic not installed in this env")
    llm = mod.make_llm("anthropic:claude-sonnet-4-5")
    assert llm.__class__.__name__ == "ChatAnthropic"
    assert getattr(llm, "model", None) or getattr(llm, "model_name", None)


def test_ingest_folder_skips_unchanged_on_second_call(client, tmp_path):
    (tmp_path / "x.txt").write_text("hello", encoding="utf-8")

    r1 = client.post("/ingest_folder", json={
        "folder": str(tmp_path), "embedding_id": "fake:any", "version": "vtest",
    })
    assert r1.status_code == 200
    assert len(r1.get_json()["ingested"]) == 1

    r2 = client.post("/ingest_folder", json={
        "folder": str(tmp_path), "embedding_id": "fake:any", "version": "vtest",
    })
    assert r2.status_code == 200
    js = r2.get_json()
    assert js["ingested"] == []
    assert js["skipped_unchanged"] == 1
