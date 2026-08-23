import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))

import os
import io
import re
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


def test_answer_reports_empty_content_instead_of_object_dump(client, tmp_path, monkeypatch):
    """A model that spends its whole budget on reasoning returns content=''.

    The old `content or str(resp)` fallback dumped the raw AIMessage repr, which
    hid the cause. Assert the reply names the token limit instead.
    """
    import local_rag_api as mod

    class _Resp:
        content = ""
        response_metadata = {"token_usage": {"completion_tokens": 6585, "total_tokens": 8192}}

    monkeypatch.setattr(mod, "make_llm", lambda _id: type("L", (), {"invoke": lambda s, m: _Resp()})())
    (tmp_path / "a.txt").write_text("acetic acid", encoding="utf-8")
    client.post("/ingest_folder", json={"folder": str(tmp_path), "embedding_id": "fake:any", "version": "vtest"})

    js = client.post("/answer", json={"query": "q", "llm_id": "lmstudio:x"}).get_json()
    assert "empty answer" in js["answer"]
    assert "8192" in js["answer"]
    assert "additional_kwargs" not in js["answer"]


def test_sources_pages_instead_of_one_huge_get(tmp_path):
    """Chroma's SQLite backend fails past ~32k bound variables.

    /sources must page rather than ask for everything at once, so simulate a
    collection that rejects large limits and assert it still returns all rows.
    """
    app = _make_app(tmp_path)
    import local_rag_api as mod

    class _Collection:
        MAX = 20000

        def __init__(self, n):
            self.rows = [{"source_path": f"/docs/f{i}.pdf"} for i in range(n)]
            self.max_limit_seen = 0

        def get(self, include=None, limit=None, offset=0, where=None):
            self.max_limit_seen = max(self.max_limit_seen, limit or 0)
            if (limit or 0) > self.MAX:
                raise RuntimeError("too many SQL variables")
            return {"metadatas": self.rows[offset:offset + limit]}

    class _Store:
        def __init__(self, coll):
            self._collection = coll

    coll = _Collection(45000)
    monkey = lambda *a, **k: (_Store(coll), None, "c", "chroma")
    original = mod.get_store
    mod.get_store = monkey
    try:
        r = app.test_client().get("/sources")
        assert r.status_code == 200
        assert len(r.get_json()["sources"]) == 45000
        assert coll.max_limit_seen <= _Collection.MAX
    finally:
        mod.get_store = original


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


def test_reranker_disabled_by_default(tmp_path):
    app = _make_app(tmp_path)  # noqa: F841
    import local_rag_api as mod
    import reranker

    assert mod.DEFAULT_RERANKER_ID == "none:"
    assert reranker.make_reranker("none:") is None
    assert reranker.make_reranker("") is None


def test_reranker_rejects_unknown_provider():
    import reranker

    with pytest.raises(ValueError, match="Unsupported reranker provider"):
        reranker.make_reranker("bogus:model")
    with pytest.raises(ValueError, match="requires a model name"):
        reranker.make_reranker("ce:")


def test_rerank_reorders_by_score_and_truncates():
    import reranker

    class _Doc:
        def __init__(self, text):
            self.page_content = text
            self.metadata = {}

    class _Stub:
        """Scores by position of the query term: later match -> lower score."""
        def predict(self, pairs):
            return [10.0 if "match" in text else 1.0 for _, text in pairs]

    docs = [_Doc("no"), _Doc("match here"), _Doc("no"), _Doc("match too")]
    out = reranker.rerank(_Stub(), "q", docs, top_k=2)
    assert [d.page_content for d in out] == ["match here", "match too"]

    # Disabled reranker is a pass-through that still honours top_k.
    assert reranker.rerank(None, "q", docs, top_k=3) == docs[:3]
    assert reranker.rerank(_Stub(), "q", [], top_k=3) == []


def test_retrieve_endpoint_echoes_reranker_id(client, tmp_path):
    (tmp_path / "a.txt").write_text("electric arc furnace dust limit", encoding="utf-8")
    client.post("/ingest_folder", json={
        "folder": str(tmp_path), "embedding_id": "fake:any", "version": "vtest",
    })
    r = client.post("/retrieve", json={"query": "dust limit", "top_k": 2})
    assert r.status_code == 200
    assert r.get_json()["reranker_id"] == "none:"


def test_config_exposes_reranker_id(client):
    assert client.get("/config").get_json()["reranker_id"] == "none:"


def test_config_reports_active_folder(client, tmp_path):
    js = client.get("/config").get_json()
    assert js["active_folder"] == js["watched_folder"]


def test_make_llm_lmstudio_targets_local_server(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key-must-not-leak")
    app = _make_app(tmp_path)  # noqa: F841
    import local_rag_api as mod

    if mod.ChatOpenAI is None:
        pytest.skip("langchain-openai not installed in this env")
    llm = mod.make_llm("lmstudio:google/gemma-4-12b-qat")
    assert llm.__class__.__name__ == "ChatOpenAI"
    assert llm.model_name == "google/gemma-4-12b-qat"
    assert "1234" in str(llm.openai_api_base)
    assert llm.openai_api_key.get_secret_value() == "lm-studio"


def test_make_embeddings_ollama_returns_ollama_embeddings(tmp_path):
    app = _make_app(tmp_path)  # noqa: F841
    import local_rag_api as mod

    if mod.OllamaEmbeddings is None:
        pytest.skip("langchain-ollama not installed in this env")
    emb = mod.make_embeddings("ollama:qllama/bge-m3:latest")
    assert emb.__class__.__name__ == "OllamaEmbeddings"
    assert emb.model == "qllama/bge-m3:latest"


def test_ollama_embedding_id_gets_its_own_collection(tmp_path):
    app = _make_app(tmp_path)  # noqa: F841
    import local_rag_api as mod

    ollama_name = mod._collection_name("ollama:qllama/bge-m3:latest", "v1")
    hf_name = mod._collection_name("hf:BAAI/bge-large-en-v1.5", "v1")
    assert ollama_name != hf_name
    # Chroma requires alphanumeric first/last char and no other punctuation.
    assert re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]+[a-zA-Z0-9]", ollama_name)


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


def test_retrieval_query_expands_anaphoric_followup():
    """A pronoun follow-up retrieves on the previous user turn, not the pronoun."""
    from local_rag_api import _retrieval_query

    history = [
        {"role": "user", "content": "What is acetone mainly used for?"},
        {"role": "assistant", "content": "As a solvent and chemical intermediate."},
    ]
    assert _retrieval_query("How is it produced industrially?", history) == (
        "What is acetone mainly used for? How is it produced industrially?"
    )


def test_retrieval_query_leaves_self_contained_followup_alone():
    """A follow-up naming its own subject must not be diluted by the prior turn."""
    from local_rag_api import _retrieval_query

    history = [
        {"role": "user", "content": "What is acetone mainly used for?"},
        {"role": "assistant", "content": "As a solvent."},
    ]
    q = "What is the boiling point of benzene?"
    assert _retrieval_query(q, history) == q
    assert _retrieval_query(q, []) == q


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


def test_retrieve_folder_filter_excludes_other_folders(client, tmp_path):
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


# ----------------------------
# sync_folder: three-way reconciliation
# ----------------------------

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


# ----------------------------
# /folders
# ----------------------------

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


# ----------------------------
# /watched_folder
# ----------------------------

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


def test_watch_delete_swallows_exceptions(client, monkeypatch):
    """A failed delete must not escape _watch_delete -- it runs on the single
    observer dispatcher thread, and watchdog only catches queue.Empty around
    dispatch, so any other exception would silently end all future watching."""
    import local_rag_api as mod

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "delete_by_source", _raise)

    mod._watch_delete("/nonexistent/does-not-matter.txt")


def test_sync_with_narrowed_extensions_does_not_delete_other_types(tmp_path):
    """`extensions` limits what gets ingested; it must not widen what gets deleted."""
    app = _make_app(tmp_path)
    import local_rag_api as mod

    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("text content about dust limits")
    (folder / "b.md").write_text("markdown content about dust limits")
    mod.sync_folder(str(folder), "fake:any", "vtest")

    res = mod.sync_folder(str(folder), "fake:any", "vtest", extensions=[".pdf"])

    assert res["deleted"] == []
    store, _, _, _ = mod.get_store("fake:any", "vtest")
    assert len(mod._sources_in(store, str(folder.resolve()))) == 2


def test_sync_deletes_only_within_the_requested_extensions(tmp_path):
    """A removed .txt is still deleted when the sync is scoped to .txt."""
    app = _make_app(tmp_path)
    import local_rag_api as mod

    folder = tmp_path / "docs"
    folder.mkdir()
    keep = folder / "keep.md"
    gone = folder / "gone.txt"
    keep.write_text("markdown that stays")
    gone.write_text("text that goes")
    mod.sync_folder(str(folder), "fake:any", "vtest")

    gone.unlink()
    res = mod.sync_folder(str(folder), "fake:any", "vtest", extensions=[".txt"])

    assert res["deleted"] == [str(gone.resolve())]
    store, _, _, _ = mod.get_store("fake:any", "vtest")
    assert mod._sources_in(store, str(folder.resolve())) == [str(keep.resolve())]


def test_watched_folder_default_is_resolved_like_chunk_metadata(tmp_path):
    """A symlinked watched folder must not make every scoped query miss.

    Chunks store folder = Path(source_path).resolve().parent, so the default
    watched folder has to be resolved the same way or the where-filter never
    matches and scoped answers come back silently empty.
    """
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    os.environ["WATCHED_FOLDER"] = str(link)
    try:
        app = _make_app(tmp_path)  # noqa: F841
        import local_rag_api as mod
        assert mod.WATCHED_FOLDER == str(real.resolve())
        assert mod.WATCHED_FOLDER == mod._folder_of(str(link / "any.txt"))
    finally:
        del os.environ["WATCHED_FOLDER"]


def test_watch_upsert_records_the_fingerprint_so_sync_does_not_re_embed(tmp_path):
    """The in-process watcher must update the state file like the old daemon did.

    Without it every later sync sees no fingerprint, treats the file as changed,
    and re-embeds the whole folder on each refresh.
    """
    app = _make_app(tmp_path)  # noqa: F841
    import local_rag_api as mod

    folder = tmp_path / "docs"
    folder.mkdir()
    f = folder / "a.txt"
    f.write_text("dust limit is 5 mg per normal cubic metre")

    mod._ACTIVE.update({"folder": str(folder.resolve()),
                        "embedding_id": "fake:any", "version": "vtest"})
    mod._watch_upsert(str(f.resolve()))

    res = mod.sync_folder(str(folder), "fake:any", "vtest")
    assert res["ingested"] == []
    assert res["skipped_unchanged"] == 1


def test_watch_upsert_skips_a_file_it_already_ingested_unchanged(tmp_path):
    """watchdog emits created AND modified for one write; the second is a no-op."""
    app = _make_app(tmp_path)  # noqa: F841
    import local_rag_api as mod

    folder = tmp_path / "docs"
    folder.mkdir()
    f = folder / "a.txt"
    f.write_text("dust limit is 5 mg per normal cubic metre")

    mod._ACTIVE.update({"folder": str(folder.resolve()),
                        "embedding_id": "fake:any", "version": "vtest"})

    calls = []
    real = mod._ingest_local_file

    def counting(path, emb, ver):
        calls.append(path)
        return real(path, emb, ver)

    mod._ingest_local_file = counting
    try:
        mod._watch_upsert(str(f.resolve()))
        mod._watch_upsert(str(f.resolve()))
    finally:
        mod._ingest_local_file = real

    assert len(calls) == 1
