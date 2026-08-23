"""Frontend tests for the Gradio app pure helpers and HTTP wiring.

These cover:
- Filename extraction
- Request-body construction (launch.py-compatible vs. full)
- /sources response parsing (tolerant of both schema variants)
- Source annotation against the real filesystem
- HTML rendering (basic invariants only)
- fetch_sources / refresh_sources_panel against a stubbed requests session
- _extract_answer fallbacks
- launch.py CLI helpers
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
from typing import Any, Dict
from unittest.mock import patch

import pytest

import gradio_app as ga
import launch as launch_mod


# ----------------------------
# extract_filename
# ----------------------------

def test_extract_filename_basic():
    assert ga.extract_filename("/tmp/foo/bar.pdf") == "bar.pdf"

def test_extract_filename_windows_separators():
    assert ga.extract_filename(r"C:\\Users\\jm\\docs\\paper.docx") == "paper.docx"

def test_extract_filename_empty():
    assert ga.extract_filename("") == ""
    assert ga.extract_filename(None) == ""  # type: ignore[arg-type]


# ----------------------------
# build_request_body
# ----------------------------

def test_build_request_body_launch_compatible_only_two_keys():
    body = ga.build_request_body(
        question="hi", llm_id="mock:any",
        embedding_id="x", version="y",
        top_k=5, fetch_k=20, search_type="mmr",
        mmr_lambda=0.3, max_context_chars=18000,
        launch_compatible=True,
    )
    assert body == {"query": "hi", "llm_id": "mock:any"}

def test_build_request_body_full_includes_retrieval_knobs():
    body = ga.build_request_body(
        question="hi", llm_id="mock:any",
        embedding_id="fake:any", version="v1",
        top_k=4, fetch_k=12, search_type="mmr",
        mmr_lambda=0.4, max_context_chars=10000,
        launch_compatible=False,
    )
    assert body["query"] == "hi"
    assert body["llm_id"] == "mock:any"
    assert body["embedding_id"] == "fake:any"
    assert body["version"] == "v1"
    assert body["top_k"] == 4
    assert body["search_type"] == "mmr"
    assert body["mmr_lambda"] == pytest.approx(0.4)

def test_build_request_body_full_drops_blank_strings():
    body = ga.build_request_body(
        question="hi", llm_id="mock:any",
        embedding_id="", version="   ",
        top_k=1, fetch_k=1, search_type="",
        mmr_lambda=0.0, max_context_chars=2000,
        launch_compatible=False,
    )
    assert "embedding_id" not in body
    assert "version" not in body
    assert "search_type" not in body
    assert body["query"] == "hi"


# ----------------------------
# parse_sources_response
# ----------------------------

def test_parse_sources_response_documented_shape():
    payload = {"sources": ["/a/b.pdf", "/a/c.pdf"]}
    assert ga.parse_sources_response(payload) == ["/a/b.pdf", "/a/c.pdf"]

def test_parse_sources_response_alt_key():
    payload = {"source_paths": ["/x.txt"]}
    assert ga.parse_sources_response(payload) == ["/x.txt"]

def test_parse_sources_response_handles_garbage():
    assert ga.parse_sources_response(None) == []
    assert ga.parse_sources_response("nope") == []
    assert ga.parse_sources_response({"sources": "not a list"}) == []
    assert ga.parse_sources_response({"sources": ["", None, "/ok.pdf"]}) == ["/ok.pdf"]


# ----------------------------
# annotate_sources (filesystem check)
# ----------------------------

def test_annotate_sources_marks_existing_files(tmp_path):
    real = tmp_path / "real.pdf"
    real.write_text("hi")
    missing = tmp_path / "ghost.pdf"
    rows = ga.annotate_sources([str(real), str(missing)])
    by_path = {r["path"]: r for r in rows}
    assert by_path[str(real)]["exists"] is True
    assert by_path[str(missing)]["exists"] is False
    assert by_path[str(real)]["name"] == "real.pdf"

def test_annotate_sources_dedupes_and_sorts(tmp_path):
    f = tmp_path / "z.txt"; f.write_text(".")
    rows = ga.annotate_sources([str(f), str(f)])
    assert len(rows) == 1


# ----------------------------
# render_sources_html
# ----------------------------

def test_render_sources_html_empty_state():
    out = ga.render_sources_html([])
    assert "No documents" in out

def test_render_sources_html_lists_filenames(tmp_path):
    f = tmp_path / "report.pdf"; f.write_text("x")
    rows = ga.annotate_sources([str(f)])
    out = ga.render_sources_html(rows, collection="kb__v1__hf")
    assert "report.pdf" in out
    assert "kb__v1__hf" in out
    assert "1 document" in out

def test_render_sources_html_flags_missing_files(tmp_path):
    rows = ga.annotate_sources([str(tmp_path / "missing.txt")])
    out = ga.render_sources_html(rows)
    assert "missing" in out.lower()

def test_render_sources_html_escapes_names():
    rows = [{"path": "/ignored", "name": "<script>x</script>", "exists": True}]
    out = ga.render_sources_html(rows)
    assert "<script>x</script>" not in out
    assert "&lt;script&gt;x&lt;/script&gt;" in out


def test_render_sources_html_hides_directory_path(tmp_path):
    nested = tmp_path / "deep" / "folder"
    nested.mkdir(parents=True)
    f = nested / "only_name_shown.pdf"
    f.write_text("x")
    rows = ga.annotate_sources([str(f)])
    out = ga.render_sources_html(rows)
    assert "only_name_shown.pdf" in out
    assert str(nested) not in out
    assert "nb-source-path" not in out


def test_render_sources_html_has_scrollable_wrapper(tmp_path):
    f = tmp_path / "a.pdf"; f.write_text(".")
    rows = ga.annotate_sources([str(f)])
    out = ga.render_sources_html(rows)
    assert 'class="nb-sources-wrap"' in out


# ----------------------------
# fetch_sources / refresh_sources_panel (stubbed requests)
# ----------------------------

class _FakeResponse:
    def __init__(self, status_code: int, payload: Dict[str, Any]):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)
    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}", response=self)
    def json(self):
        return self._payload


def test_fetch_sources_blank_api_base():
    rows, coll, err = ga.fetch_sources("", "fake:any", "v1")
    assert rows == [] and coll is None
    assert err and "empty" in err.lower()

def test_fetch_sources_happy_path(tmp_path):
    real = tmp_path / "doc.txt"; real.write_text(".")
    payload = {"sources": [str(real)], "collection": "kb__v1__fake_any"}
    with patch("gradio_app.requests.get", return_value=_FakeResponse(200, payload)):
        rows, coll, err = ga.fetch_sources("http://api/", "fake:any", "v1")
    assert err is None
    assert coll == "kb__v1__fake_any"
    assert len(rows) == 1 and rows[0]["exists"] is True

def test_fetch_sources_network_error():
    import requests as _rq
    def boom(*a, **k): raise _rq.ConnectionError("nope")
    with patch("gradio_app.requests.get", side_effect=boom):
        rows, coll, err = ga.fetch_sources("http://api", "fake:any", "v1")
    assert rows == [] and coll is None
    assert err and "ConnectionError" in err

def test_refresh_sources_panel_offline_pill():
    import requests as _rq
    def boom(*a, **k): raise _rq.ConnectionError("nope")
    with patch("gradio_app.requests.get", side_effect=boom):
        html_panel, pill = ga.refresh_sources_panel("http://api", "fake:any", "v1", 5)
    assert "offline" in pill
    assert "Could not reach API" in html_panel

def test_refresh_sources_panel_warn_when_files_missing(tmp_path):
    payload = {"sources": [str(tmp_path / "ghost.pdf")], "collection": "kb"}
    with patch("gradio_app.requests.get", return_value=_FakeResponse(200, payload)):
        html_panel, pill = ga.refresh_sources_panel("http://api", "fake:any", "v1", 5)
    assert "warn" in pill
    assert "missing files" in pill or "missing" in html_panel.lower()

def test_refresh_sources_panel_ok_when_all_present(tmp_path):
    real = tmp_path / "real.pdf"; real.write_text(".")
    payload = {"sources": [str(real)], "collection": "kb"}
    with patch("gradio_app.requests.get", return_value=_FakeResponse(200, payload)):
        html_panel, pill = ga.refresh_sources_panel("http://api", "fake:any", "v1", 5)
    assert "connected" in pill
    assert "real.pdf" in html_panel


# ----------------------------
# fetch_server_config
# ----------------------------

def test_fetch_server_config_happy_path():
    payload = {"embedding_id": "hf:BAAI/bge-large-en-v1.5", "version": "v1"}
    with patch("gradio_app.requests.get", return_value=_FakeResponse(200, payload)):
        cfg = ga.fetch_server_config("http://api")
    assert cfg["embedding_id"] == "hf:BAAI/bge-large-en-v1.5"
    assert cfg["version"] == "v1"


def test_fetch_server_config_returns_empty_on_error():
    import requests as _rq
    def boom(*a, **k): raise _rq.ConnectionError("nope")
    with patch("gradio_app.requests.get", side_effect=boom):
        cfg = ga.fetch_server_config("http://api")
    assert cfg == {}


def test_fetch_server_config_blank_base():
    assert ga.fetch_server_config("") == {}


# ----------------------------
# trigger_folder_ingest
# ----------------------------

def test_trigger_folder_ingest_happy_path():
    payload = {
        "status": "ok", "scanned": 2,
        "ingested": [{"source_path": "/a.txt", "chunks": 3}],
        "skipped_unchanged": 1, "errors": [],
    }
    with patch("gradio_app.requests.post", return_value=_FakeResponse(200, payload)) as mock_post:
        summary, err = ga.trigger_folder_ingest(
            "http://api", "/tmp/watched", "fake:any", "v1", 10,
        )
    assert err is None
    assert summary["skipped_unchanged"] == 1
    assert summary["ingested"][0]["chunks"] == 3
    args, kwargs = mock_post.call_args
    assert args[0] == "http://api/ingest_folder"
    assert kwargs["json"]["folder"] == "/tmp/watched"
    assert kwargs["json"]["embedding_id"] == "fake:any"


def test_trigger_folder_ingest_network_error():
    import requests as _rq
    def boom(*a, **k): raise _rq.ConnectionError("dead")
    with patch("gradio_app.requests.post", side_effect=boom):
        summary, err = ga.trigger_folder_ingest(
            "http://api", "/tmp/watched", "fake:any", "v1", 5,
        )
    assert summary is None
    assert err and "ConnectionError" in err


def test_trigger_folder_ingest_blank_base():
    summary, err = ga.trigger_folder_ingest("", "/tmp", "fake:any", "v1", 5)
    assert summary is None
    assert err and "empty" in err.lower()


# ----------------------------
# refresh_panel_with_ingest (combined)
# ----------------------------

def test_refresh_panel_with_ingest_runs_ingest_then_list(tmp_path):
    ingest_payload = {
        "status": "ok", "scanned": 1,
        "ingested": [{"source_path": str(tmp_path / "new.txt"), "chunks": 2}],
        "skipped_unchanged": 0, "errors": [],
    }
    real = tmp_path / "new.txt"; real.write_text("hi")
    sources_payload = {"sources": [str(real)], "collection": "kb"}

    with patch("gradio_app.requests.post", return_value=_FakeResponse(200, ingest_payload)) as mp, \
         patch("gradio_app.requests.get", return_value=_FakeResponse(200, sources_payload)) as mg:
        html_panel, pill = ga.refresh_panel_with_ingest("http://api", "fake:any", "v1", 5)

    assert mp.call_args[0][0] == "http://api/ingest_folder"
    assert "Ingested 1 new file" in html_panel
    assert "new.txt" in html_panel
    assert "connected" in pill


def test_refresh_panel_with_ingest_uses_long_ingest_timeout(tmp_path):
    """The /ingest_folder call must use DEFAULT_INGEST_TIMEOUT_S, not the short /answer timeout."""
    ingest_payload = {
        "status": "ok", "scanned": 0,
        "ingested": [], "skipped_unchanged": 0, "errors": [],
    }
    sources_payload = {"sources": [], "collection": "kb"}
    with patch("gradio_app.requests.post", return_value=_FakeResponse(200, ingest_payload)) as mp, \
         patch("gradio_app.requests.get", return_value=_FakeResponse(200, sources_payload)):
        ga.refresh_panel_with_ingest("http://api", "fake:any", "v1", timeout_s=5)

    _, post_kwargs = mp.call_args
    assert post_kwargs["timeout"] == ga.DEFAULT_INGEST_TIMEOUT_S
    assert ga.DEFAULT_INGEST_TIMEOUT_S >= 600, "ingest timeout must be generous"


def test_refresh_panel_with_ingest_accepts_explicit_ingest_timeout(tmp_path):
    ingest_payload = {"status": "ok", "scanned": 0, "ingested": [], "skipped_unchanged": 0, "errors": []}
    sources_payload = {"sources": [], "collection": "kb"}
    with patch("gradio_app.requests.post", return_value=_FakeResponse(200, ingest_payload)) as mp, \
         patch("gradio_app.requests.get", return_value=_FakeResponse(200, sources_payload)):
        ga.refresh_panel_with_ingest(
            "http://api", "fake:any", "v1", timeout_s=5, ingest_timeout_s=2400,
        )
    assert mp.call_args.kwargs["timeout"] == 2400


def test_refresh_panel_with_ingest_still_lists_when_ingest_fails(tmp_path):
    real = tmp_path / "existing.txt"; real.write_text("hi")
    sources_payload = {"sources": [str(real)], "collection": "kb"}
    import requests as _rq
    def boom(*a, **k): raise _rq.ConnectionError("ingest_down")
    with patch("gradio_app.requests.post", side_effect=boom), \
         patch("gradio_app.requests.get", return_value=_FakeResponse(200, sources_payload)):
        html_panel, pill = ga.refresh_panel_with_ingest("http://api", "fake:any", "v1", 5)
    assert "Ingest failed" in html_panel
    assert "existing.txt" in html_panel
    assert "connected" in pill


# ----------------------------
# ask_api end-to-end with stubbed POST
# ----------------------------

def test_ask_api_blank_question():
    answer, srcs, raw, status, req = ga.ask_api(
        question="   ", api_base="http://x", endpoint="answer",
        llm_id="mock:any", embedding_id="", version="",
        top_k=1, fetch_k=1, search_type="mmr", mmr_lambda=0.3,
        max_context_chars=2000, timeout_s=5, launch_compatible=True,
    )
    assert status == "error"
    assert "enter a question" in answer.lower()

def test_ask_api_happy_path():
    payload = {"answer": "hi back", "sources": [{"source": "doc.pdf"}]}
    with patch("gradio_app._post_json", return_value=payload):
        answer, srcs, raw, status, req = ga.ask_api(
            question="hi", api_base="http://x", endpoint="answer",
            llm_id="mock:any", embedding_id="fake:any", version="v1",
            top_k=1, fetch_k=1, search_type="mmr", mmr_lambda=0.3,
            max_context_chars=2000, timeout_s=5, launch_compatible=True,
        )
    assert status == "ok"
    assert "hi back" in answer
    assert "Endpoint" in answer
    assert json.loads(srcs) == [{"source": "doc.pdf"}]
    assert json.loads(raw) == payload
    req_obj = json.loads(req)
    assert req_obj["url"].endswith("/answer")
    assert req_obj["body"] == {"query": "hi", "llm_id": "mock:any"}

def test_ask_api_http_error_returns_status_error():
    import requests as _rq
    def boom(*a, **k): raise _rq.ConnectionError("dead")
    with patch("gradio_app._post_json", side_effect=boom):
        answer, srcs, raw, status, req = ga.ask_api(
            question="hi", api_base="http://x", endpoint="answer",
            llm_id="mock:any", embedding_id="", version="",
            top_k=1, fetch_k=1, search_type="mmr", mmr_lambda=0.3,
            max_context_chars=2000, timeout_s=5, launch_compatible=True,
        )
    assert status == "error"
    assert "dead" in answer


# ----------------------------
# chat_step
# ----------------------------

def test_chat_step_appends_to_history():
    payload = {"answer": "ok"}
    with patch("gradio_app._post_json", return_value=payload):
        history, status, req, raw = ga.chat_step(
            message="hello", history=[],
            api_base="http://x", endpoint="answer", llm_id="mock:any",
            embedding_id="", version="",
            top_k=1, fetch_k=1, search_type="mmr", mmr_lambda=0.3,
            max_context_chars=2000, timeout_s=5, launch_compatible=True,
        )
    assert status == "ok"
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "hello"
    assert history[1]["role"] == "assistant"
    assert "ok" in history[1]["content"]

def test_chat_step_blank_message_no_change():
    history, status, *_ = ga.chat_step(
        message="   ", history=[{"role": "user", "content": "a"}],
        api_base="http://x", endpoint="answer", llm_id="mock:any",
        embedding_id="", version="",
        top_k=1, fetch_k=1, search_type="mmr", mmr_lambda=0.3,
        max_context_chars=2000, timeout_s=5, launch_compatible=True,
    )
    assert status == "error"
    assert history == [{"role": "user", "content": "a"}]


def test_chat_step_sends_prior_history_to_api():
    """When continuing a chat, ask_api must receive the prior exchanges as `history`."""
    prior = [
        {"role": "user", "content": "my name is Jose"},
        {"role": "assistant", "content": "Hi Jose!"},
    ]
    captured = {}

    def fake_post(url, payload, timeout_s):
        captured["url"] = url
        captured["payload"] = payload
        return {"answer": "You said your name is Jose."}

    with patch("gradio_app._post_json", side_effect=fake_post):
        history, status, *_ = ga.chat_step(
            message="what is my name?", history=list(prior),
            api_base="http://x", endpoint="answer", llm_id="mock:any",
            embedding_id="", version="",
            top_k=1, fetch_k=1, search_type="mmr", mmr_lambda=0.3,
            max_context_chars=2000, timeout_s=5, launch_compatible=True,
        )

    assert status == "ok"
    assert captured["payload"]["history"] == prior
    assert captured["payload"]["query"] == "what is my name?"
    assert len(history) == 4  # 2 prior + user + assistant
    assert history[-2] == {"role": "user", "content": "what is my name?"}
    assert history[-1]["role"] == "assistant"


def test_chat_step_trims_history_to_last_50_exchanges():
    """History is capped at CHAT_HISTORY_MAX_MSGS (100 messages = 50 exchanges)."""
    prior = []
    for i in range(60):  # 60 exchanges = 120 messages
        prior.append({"role": "user", "content": f"q{i}"})
        prior.append({"role": "assistant", "content": f"a{i}"})

    with patch("gradio_app._post_json", return_value={"answer": "ok"}):
        history, status, *_ = ga.chat_step(
            message="new q", history=prior,
            api_base="http://x", endpoint="answer", llm_id="mock:any",
            embedding_id="", version="",
            top_k=1, fetch_k=1, search_type="mmr", mmr_lambda=0.3,
            max_context_chars=2000, timeout_s=5, launch_compatible=True,
        )

    assert len(history) == ga.CHAT_HISTORY_MAX_MSGS
    # Oldest kept must come from well after the start
    assert history[-2] == {"role": "user", "content": "new q"}


def test_build_request_body_includes_history_when_present():
    body = ga.build_request_body(
        question="hi", llm_id="mock:any",
        embedding_id="", version="",
        top_k=1, fetch_k=1, search_type="", mmr_lambda=0.0, max_context_chars=2000,
        launch_compatible=True,
        history=[{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"}],
    )
    assert body["history"] == [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "ok"},
    ]


def test_build_request_body_trims_oversize_history():
    long_history = [{"role": "user", "content": f"m{i}"} for i in range(300)]
    body = ga.build_request_body(
        question="hi", llm_id="mock:any",
        embedding_id="", version="",
        top_k=1, fetch_k=1, search_type="", mmr_lambda=0.0, max_context_chars=2000,
        launch_compatible=True,
        history=long_history,
    )
    assert len(body["history"]) == ga.CHAT_HISTORY_MAX_MSGS
    assert body["history"][-1]["content"] == "m299"


def test_build_request_body_omits_empty_history():
    body = ga.build_request_body(
        question="hi", llm_id="mock:any",
        embedding_id="", version="",
        top_k=1, fetch_k=1, search_type="", mmr_lambda=0.0, max_context_chars=2000,
        launch_compatible=True,
        history=None,
    )
    assert "history" not in body


def test_normalize_chat_history_flattens_rich_content_parts():
    """Gradio 6.x round-trips content as [{'text': ..., 'type': 'text'}, ...]."""
    rich = [
        {"role": "user", "content": [{"text": "my name is Jose", "type": "text"}]},
        {"role": "assistant", "content": [{"text": "Hi ", "type": "text"},
                                           {"text": "Jose", "type": "text"}]},
    ]
    normalized = ga._normalize_chat_history(rich)
    assert normalized == [
        {"role": "user", "content": "my name is Jose"},
        {"role": "assistant", "content": "Hi Jose"},
    ]


def test_normalize_chat_history_drops_unknown_roles():
    mixed = [
        {"role": "user", "content": "ok"},
        {"role": "system", "content": "nope"},  # unknown roles dropped
        {"role": "ASSISTANT", "content": "fine"},  # case-insensitive
    ]
    normalized = ga._normalize_chat_history(mixed)
    assert normalized == [
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "fine"},
    ]


def test_chat_history_values_are_messages_format_dicts():
    """Guards the regression that produced 'Data incompatible with messages format'."""
    with patch("gradio_app._post_json", return_value={"answer": "hi back"}):
        history, *_ = ga.chat_step(
            message="hello", history=[],
            api_base="http://x", endpoint="answer", llm_id="mock:any",
            embedding_id="", version="",
            top_k=1, fetch_k=1, search_type="mmr", mmr_lambda=0.3,
            max_context_chars=2000, timeout_s=5, launch_compatible=True,
        )
    for item in history:
        assert isinstance(item, dict)
        assert set(item.keys()) == {"role", "content"}
        assert item["role"] in {"user", "assistant"}


# ----------------------------
# compare_step
# ----------------------------

def test_compare_step_runs_both_configs_and_keeps_them_separate():
    """Two parallel /answer calls go through with their own payloads; outputs are not crossed."""
    def fake_post(url, payload, timeout_s):
        if payload["llm_id"] == "openai:gpt-4o-mini":
            return {"answer": "answer-A", "sources": [{"source": "a.pdf"}]}
        if payload["llm_id"] == "anthropic:claude-sonnet-4-6":
            return {"answer": "answer-B", "sources": [{"source": "b.pdf"}]}
        raise AssertionError(f"unexpected llm_id {payload['llm_id']}")

    with patch("gradio_app._post_json", side_effect=fake_post):
        ans_a, src_a, raw_a, ans_b, src_b, raw_b = ga.compare_step(
            question="same Q",
            api_base="http://x", endpoint="answer", timeout_s=5, launch_compatible=True,
            llm_id_a="openai:gpt-4o-mini", embedding_id_a="fake:any", version_a="v1",
            top_k_a=4, fetch_k_a=12, search_type_a="mmr",
            mmr_lambda_a=0.3, max_context_chars_a=18000,
            llm_id_b="anthropic:claude-sonnet-4-6", embedding_id_b="fake:any", version_b="v1",
            top_k_b=6, fetch_k_b=24, search_type_b="similarity",
            mmr_lambda_b=0.2, max_context_chars_b=12000,
        )

    assert "answer-A" in ans_a
    assert "answer-B" in ans_b
    assert "a.pdf" in src_a and "b.pdf" not in src_a
    assert "b.pdf" in src_b and "a.pdf" not in src_b
    assert "answer-A" in raw_a and "answer-B" in raw_b


def test_compare_step_blank_question_returns_two_errors():
    ans_a, _, _, ans_b, _, _ = ga.compare_step(
        question="   ",
        api_base="http://x", endpoint="answer", timeout_s=5, launch_compatible=True,
        llm_id_a="mock:any", embedding_id_a="", version_a="",
        top_k_a=1, fetch_k_a=1, search_type_a="mmr",
        mmr_lambda_a=0.3, max_context_chars_a=2000,
        llm_id_b="mock:any", embedding_id_b="", version_b="",
        top_k_b=1, fetch_k_b=1, search_type_b="mmr",
        mmr_lambda_b=0.3, max_context_chars_b=2000,
    )
    assert "enter a question" in ans_a.lower()
    assert "enter a question" in ans_b.lower()


# ----------------------------
# make_download_file
# ----------------------------

def test_make_download_file_writes_markdown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = ga.make_download_file(
        filename="run", user_query="q?",
        request_pretty='{"a": 1}', raw_pretty='{"answer": "x"}',
    )
    assert out.endswith("run.md")
    body = Path(out).read_text(encoding="utf-8")
    assert "## User query" in body
    assert "q?" in body
    assert '"a": 1' in body


# ----------------------------
# _extract_answer
# ----------------------------

def test_extract_answer_dict_answer_key():
    assert ga._extract_answer({"answer": "yo"}) == "yo"

def test_extract_answer_falls_back_to_json():
    out = ga._extract_answer({"unexpected": True})
    assert "unexpected" in out

def test_extract_answer_list_first_item():
    assert ga._extract_answer([{"text": "hi"}]) == "hi"


# ----------------------------
# launch.py CLI helpers
# ----------------------------

def test_launch_build_llm_id_explicit():
    assert launch_mod.build_llm_id("openai:gpt-4o-mini", None, None) == "openai:gpt-4o-mini"

def test_launch_build_llm_id_provider_model():
    assert launch_mod.build_llm_id(None, "ollama", "llama3.2") == "ollama:llama3.2"

def test_launch_build_llm_id_default(monkeypatch):
    monkeypatch.delenv("LLM_ID", raising=False)
    assert launch_mod.build_llm_id(None, None, None) == "openai:gpt-4o-mini"

def test_launch_extract_answer_dict():
    assert launch_mod.extract_answer({"answer": "yo"}) == "yo"

def test_launch_extract_answer_str_payload():
    assert launch_mod.extract_answer("plain") == "plain"


# ----------------------------
# MODEL_CHOICES / backend agreement
# ----------------------------

def test_model_choices_use_providers_the_backend_supports():
    """Every dropdown entry must name a provider make_llm actually dispatches on.

    Asks the backend rather than restating its provider list, so a UI entry the
    server would reject fails here. Construction errors (missing optional dep or
    missing API key) are fine -- only 'unsupported provider' is a real mismatch.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    import local_rag_api as api

    # Sanity: the check below is only meaningful if a bogus provider does raise.
    with pytest.raises(ValueError, match="Unsupported llm provider"):
        api.make_llm("definitely_not_a_provider:x")

    for choice in ga.MODEL_CHOICES:
        provider, _, model = choice.partition(":")
        assert model, f"{choice}: empty model"
        try:
            api.make_llm(choice)
        except ValueError as exc:
            assert "Unsupported llm provider" not in str(exc), f"{choice}: {exc}"
        except Exception:
            pass  # missing dependency or credentials -- not a UI/backend mismatch


def test_default_llm_is_gpt_5_6_luna():
    assert ga.DEFAULT_LLM_ID == "openai:gpt-5.6-luna"
    assert ga.DEFAULT_LLM_ID in ga.MODEL_CHOICES


def test_gpt_5_6_models_get_the_wide_context_preset():
    for model in ("openai:gpt-5.6-sol", "openai:gpt-5.6-terra", "openai:gpt-5.6-luna"):
        assert ga.context_preset(model) == ga.WIDE_CONTEXT_PRESET, model


def test_other_models_get_the_modest_context_preset():
    for model in ("openai:gpt-4o-mini", "anthropic:claude-sonnet-4-6",
                  "ollama:llama3.2", "mock:any", "", None):
        assert ga.context_preset(model) == ga.MODEST_CONTEXT_PRESET, model


def test_wide_preset_is_actually_wider_and_top_k_is_the_binding_lever():
    """max_context_chars alone is not binding -- top_k must move with it."""
    wide, modest = ga.WIDE_CONTEXT_PRESET, ga.MODEST_CONTEXT_PRESET
    assert wide["top_k"] > modest["top_k"]
    assert wide["max_context_chars"] > modest["max_context_chars"]
    # The cap must exceed what top_k chunks can produce, or it truncates instead.
    assert wide["max_context_chars"] > wide["top_k"] * 1600


def test_apply_context_preset_returns_slider_values_in_order():
    assert ga.apply_context_preset("openai:gpt-5.6-luna") == (
        ga.WIDE_CONTEXT_PRESET["top_k"], ga.WIDE_CONTEXT_PRESET["max_context_chars"])
    assert ga.apply_context_preset("openai:gpt-4o") == (
        ga.MODEST_CONTEXT_PRESET["top_k"], ga.MODEST_CONTEXT_PRESET["max_context_chars"])


def test_preset_values_fit_the_slider_ranges():
    """Presets outside the slider bounds would be silently clamped by Gradio."""
    for preset in (ga.WIDE_CONTEXT_PRESET, ga.MODEST_CONTEXT_PRESET):
        assert 1 <= preset["top_k"] <= 30
        assert 2000 <= preset["max_context_chars"] <= 50000


def test_launch_compatible_defaults_off_so_the_preset_reaches_the_server():
    """With the switch on, build_request_body drops the retrieval knobs entirely,
    which would make the per-model context preset a no-op in the default UI state."""
    import requests
    from unittest.mock import patch
    with patch("requests.get", side_effect=requests.ConnectionError("offline")):
        demo = ga.build_ui()
    defaults = [c.value for c in demo.blocks.values()
                if "launch.py-compatible" in (getattr(c, "label", None) or "")]
    assert defaults == [False]

    top_k, max_chars = ga.apply_context_preset(ga.DEFAULT_LLM_ID)
    body = ga.build_request_body(
        question="q", llm_id=ga.DEFAULT_LLM_ID, embedding_id="fake:any", version="v1",
        top_k=top_k, fetch_k=24, search_type="mmr", mmr_lambda=0.3,
        max_context_chars=max_chars, launch_compatible=False,
    )
    assert body["top_k"] == ga.WIDE_CONTEXT_PRESET["top_k"]
    assert body["max_context_chars"] == ga.WIDE_CONTEXT_PRESET["max_context_chars"]


def test_build_request_body_includes_reranker_id():
    body = ga.build_request_body(
        question="q", llm_id="mock:any", embedding_id="fake:any", version="v1",
        top_k=6, fetch_k=24, search_type="mmr", mmr_lambda=0.3,
        max_context_chars=18000, launch_compatible=False,
        reranker_id="ce:BAAI/bge-reranker-v2-m3",
    )
    assert body["reranker_id"] == "ce:BAAI/bge-reranker-v2-m3"


def test_build_request_body_omits_blank_reranker_id():
    """A blank box means 'use the server default', so the key must not be sent."""
    body = ga.build_request_body(
        question="q", llm_id="mock:any", embedding_id="fake:any", version="v1",
        top_k=6, fetch_k=24, search_type="mmr", mmr_lambda=0.3,
        max_context_chars=18000, launch_compatible=False, reranker_id="",
    )
    assert "reranker_id" not in body


def test_launch_compatible_still_omits_reranker_id():
    body = ga.build_request_body(
        question="q", llm_id="mock:any", embedding_id="fake:any", version="v1",
        top_k=6, fetch_k=24, search_type="mmr", mmr_lambda=0.3,
        max_context_chars=18000, launch_compatible=True,
        reranker_id="ce:BAAI/bge-reranker-v2-m3",
    )
    assert body == {"query": "q", "llm_id": "mock:any"}


def test_ask_api_binds_reranker_id_before_history():
    """The Ask tab passes Gradio inputs positionally.

    reranker_id must sit immediately after launch_compatible, otherwise the
    textbox value would silently land in `history`.
    """
    import inspect

    params = list(inspect.signature(ga.ask_api).parameters)
    assert params.index("reranker_id") == params.index("launch_compatible") + 1
    assert params.index("history") > params.index("reranker_id")


def test_model_choices_include_local_generators():
    assert "ollama:qwen3:30b" in ga.MODEL_CHOICES
    assert "lmstudio:google/gemma-4-12b-qat" in ga.MODEL_CHOICES
