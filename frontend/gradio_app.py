from __future__ import annotations

"""Gradio frontend for the Local RAG Flask API.

Layout is inspired by NotebookLM: a left "Sources" panel that lists every
file already ingested in the active Chroma collection, and a right side that
holds the Q&A / Chat experience. The connection settings live in a single
collapsed accordion so the main view stays clean.
"""

import html
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import gradio as gr


# ----------------------------
# Defaults
# ----------------------------
DEFAULT_API_BASE = os.environ.get("RAG_API", "http://127.0.0.1:5050")
DEFAULT_ENDPOINT = os.environ.get("RAG_ENDPOINT", "answer")
DEFAULT_EMBEDDING_ID = os.environ.get("EMBEDDING_ID", "tfidf:local")
DEFAULT_RERANKER_ID = os.environ.get("RERANKER_ID", "none:")
DEFAULT_VERSION = os.environ.get("CHROMA_COLLECTION_VERSION", "v1")
DEFAULT_LLM_ID = os.environ.get("LLM_ID", "openai:gpt-4o-mini")
DEFAULT_TIMEOUT_S = int(os.environ.get("RAG_TIMEOUT", "120"))
# Folder ingest can take several minutes for many/large PDFs + real embeddings.
# Use a much larger ceiling than /answer, configurable via env var.
DEFAULT_INGEST_TIMEOUT_S = int(os.environ.get("RAG_INGEST_TIMEOUT", "1800"))

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WATCHED_FOLDER = os.environ.get("WATCHED_FOLDER", str(_REPO_ROOT / "watched_folder"))

MODEL_CHOICES = [
    "openai:gpt-4o-mini",
    "openai:gpt-4o",
    "openai:gpt-4.1-mini",
    "gemini:gemini-2.5-flash",
    "anthropic:claude-sonnet-4-6",
    "anthropic:claude-haiku-4-5",
    "anthropic:claude-opus-4-7",
    # Local generators served by Ollama.
    "ollama:qwen3:30b",
    "ollama:gpt-oss:20b",
    "ollama:gemma4:31b-mlx",
    "ollama:llama3.2",
    # Local generators served by LM Studio (needs its server started).
    "lmstudio:google/gemma-4-12b-qat",
    "lmstudio:qwen/qwen3.6-27b",
    "lmstudio:mistralai/magistral-small-2509",
    "mock:any",
]

# Chat memory: keep the last N exchanges (1 exchange = 1 user msg + 1 assistant msg).
CHAT_HISTORY_EXCHANGES = 50
CHAT_HISTORY_MAX_MSGS = CHAT_HISTORY_EXCHANGES * 2


# ----------------------------
# CSS (NotebookLM-inspired light theme)
# ----------------------------
CUSTOM_CSS = """
:root {
  --nb-bg:        #ffffff;
  --nb-surface:   #f8f9fa;
  --nb-surface-2: #f1f3f4;
  --nb-border:    #e8eaed;
  --nb-text:      #202124;
  --nb-muted:     #5f6368;
  --nb-primary:   #1a73e8;
  --nb-primary-2: #1967d2;
  --nb-success:   #137333;
  --nb-danger:    #c5221f;
  --nb-warning:   #b06000;
}

body, .gradio-container {
  background: var(--nb-bg) !important;
  color: var(--nb-text) !important;
  font-family: "Google Sans", "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
}

.gradio-container .prose, .gradio-container label, .gradio-container span,
.gradio-container h1, .gradio-container h2, .gradio-container h3,
.gradio-container p, .gradio-container li {
  color: var(--nb-text) !important;
}

.nb-header {
  padding: 18px 24px 14px 24px;
  border-bottom: 1px solid var(--nb-border);
  background: var(--nb-bg);
}
.nb-header h1 {
  margin: 0;
  font-weight: 500;
  font-size: 22px;
  letter-spacing: -0.2px;
}
.nb-header .nb-sub {
  margin-top: 2px;
  color: var(--nb-muted);
  font-size: 13px;
}

.nb-card {
  background: var(--nb-bg) !important;
  border: 1px solid var(--nb-border) !important;
  border-radius: 12px !important;
  padding: 16px !important;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}

.nb-side {
  background: var(--nb-surface) !important;
  border: 1px solid var(--nb-border) !important;
  border-radius: 12px !important;
  padding: 14px !important;
}

.nb-section-title {
  font-size: 13px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  color: var(--nb-muted);
  margin: 0 0 10px 0;
}

/* Sources list */
.nb-sources-wrap {
  max-height: 70vh;
  overflow-y: scroll;
  padding-right: 8px;
  scrollbar-gutter: stable;
}
.nb-sources-wrap::-webkit-scrollbar { width: 10px; }
.nb-sources-wrap::-webkit-scrollbar-thumb {
  background: #c8ccd1;
  border-radius: 8px;
}
.nb-sources-wrap::-webkit-scrollbar-thumb:hover { background: #aab0b7; }
.nb-sources-wrap::-webkit-scrollbar-track {
  background: var(--nb-surface-2);
  border-radius: 8px;
}
.nb-source {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 10px 12px;
  margin-bottom: 8px;
  border: 1px solid var(--nb-border);
  border-radius: 10px;
  background: var(--nb-bg);
  transition: background 0.15s ease;
}
.nb-source:hover { background: var(--nb-surface-2); }
.nb-source-icon {
  flex: 0 0 auto;
  width: 28px;
  height: 28px;
  border-radius: 6px;
  background: var(--nb-surface-2);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  color: var(--nb-muted);
  font-weight: 600;
}
.nb-source-body { flex: 1 1 auto; min-width: 0; }
.nb-source-name {
  font-size: 13px;
  font-weight: 500;
  color: var(--nb-text);
  word-break: break-word;
}
.nb-source-path {
  font-size: 11px;
  color: var(--nb-muted);
  margin-top: 2px;
  word-break: break-all;
}
.nb-status-dot {
  flex: 0 0 auto;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-top: 10px;
}
.nb-status-ok { background: var(--nb-success); }
.nb-status-missing { background: var(--nb-danger); }

.nb-empty {
  padding: 24px 12px;
  text-align: center;
  color: var(--nb-muted);
  font-size: 13px;
  border: 1px dashed var(--nb-border);
  border-radius: 10px;
  background: var(--nb-bg);
}

.nb-counts {
  font-size: 12px;
  color: var(--nb-muted);
  margin: 8px 0 12px 0;
}

/* Inputs */
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  border-radius: 8px !important;
  border: 1px solid var(--nb-border) !important;
  background: var(--nb-bg) !important;
  color: var(--nb-text) !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus,
.gradio-container select:focus {
  outline: none !important;
  border-color: var(--nb-primary) !important;
  box-shadow: 0 0 0 3px rgba(26, 115, 232, 0.15) !important;
}

/* Buttons */
button.primary, button.lg.primary, .gradio-container button.primary {
  background: var(--nb-primary) !important;
  color: #ffffff !important;
  border: 1px solid var(--nb-primary) !important;
  border-radius: 999px !important;
  padding: 8px 18px !important;
  font-weight: 500 !important;
}
button.primary:hover, .gradio-container button.primary:hover {
  background: var(--nb-primary-2) !important;
  border-color: var(--nb-primary-2) !important;
}
.gradio-container button.secondary {
  background: var(--nb-bg) !important;
  color: var(--nb-text) !important;
  border: 1px solid var(--nb-border) !important;
  border-radius: 999px !important;
  padding: 6px 14px !important;
}

/* Chat bubbles */
.gradio-container .message {
  border-radius: 12px !important;
  border: 1px solid var(--nb-border) !important;
}
.gradio-container .message.user {
  background: var(--nb-surface-2) !important;
  color: var(--nb-text) !important;
}
.gradio-container .message.bot {
  background: var(--nb-bg) !important;
  color: var(--nb-text) !important;
}

/* Tabs */
.gradio-container .tab-nav button {
  border-radius: 8px 8px 0 0 !important;
  font-weight: 500 !important;
}

/* Status pill */
.nb-status-pill {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 500;
  border: 1px solid var(--nb-border);
  background: var(--nb-surface);
  color: var(--nb-muted);
}
.nb-status-pill.ok { background: #e6f4ea; color: var(--nb-success); border-color: #b7dfc1; }
.nb-status-pill.warn { background: #fef7e0; color: var(--nb-warning); border-color: #fde293; }
.nb-status-pill.err { background: #fce8e6; color: var(--nb-danger); border-color: #f4c7c3; }
"""


# ----------------------------
# Pure helpers (unit-tested)
# ----------------------------

def extract_filename(source_path: str) -> str:
    """Return just the filename portion of a path (cross-platform)."""
    if not source_path:
        return ""
    return os.path.basename(source_path.replace("\\", "/"))


def build_request_body(
    question: str,
    llm_id: str,
    embedding_id: str,
    version: str,
    top_k: int,
    fetch_k: int,
    search_type: str,
    mmr_lambda: float,
    max_context_chars: int,
    launch_compatible: bool,
    history: Optional[List[Dict[str, str]]] = None,
    reranker_id: str = "",
) -> Dict[str, Any]:
    """Construct the JSON body sent to /answer.

    When launch_compatible is True, only `query`, `llm_id`, and (if non-empty)
    `history` are sent. Otherwise all retrieval knobs go too, minus blank values.
    """
    trimmed_history = (history or [])[-CHAT_HISTORY_MAX_MSGS:]

    if launch_compatible:
        body: Dict[str, Any] = {"query": question, "llm_id": (llm_id or "").strip()}
        if trimmed_history:
            body["history"] = trimmed_history
        return body

    body = {
        "query": question,
        "llm_id": (llm_id or "").strip(),
        "embedding_id": (embedding_id or "").strip(),
        "version": (version or "").strip(),
        "top_k": int(top_k),
        "fetch_k": int(fetch_k),
        "search_type": (search_type or "").strip(),
        "mmr_lambda": float(mmr_lambda),
        "max_context_chars": int(max_context_chars),
        "reranker_id": (reranker_id or "").strip(),
    }
    if trimmed_history:
        body["history"] = trimmed_history
    return {k: v for k, v in body.items() if v not in ("", None, [])}


def parse_sources_response(payload: Any) -> List[str]:
    """Extract the list of source paths from a /sources response.

    Tolerates both the documented shape (`{"sources": [...]}`) and the
    folder_watcher's expected shape (`{"source_paths": [...]}`).
    """
    if not isinstance(payload, dict):
        return []
    raw = payload.get("sources")
    if raw is None:
        raw = payload.get("source_paths")
    if not isinstance(raw, list):
        return []
    return [s for s in raw if isinstance(s, str) and s]


def annotate_sources(paths: List[str]) -> List[Dict[str, Any]]:
    """Decorate raw source paths with name + on-disk existence."""
    out: List[Dict[str, Any]] = []
    for p in sorted(set(paths)):
        out.append({
            "path": p,
            "name": extract_filename(p) or p,
            "exists": Path(p).is_file(),
        })
    return out


def render_sources_html(sources: List[Dict[str, Any]], collection: Optional[str] = None) -> str:
    """Build the HTML shown inside the Sources panel."""
    if not sources:
        return (
            '<div class="nb-empty">No documents ingested yet.<br>'
            'Drop files into <code>watched_folder/</code> or POST to '
            '<code>/ingest</code>.</div>'
        )
    total = len(sources)
    on_disk = sum(1 for s in sources if s.get("exists"))
    missing = total - on_disk
    bits: List[str] = []
    counts = f'{total} document{"s" if total != 1 else ""} · {on_disk} on disk'
    if missing:
        counts += f' · <span style="color:var(--nb-danger)">{missing} missing</span>'
    if collection:
        counts += f' · <span style="color:var(--nb-muted)">collection: {html.escape(collection)}</span>'
    bits.append(f'<div class="nb-counts">{counts}</div>')
    bits.append('<div class="nb-sources-wrap">')
    for s in sources:
        ext = (os.path.splitext(s["name"])[1] or "•").lstrip(".").upper()[:4] or "FILE"
        dot = "nb-status-ok" if s["exists"] else "nb-status-missing"
        title = "File present on disk" if s["exists"] else "File missing on disk"
        bits.append(
            '<div class="nb-source">'
            f'  <div class="nb-source-icon">{html.escape(ext)}</div>'
            '  <div class="nb-source-body">'
            f'    <div class="nb-source-name">{html.escape(s["name"])}</div>'
            '  </div>'
            f'  <div class="nb-status-dot {dot}" title="{title}"></div>'
            '</div>'
        )
    bits.append('</div>')
    return "\n".join(bits)


def _extract_answer(payload: Any) -> str:
    """Best-effort extraction of an answer string from a RAG API response."""
    if isinstance(payload, dict):
        for k in ("answer", "result", "text", "message", "content"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return v
        if "data" in payload:
            return _extract_answer(payload["data"])
        return json.dumps(payload, indent=2, ensure_ascii=False)
    if isinstance(payload, list):
        if not payload:
            return "<empty list response>"
        return _extract_answer(payload[0])
    return str(payload)


# ----------------------------
# IO helpers (call the API)
# ----------------------------

def _post_json(url: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def fetch_sources(
    api_base: str,
    embedding_id: str,
    version: str,
    timeout_s: int = 30,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    """Call GET /sources and return (annotated_sources, collection, error)."""
    api_base = (api_base or "").strip().rstrip("/")
    if not api_base:
        return [], None, "API base URL is empty."
    try:
        params = {
            "embedding_id": (embedding_id or "").strip(),
            "version": (version or "").strip(),
        }
        params = {k: v for k, v in params.items() if v}
        r = requests.get(f"{api_base}/sources", params=params, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return [], None, f"{type(e).__name__}: {e}"
    paths = parse_sources_response(data)
    return annotate_sources(paths), (data.get("collection") if isinstance(data, dict) else None), None


def refresh_sources_panel(
    api_base: str,
    embedding_id: str,
    version: str,
    timeout_s: int,
) -> Tuple[str, str]:
    """Returns (panel_html, status_pill_html)."""
    sources, collection, err = fetch_sources(api_base, embedding_id, version, timeout_s)
    if err is not None:
        return (
            f'<div class="nb-empty">Could not reach API: {html.escape(err)}</div>',
            '<span class="nb-status-pill err">offline</span>',
        )
    pill = '<span class="nb-status-pill ok">connected</span>'
    if any(not s["exists"] for s in sources):
        pill = '<span class="nb-status-pill warn">missing files on disk</span>'
    return render_sources_html(sources, collection=collection), pill


def fetch_server_config(api_base: str, timeout_s: int = 2) -> Dict[str, Any]:
    """Probe GET /config so the UI can align with the running backend.

    Returns {} on any failure so the caller falls back to env/hardcoded defaults.
    """
    api_base = (api_base or "").strip().rstrip("/")
    if not api_base:
        return {}
    try:
        r = requests.get(f"{api_base}/config", timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def trigger_folder_ingest(
    api_base: str,
    folder: str,
    embedding_id: str,
    version: str,
    timeout_s: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Call POST /ingest_folder. Returns (summary_dict, error_str)."""
    api_base = (api_base or "").strip().rstrip("/")
    if not api_base:
        return None, "API base URL is empty."
    body = {
        "folder": (folder or "").strip(),
        "embedding_id": (embedding_id or "").strip(),
        "version": (version or "").strip(),
    }
    body = {k: v for k, v in body.items() if v}
    try:
        r = requests.post(f"{api_base}/ingest_folder", json=body, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    return data if isinstance(data, dict) else {}, None


def _format_ingest_banner(summary: Optional[Dict[str, Any]], err: Optional[str]) -> str:
    if err:
        return f'<div class="nb-counts" style="color:var(--nb-danger)">Ingest failed: {html.escape(err)}</div>'
    if not summary:
        return ""
    ingested = summary.get("ingested") or []
    skipped = int(summary.get("skipped_unchanged") or 0)
    errors = summary.get("errors") or []
    total_chunks = sum(int(item.get("chunks") or 0) for item in ingested)
    plural = "s" if len(ingested) != 1 else ""
    base = html.escape(
        f"Ingested {len(ingested)} new file{plural} · {total_chunks} chunks · {skipped} unchanged skipped"
    )
    if errors:
        base += f' · <span style="color:var(--nb-danger)">{len(errors)} error(s)</span>'
    return f'<div class="nb-counts">{base}</div>'


def refresh_panel_with_ingest(
    api_base: str,
    embedding_id: str,
    version: str,
    timeout_s: int,
    ingest_timeout_s: Optional[int] = None,
) -> Tuple[str, str]:
    """Trigger a folder ingest (watched_folder) then re-list /sources.

    ``timeout_s`` is used for the short /sources call. ``ingest_timeout_s``
    governs /ingest_folder, which may take minutes on large PDFs with real
    embedding models; it defaults to DEFAULT_INGEST_TIMEOUT_S.
    """
    ingest_to = int(ingest_timeout_s) if ingest_timeout_s else DEFAULT_INGEST_TIMEOUT_S
    summary, err = trigger_folder_ingest(
        api_base=api_base,
        folder=DEFAULT_WATCHED_FOLDER,
        embedding_id=embedding_id,
        version=version,
        timeout_s=ingest_to,
    )
    panel_html, pill = refresh_sources_panel(api_base, embedding_id, version, timeout_s)
    banner = _format_ingest_banner(summary, err)
    return (banner + panel_html) if banner else panel_html, pill


def ask_api(
    question: str,
    api_base: str,
    endpoint: str,
    llm_id: str,
    embedding_id: str,
    version: str,
    top_k: int,
    fetch_k: int,
    search_type: str,
    mmr_lambda: float,
    max_context_chars: int,
    timeout_s: int,
    launch_compatible: bool,
    # reranker_id precedes history so the Ask tab's positional Gradio inputs
    # bind to it; history is always passed by keyword.
    reranker_id: str = "",
    history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, str, str, str, str]:
    """Returns (answer_md, sources_pretty_json, raw_pretty_json, status, request_pretty_json)."""
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", "[]", "{}", "error", "{}"

    api_base = (api_base or "").strip().rstrip("/")
    endpoint = (endpoint or "").strip().lstrip("/")
    if not api_base:
        return "API base URL is empty.", "[]", "{}", "error", "{}"
    if not endpoint:
        return "Endpoint is empty.", "[]", "{}", "error", "{}"
    url = f"{api_base}/{endpoint}"

    body = build_request_body(
        question=question,
        llm_id=llm_id,
        embedding_id=embedding_id,
        version=version,
        top_k=top_k,
        fetch_k=fetch_k,
        search_type=search_type,
        mmr_lambda=mmr_lambda,
        max_context_chars=max_context_chars,
        launch_compatible=launch_compatible,
        history=history,
        reranker_id=reranker_id,
    )
    request_pretty = json.dumps({"url": url, "body": body}, indent=2, ensure_ascii=False)

    t0 = time.time()
    try:
        data = _post_json(url, body, timeout_s=int(timeout_s))
        dt = time.time() - t0
    except requests.HTTPError as e:
        resp_text = ""
        if e.response is not None:
            try:
                resp_text = e.response.text
            except Exception:
                pass
        msg = f"HTTP error: {e}\n\nServer response:\n{resp_text}" if resp_text else f"HTTP error: {e}"
        return f"```text\n{msg}\n```", "[]", "{}", "error", request_pretty
    except Exception as e:
        return f"```text\nRequest failed: {e}\n```", "[]", "{}", "error", request_pretty

    answer_text = _extract_answer(data)
    sources = data.get("sources", []) if isinstance(data, dict) else []
    sources_pretty = json.dumps(sources, indent=2, ensure_ascii=False)
    raw_pretty = json.dumps(data, indent=2, ensure_ascii=False)
    header = f"**Endpoint:** `{url}`  ·  **Latency:** `{dt:.2f}s`"
    return f"{header}\n\n{answer_text}", sources_pretty, raw_pretty, "ok", request_pretty


def _flatten_content(content: Any) -> str:
    """Coerce a Chatbot message's ``content`` field into a plain string.

    Gradio 6.x returns rich-content as ``[{"text": ..., "type": "text"}, ...]``
    when the Chatbot is passed back in as an input. Recurse through that
    shape so the string we forward to the API is actually human text.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content:
            return _flatten_content(content["text"])
        if "value" in content:
            return _flatten_content(content["value"])
        return ""
    if isinstance(content, (list, tuple)):
        parts = [_flatten_content(p) for p in content]
        return "".join(p for p in parts if p)
    return str(content)


def _normalize_chat_history(history: Any) -> List[Dict[str, str]]:
    """Normalize mixed/legacy chatbot history into Gradio messages format."""
    out: List[Dict[str, str]] = []
    for item in (history or []):
        if isinstance(item, dict):
            role = str(item.get("role", "")).strip().lower()
            content = _flatten_content(item.get("content"))
            if role in {"user", "assistant"}:
                out.append({"role": role, "content": content})
            continue
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append({"role": "user", "content": _flatten_content(item[0])})
            out.append({"role": "assistant", "content": _flatten_content(item[1])})
    return out


def chat_step(
    message: str,
    history: List[Dict[str, str]],
    api_base: str,
    endpoint: str,
    llm_id: str,
    embedding_id: str,
    version: str,
    top_k: int,
    fetch_k: int,
    search_type: str,
    mmr_lambda: float,
    max_context_chars: int,
    timeout_s: int,
    launch_compatible: bool,
    reranker_id: str = "",
) -> Tuple[List[Dict[str, str]], str, str, str]:
    message = (message or "").strip()
    history = _normalize_chat_history(history)
    if not message:
        return history, "error", "{}", "{}"
    answer_md, _, raw_pretty, status, request_pretty = ask_api(
        question=message,
        api_base=api_base, endpoint=endpoint, llm_id=llm_id,
        embedding_id=embedding_id, version=version,
        top_k=top_k, fetch_k=fetch_k, search_type=search_type,
        mmr_lambda=mmr_lambda, max_context_chars=max_context_chars,
        timeout_s=timeout_s, launch_compatible=launch_compatible,
        history=history, reranker_id=reranker_id,
    )
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer_md})
    if len(history) > CHAT_HISTORY_MAX_MSGS:
        history = history[-CHAT_HISTORY_MAX_MSGS:]
    return history, status, request_pretty, raw_pretty


def compare_step(
    question: str,
    api_base: str,
    endpoint: str,
    timeout_s: int,
    launch_compatible: bool,
    # Config A
    llm_id_a: str, embedding_id_a: str, version_a: str,
    top_k_a: int, fetch_k_a: int, search_type_a: str,
    mmr_lambda_a: float, max_context_chars_a: int,
    # Config B
    llm_id_b: str, embedding_id_b: str, version_b: str,
    top_k_b: int, fetch_k_b: int, search_type_b: str,
    mmr_lambda_b: float, max_context_chars_b: int,
    reranker_id_a: str = "", reranker_id_b: str = "",
) -> Tuple[str, str, str, str, str, str]:
    """Run the same question against two configs in parallel.

    Returns (answer_a, sources_a_json, raw_a_json, answer_b, sources_b_json, raw_b_json).
    Parallelism keeps the perceived latency comparable when the user is judging
    speed alongside answer quality.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _one(llm_id, embedding_id, version, top_k, fetch_k, search_type, mmr_lambda, max_context_chars, reranker_id):
        return ask_api(
            question=question,
            api_base=api_base, endpoint=endpoint, llm_id=llm_id,
            embedding_id=embedding_id, version=version,
            top_k=top_k, fetch_k=fetch_k, search_type=search_type,
            mmr_lambda=mmr_lambda, max_context_chars=max_context_chars,
            timeout_s=timeout_s, launch_compatible=launch_compatible,
            history=None, reranker_id=reranker_id,
        )

    with ThreadPoolExecutor(max_workers=2) as ex:
        fa = ex.submit(_one, llm_id_a, embedding_id_a, version_a, top_k_a, fetch_k_a, search_type_a, mmr_lambda_a, max_context_chars_a, reranker_id_a)
        fb = ex.submit(_one, llm_id_b, embedding_id_b, version_b, top_k_b, fetch_k_b, search_type_b, mmr_lambda_b, max_context_chars_b, reranker_id_b)
        ans_a, src_a, raw_a, _, _ = fa.result()
        ans_b, src_b, raw_b, _, _ = fb.result()

    return ans_a, src_a, raw_a, ans_b, src_b, raw_b


def make_download_file(filename: str, user_query: str, request_pretty: str, raw_pretty: str) -> str:
    """Write the last exchange to a Markdown file and return its path."""
    filename = (filename or "rag_exchange.md").strip()
    filename = os.path.basename(filename)
    if not filename.lower().endswith((".md", ".markdown")):
        filename = os.path.splitext(filename)[0] + ".md"
    out_path = os.path.join(os.getcwd(), filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# RAG exchange\n\n")
        f.write("## User query\n\n```text\n" + ((user_query or "").strip() or "<empty>") + "\n```\n\n")
        f.write("## Request sent to Flask API\n\n```json\n" + (request_pretty or "{}") + "\n```\n\n")
        f.write("## Response from Flask API\n\n```json\n" + (raw_pretty or "{}") + "\n```\n")
    return out_path


# ----------------------------
# UI
# ----------------------------

def make_theme() -> gr.themes.Soft:
    return gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    )


def build_ui() -> gr.Blocks:
    # Align defaults with the running backend so the Sources panel points at
    # the collection the API and folder_watcher are actually writing to.
    srv = fetch_server_config(DEFAULT_API_BASE, timeout_s=2)
    initial_embedding_id = srv.get("embedding_id") or DEFAULT_EMBEDDING_ID
    initial_version = srv.get("version") or DEFAULT_VERSION
    initial_reranker_id = srv.get("reranker_id") or DEFAULT_RERANKER_ID

    with gr.Blocks(title="Local RAG") as demo:
        with gr.Row(elem_classes=["nb-header"]):
            gr.HTML(
                '<div>'
                '  <h1>Local RAG</h1>'
                '  <div class="nb-sub">Ask questions over the documents you have ingested into your local Chroma collection.</div>'
                '</div>'
            )

        with gr.Row():
            # ---- Sources panel ----
            with gr.Column(scale=3, min_width=280):
                with gr.Group(elem_classes=["nb-side"]):
                    with gr.Row():
                        gr.HTML('<div class="nb-section-title">Sources</div>')
                        status_pill = gr.HTML('<span class="nb-status-pill">checking…</span>')
                    sources_html = gr.HTML(
                        '<div class="nb-empty">Loading…</div>',
                        elem_id="nb_sources",
                    )
                    refresh_btn = gr.Button("Refresh sources", variant="secondary", size="sm")

            # ---- Chat / Q&A ----
            with gr.Column(scale=7, min_width=420):
                with gr.Group(elem_classes=["nb-card"]):
                    with gr.Accordion("Connection & model", open=False):
                        with gr.Row():
                            api_base = gr.Textbox(
                                label="API base URL",
                                value=DEFAULT_API_BASE,
                                placeholder="http://127.0.0.1:5050",
                            )
                            endpoint = gr.Textbox(
                                label="Endpoint",
                                value=DEFAULT_ENDPOINT,
                                placeholder="answer",
                            )
                        with gr.Row():
                            llm_id = gr.Dropdown(
                                label="LLM (provider:model)",
                                choices=MODEL_CHOICES,
                                value=DEFAULT_LLM_ID,
                                allow_custom_value=True,
                                interactive=True,
                            )
                            launch_compatible = gr.Checkbox(
                                label="launch.py-compatible (only query + llm_id)",
                                value=True,
                            )
                        with gr.Row():
                            embedding_id = gr.Textbox(label="embedding_id", value=initial_embedding_id)
                            version = gr.Textbox(label="version", value=initial_version)
                        with gr.Row():
                            top_k = gr.Slider(1, 30, value=6, step=1, label="top_k")
                            fetch_k = gr.Slider(5, 80, value=24, step=1, label="fetch_k")
                        with gr.Row():
                            search_type = gr.Dropdown(
                                choices=["mmr", "similarity"], value="mmr", label="search_type",
                            )
                            mmr_lambda = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="mmr_lambda")
                        with gr.Row():
                            max_context_chars = gr.Slider(
                                2000, 50000, value=18000, step=500, label="max_context_chars",
                            )
                            timeout_s = gr.Slider(
                                5, 600, value=DEFAULT_TIMEOUT_S, step=5, label="timeout (s)",
                            )
                        with gr.Row():
                            reranker_id = gr.Textbox(
                                label="reranker_id", value=initial_reranker_id,
                                placeholder="none: or ce:BAAI/bge-reranker-v2-m3",
                            )

                last_query = gr.State("")
                last_request = gr.State("{}")
                last_raw = gr.State("{}")

                with gr.Tabs():
                    with gr.Tab("Ask"):
                        with gr.Group(elem_classes=["nb-card"]):
                            question = gr.Textbox(
                                label="Your question",
                                lines=3,
                                placeholder="What would you like to know about your documents?",
                            )
                            with gr.Row():
                                ask_btn = gr.Button("Ask", variant="primary")
                                clear_q_btn = gr.Button("Clear", variant="secondary")
                            status = gr.State("")
                            answer_md = gr.Markdown(elem_id="nb_answer")
                            with gr.Accordion("Cited chunks (JSON)", open=False):
                                sources_json = gr.Code(language="json")
                            with gr.Accordion("Raw API response (JSON)", open=False):
                                raw_json = gr.Code(language="json")

                        with gr.Group(elem_classes=["nb-card"]):
                            gr.HTML('<div class="nb-section-title">Download last exchange</div>')
                            with gr.Row():
                                filename = gr.Textbox(
                                    label="Filename",
                                    value="rag_exchange.md",
                                    placeholder="e.g. my_run.md",
                                    scale=4,
                                )
                                dl_btn = gr.Button("Prepare download", variant="secondary", scale=1)
                            dl_file = gr.File(label="Download", file_count="single")

                        dl_btn.click(
                            fn=make_download_file,
                            inputs=[filename, last_query, last_request, last_raw],
                            outputs=[dl_file],
                        )
                        clear_q_btn.click(fn=lambda: "", inputs=[], outputs=[question])

                        (
                            ask_btn.click(
                                fn=ask_api,
                                inputs=[
                                    question,
                                    api_base, endpoint, llm_id,
                                    embedding_id, version,
                                    top_k, fetch_k, search_type, mmr_lambda, max_context_chars,
                                    timeout_s, launch_compatible, reranker_id,
                                ],
                                outputs=[answer_md, sources_json, raw_json, status, last_request],
                            )
                            .then(fn=lambda q: q, inputs=[question], outputs=[last_query])
                            .then(fn=lambda raw: raw, inputs=[raw_json], outputs=[last_raw])
                        )

                    with gr.Tab("Chat"):
                        with gr.Group(elem_classes=["nb-card"]):
                            chatbot = gr.Chatbot(label="Conversation", height=440, value=[])
                            with gr.Row():
                                chat_msg = gr.Textbox(
                                    label="Message",
                                    placeholder="Ask something…",
                                    lines=2,
                                    scale=4,
                                )
                                send_btn = gr.Button("Send", variant="primary", scale=1)
                                clear_chat_btn = gr.Button("Clear", variant="secondary", scale=1)
                            chat_status = gr.State("")

                        (
                            send_btn.click(fn=lambda m: m, inputs=[chat_msg], outputs=[last_query])
                            .then(
                                fn=chat_step,
                                inputs=[
                                    chat_msg, chatbot,
                                    api_base, endpoint, llm_id,
                                    embedding_id, version,
                                    top_k, fetch_k, search_type, mmr_lambda, max_context_chars,
                                    timeout_s, launch_compatible, reranker_id,
                                ],
                                outputs=[chatbot, chat_status, last_request, last_raw],
                            )
                            .then(fn=lambda: "", inputs=[], outputs=[chat_msg])
                        )
                        clear_chat_btn.click(fn=lambda: [], inputs=[], outputs=[chatbot])

                    with gr.Tab("Compare"):
                        with gr.Group(elem_classes=["nb-card"]):
                            gr.HTML(
                                '<div class="nb-section-title">Side-by-side comparison</div>'
                                '<div class="nb-counts">Run the same question against two configs in parallel. '
                                'Defaults A → main settings, B → a second LLM. Override any field.</div>'
                            )
                            cmp_question = gr.Textbox(
                                label="Question",
                                lines=3,
                                placeholder="Same question goes to both columns…",
                            )
                            with gr.Row():
                                compare_btn = gr.Button("Compare", variant="primary")
                                clear_cmp_btn = gr.Button("Clear", variant="secondary")

                        with gr.Row():
                            # ---- Config A ----
                            with gr.Column():
                                with gr.Group(elem_classes=["nb-card"]):
                                    gr.HTML('<div class="nb-section-title">Config A</div>')
                                    llm_id_a = gr.Dropdown(
                                        label="LLM", choices=MODEL_CHOICES,
                                        value=DEFAULT_LLM_ID, allow_custom_value=True,
                                    )
                                    with gr.Row():
                                        embedding_id_a = gr.Textbox(label="embedding_id", value=initial_embedding_id)
                                        version_a = gr.Textbox(label="version", value=initial_version)
                                    with gr.Row():
                                        top_k_a = gr.Slider(1, 30, value=6, step=1, label="top_k")
                                        fetch_k_a = gr.Slider(5, 80, value=24, step=1, label="fetch_k")
                                    with gr.Row():
                                        search_type_a = gr.Dropdown(
                                            choices=["mmr", "similarity"], value="mmr", label="search_type",
                                        )
                                        mmr_lambda_a = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="mmr_lambda")
                                    max_context_chars_a = gr.Slider(
                                        2000, 50000, value=18000, step=500, label="max_context_chars",
                                    )
                                    reranker_id_a = gr.Textbox(
                                        label="reranker_id", value=initial_reranker_id,
                                        placeholder="none: or ce:BAAI/bge-reranker-v2-m3",
                                    )
                                    answer_a = gr.Markdown()
                                    with gr.Accordion("Cited chunks (JSON)", open=False):
                                        sources_a_json = gr.Code(language="json")
                                    with gr.Accordion("Raw API response (JSON)", open=False):
                                        raw_a_json = gr.Code(language="json")

                            # ---- Config B ----
                            with gr.Column():
                                with gr.Group(elem_classes=["nb-card"]):
                                    gr.HTML('<div class="nb-section-title">Config B</div>')
                                    llm_id_b = gr.Dropdown(
                                        label="LLM", choices=MODEL_CHOICES,
                                        value="anthropic:claude-sonnet-4-6", allow_custom_value=True,
                                    )
                                    with gr.Row():
                                        embedding_id_b = gr.Textbox(label="embedding_id", value=initial_embedding_id)
                                        version_b = gr.Textbox(label="version", value=initial_version)
                                    with gr.Row():
                                        top_k_b = gr.Slider(1, 30, value=6, step=1, label="top_k")
                                        fetch_k_b = gr.Slider(5, 80, value=24, step=1, label="fetch_k")
                                    with gr.Row():
                                        search_type_b = gr.Dropdown(
                                            choices=["mmr", "similarity"], value="mmr", label="search_type",
                                        )
                                        mmr_lambda_b = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="mmr_lambda")
                                    max_context_chars_b = gr.Slider(
                                        2000, 50000, value=18000, step=500, label="max_context_chars",
                                    )
                                    reranker_id_b = gr.Textbox(
                                        label="reranker_id", value=initial_reranker_id,
                                        placeholder="none: or ce:BAAI/bge-reranker-v2-m3",
                                    )
                                    answer_b = gr.Markdown()
                                    with gr.Accordion("Cited chunks (JSON)", open=False):
                                        sources_b_json = gr.Code(language="json")
                                    with gr.Accordion("Raw API response (JSON)", open=False):
                                        raw_b_json = gr.Code(language="json")

                        compare_btn.click(
                            fn=compare_step,
                            inputs=[
                                cmp_question, api_base, endpoint, timeout_s, launch_compatible,
                                llm_id_a, embedding_id_a, version_a,
                                top_k_a, fetch_k_a, search_type_a, mmr_lambda_a, max_context_chars_a,
                                llm_id_b, embedding_id_b, version_b,
                                top_k_b, fetch_k_b, search_type_b, mmr_lambda_b, max_context_chars_b,
                                reranker_id_a, reranker_id_b,
                            ],
                            outputs=[answer_a, sources_a_json, raw_a_json, answer_b, sources_b_json, raw_b_json],
                        )
                        clear_cmp_btn.click(fn=lambda: "", inputs=[], outputs=[cmp_question])

        # Wire up the Sources panel:
        # - refresh_btn: ingest any new files in watched_folder/ then re-list /sources.
        # - initial load: just list /sources (no implicit ingest).
        refresh_btn.click(
            fn=refresh_panel_with_ingest,
            inputs=[api_base, embedding_id, version, timeout_s],
            outputs=[sources_html, status_pill],
        )
        demo.load(
            fn=refresh_sources_panel,
            inputs=[api_base, embedding_id, version, timeout_s],
            outputs=[sources_html, status_pill],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(theme=make_theme(), css=CUSTOM_CSS)
