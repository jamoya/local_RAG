from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Tuple

import requests
import gradio as gr


# ----------------------------
# Defaults (aligned with your existing client/server)
# ----------------------------
DEFAULT_API_BASE = os.environ.get("RAG_API", "http://127.0.0.1:5000")
DEFAULT_ENDPOINT = os.environ.get("RAG_ENDPOINT", "answer")

# local_rag_api.py defaults
DEFAULT_EMBEDDING_ID = os.environ.get("EMBEDDING_ID", "tfidf:local")
DEFAULT_VERSION = os.environ.get("CHROMA_COLLECTION_VERSION", "v1")
DEFAULT_LLM_ID = os.environ.get("LLM_ID", "openai:gpt-4o-mini")

DEFAULT_TIMEOUT_S = int(os.environ.get("RAG_TIMEOUT", "120"))

# Models known to work well with this Flask app (provider:model).
# Availability still depends on server-side API keys and installed providers.
MODEL_CHOICES = [
    "openai:gpt-4o-mini",
    "openai:gpt-4o",
    "openai:gpt-4.1-mini",
    "gemini:gemini-2.5-flash",
    "ollama:llama3.2",
    "mock:any",
]

# ----------------------------
# Pastel UI palette (requested)
# ----------------------------
PASTEL_DARK_BLUE = "#1e2b4a"   # app background
PASTEL_PANEL = "#27375c"      # cards/panels
PASTEL_LIGHT_BLUE = "#2f4472"  # replaces default whites
PASTEL_YELLOW = "#fff3b0"     # user-editable fields
PASTEL_GREEN = "#b7f0c2"      # API output fields
PASTEL_ORANGE = "#ffd3a6"     # dropdown background
TEXT_ON_DARK = "#eaf0ff"
BORDER = "#5b6b93"

CUSTOM_CSS = f"""
/* Base */
:root {{
  --app-bg: {PASTEL_DARK_BLUE};
  --panel-bg: {PASTEL_PANEL};
  --light-bg: {PASTEL_LIGHT_BLUE};
  --text: {TEXT_ON_DARK};
  --border: {BORDER};
  --input-bg: {PASTEL_YELLOW};
  --output-bg: {PASTEL_GREEN};
  --dropdown-bg: {PASTEL_ORANGE};
}}

body, .gradio-container {{
  background: var(--app-bg) !important;
  color: var(--text) !important;
}}

/* Kill remaining default whites */
.gradio-container, .gradio-container * {{
  --color-background: var(--app-bg);
}}
.gradio-container .wrap,
.gradio-container .block,
.gradio-container .container,
.gradio-container .gr-box,
.gradio-container .gr-form,
.gradio-container .gr-panel,
.gradio-container .gr-accordion,
.gradio-container .gr-accordion .label-wrap,
.gradio-container .gr-accordion .wrap,
.gradio-container .gr-accordion .content,
.gradio-container .gr-tabitem,
.gradio-container .gr-tab-nav,
.gradio-container .gr-tab-nav button,
.gradio-container .gr-tab-nav .tabitem,
.gradio-container .gr-markdown,
.gradio-container .gr-html,
.gradio-container footer {{
  background: var(--light-bg) !important;
  color: var(--text) !important;
}}

.gradio-container .prose, .gradio-container label, .gradio-container span,
.gradio-container h1, .gradio-container h2, .gradio-container h3, .gradio-container p, .gradio-container li {{
  color: var(--text) !important;
}}

/* Ensure readable text on light pastel blocks */
.user-input, .user-input * {{
  color: #0b1220 !important;
}}
.user-dropdown, .user-dropdown * {{
  color: #0b1220 !important;
}}
.api-output, .api-output * {{
  color: #0b1220 !important;
}}

#app_title {{
  background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 18px 18px 10px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}}

.panel {{
  background: var(--panel-bg) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 18px !important;
  padding: 14px !important;
  box-shadow: 0 10px 30px rgba(0,0,0,0.22);
}}

hr {{
  border-color: rgba(255,255,255,0.12);
}}

/* User-editable fields (pastel yellow) */
.user-input textarea,
.user-input input,
.user-input .wrap > textarea,
.user-input .wrap > input,
.user-input .gr-text-input,
.user-input .gr-number,
.user-input .gr-slider input[type="range"] {{
  background: var(--input-bg) !important;
  color: #0b1220 !important;
  border: 1px solid rgba(0,0,0,0.18) !important;
}}

.user-input input::placeholder,
.user-input textarea::placeholder {{
  color: rgba(11, 18, 32, 0.55) !important;
}}

/* Dropdown (pastel orange) */
.user-dropdown select,
.user-dropdown .wrap select,
.user-dropdown .gr-dropdown select {{
  background: var(--dropdown-bg) !important;
  color: #0b1220 !important;
  border: 1px solid rgba(0,0,0,0.18) !important;
}}

/* API output fields (pastel green) */
.api-output textarea,
.api-output input,
.api-output pre,
.api-output code,
.api-output .cm-editor,
.api-output .gr-code,
.api-output .prose {{
  background: var(--output-bg) !important;
  color: #0b1220 !important;
  border-radius: 14px !important;
  border: 1px solid rgba(0,0,0,0.16) !important;
}}

#answer_box {{
  background: var(--output-bg) !important;
  color: #0b1220 !important;
  border-radius: 14px;
  padding: 12px 14px;
  border: 1px solid rgba(0,0,0,0.16);
}}

/* Chat bubbles: keep contrast, avoid white */
.gradio-container .message {{
  border-radius: 14px !important;
}}
.gradio-container .message.user {{
  background: var(--input-bg) !important;
  color: #0b1220 !important;
}}
.gradio-container .message.bot {{
  background: var(--output-bg) !important;
  color: #0b1220 !important;
}}

/* Buttons */
button {{
  border-radius: 14px !important;
}}
button.primary {{
  box-shadow: 0 10px 22px rgba(0,0,0,0.20);
}}

/* Tabs */
.tabitem {{
  border-radius: 14px !important;
}}

/* Status icons */
#status_bar {{
  display: flex;
  gap: 10px;
  align-items: center;
  margin-top: 8px;
}}

.status_icon {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 44px;
  height: 44px;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.08);
  box-shadow: 0 10px 24px rgba(0,0,0,0.25);
  font-size: 24px;
}}

.spin {{
  animation: spin 0.9s linear infinite;
}}
@keyframes spin {{
  from {{ transform: rotate(0deg); }}
  to {{ transform: rotate(360deg); }}
}}

.boom {{
  animation: pop 0.35s ease-in-out infinite alternate;
}}
@keyframes pop {{
  from {{ transform: scale(1.00); }}
  to {{ transform: scale(1.12); }}
}}

/* Mouse-follow overlays (thumb + bomb) */
.mouse_overlay {{
  position: fixed;
  left: 0px;
  top: 0px;
  z-index: 999999;
  pointer-events: none;
  display: none;
}}

/* Mouse-follow (always visible when component itself is visible) */
.mouse_follow {{
  position: fixed;
  left: 0px;
  top: 0px;
  z-index: 999999;
  pointer-events: none;
}}

/* Strong contrast rules on light pastel blocks */
.api-output, .api-output * {{
  color: #0b1220 !important;
}}
"""


def _post_json(url: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    """POST JSON and return parsed JSON response."""
    resp = requests.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _extract_answer(payload: Any) -> str:
    """Try to extract a nice answer string from typical RAG API response shapes."""
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
        first = payload[0]
        if isinstance(first, dict):
            for k in ("answer", "context", "result", "text", "message"):
                v = first.get(k)
                if isinstance(v, str) and v.strip():
                    return v
            return json.dumps(first, indent=2, ensure_ascii=False)
        if isinstance(first, str):
            return first
        return json.dumps(first, indent=2, ensure_ascii=False)

    return str(payload)


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
) -> Tuple[str, str, str, str]:
    """
    Returns:
      - answer_text (markdown)
      - sources_pretty (json)
      - raw_pretty (json)
      - status ("ok" | "error")
      - request_pretty (json)
    """
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

    if launch_compatible:
        body: Dict[str, Any] = {"query": question, "llm_id": (llm_id or "").strip()}
    else:
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
        }
        body = {k: v for k, v in body.items() if v not in ("", None)}

    request_pretty = json.dumps({"url": url, "body": body}, indent=2, ensure_ascii=False)

    t0 = time.time()
    try:
        data = _post_json(url, body, timeout_s=int(timeout_s))
        dt = time.time() - t0
    except requests.HTTPError as e:
        resp_text = ""
        try:
            resp_text = e.response.text if e.response is not None else ""
        except Exception:
            pass
        msg = f"HTTP error: {e}\n\nServer response:\n{resp_text}" if resp_text else f"HTTP error: {e}"
        return f"```text\n{msg}\n```", "[]", "{}", "error", request_pretty if "request_pretty" in locals() else "{}"
    except Exception as e:
        return f"```text\nRequest failed: {e}\n```", "[]", "{}", "error", request_pretty if "request_pretty" in locals() else "{}"

    answer_text = _extract_answer(data)
    sources = data.get("sources", [])
    sources_pretty = json.dumps(sources, indent=2, ensure_ascii=False) if sources is not None else "[]"
    raw_pretty = json.dumps(data, indent=2, ensure_ascii=False)

    header = f"**Endpoint:** `{url}`  \n**Latency:** `{dt:.2f}s`"
    return f"{header}\n\n{answer_text}", sources_pretty, raw_pretty, "ok", request_pretty


def chat_step(
    message: str,
    history: List[Tuple[str, str]],
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
) -> Tuple[List[Tuple[str, str]], str, str, str]:
    """Append one turn to the chat."""
    message = (message or "").strip()
    if not message:
        return history, "error", "{}", "{}"

    answer_md, _, raw_pretty, status, request_pretty = ask_api(
        question=message,
        api_base=api_base,
        endpoint=endpoint,
        llm_id=llm_id,
        embedding_id=embedding_id,
        version=version,
        top_k=top_k,
        fetch_k=fetch_k,
        search_type=search_type,
        mmr_lambda=mmr_lambda,
        max_context_chars=max_context_chars,
        timeout_s=timeout_s,
        launch_compatible=launch_compatible,
    )
    history = list(history or [])
    history.append((message, answer_md))
    return history, status, request_pretty, raw_pretty



def make_download_file(filename: str, user_query: str, request_pretty: str, raw_pretty: str) -> str:
    """Create a downloadable Markdown file with the last query and API response."""
    filename = (filename or "").strip()
    if not filename:
        filename = "rag_exchange.md"

    filename = os.path.basename(filename)
    if not filename.lower().endswith((".md", ".markdown")):
        filename = os.path.splitext(filename)[0] + ".md"

    q = (user_query or "").strip()
    req_txt = request_pretty or "{}"
    resp_txt = raw_pretty or "{}"

    out_path = os.path.join(os.getcwd(), filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# RAG exchange\n\n")
        f.write("## User query\n\n")
        f.write("```text\n" + (q or "<empty>") + "\n```\n\n")
        f.write("## Request sent to Flask API\n\n")
        f.write("```json\n" + req_txt + "\n```\n\n")
        f.write("## Response from Flask API\n\n")
        f.write("```json\n" + resp_txt + "\n```\n")

    return out_path

def _show_hourglass():
    return (
        gr.update(visible=True),   # hourglass
        gr.update(visible=False),  # thumbs
        gr.update(visible=False),  # bomb
    )


def _show_result_icon(status: str):
    status = (status or "").lower()
    if status == "ok":
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
    )


# Install mouse tracking and keep the overlays pinned to the cursor.
# (We use Blocks.load(js=...) because <script> inside gr.HTML may be sanitized
# depending on Gradio version / settings.)
INIT_MOUSE_JS = r"""
() => {
  if (window.__rag_mouse_overlay_init__) return;
  window.__rag_mouse_overlay_init__ = true;
  window.__ragMouse = {x: 0, y: 0};

  const ids = ['hourglass_overlay','thumb_overlay','bomb_overlay'];

  const place = (id, x, y) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.left = (x + 16) + 'px';
    el.style.top  = (y + 16) + 'px';
  };

  document.addEventListener('mousemove', (e) => {
    window.__ragMouse = {x: e.clientX, y: e.clientY};
    for (const id of ids) place(id, e.clientX, e.clientY);
  }, {passive: true});

  // Also keep updating in case components appear/disappear without a mousemove.
  const tick = () => {
    const pos = window.__ragMouse || {x: 0, y: 0};
    for (const id of ids) place(id, pos.x, pos.y);
    window.requestAnimationFrame(tick);
  };
  window.requestAnimationFrame(tick);
}
"""


AUTO_HIDE_JS = r"""
() => {
  const thumb = document.getElementById('thumb_overlay');
  const bomb  = document.getElementById('bomb_overlay');
  const pos = window.__ragMouse || {x: 0, y: 0};

  const place = (el) => {
    if (!el) return;
    el.style.left = (pos.x + 16) + 'px';
    el.style.top  = (pos.y + 16) + 'px';
  };

  if (thumb) { thumb.style.display = ''; place(thumb); }
  if (bomb)  { bomb.style.display  = ''; place(bomb); }

  const isVisible = (el) => el && el.offsetParent !== null;

  if (isVisible(thumb)) {
    thumb.style.display = 'block';
    setTimeout(() => { if (thumb) thumb.style.display = 'none'; }, 4000);
  }
  if (isVisible(bomb)) {
    bomb.style.display = 'block';
    setTimeout(() => { if (bomb) bomb.style.display = 'none'; }, 2000);
  }
}
"""


POSITION_HOURGLASS_JS = r"""
() => {
  const hg = document.getElementById('hourglass_overlay');
  const pos = window.__ragMouse || {x: 0, y: 0};
  if (hg) {
    hg.style.left = (pos.x + 16) + 'px';
    hg.style.top  = (pos.y + 16) + 'px';
  }
}
"""


def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    )

    with gr.Blocks(title="Local RAG UI", theme=theme, css=CUSTOM_CSS) as demo:
        # Make sure mouse-follow overlays work reliably across Gradio versions.
        demo.load(fn=None, inputs=None, outputs=None, js=INIT_MOUSE_JS)
        with gr.Group(elem_classes=["panel"], elem_id="app_title"):
            gr.Markdown(
                "# Local RAG (Gradio)\n"
                "Ask a question and the app will POST to your local RAG Flask API."
            )

            with gr.Row():
                hourglass = gr.HTML(
                    value='<div id="hourglass_overlay" class="mouse_follow"><div class="status_icon"><span class="spin">⏳</span></div></div>',
                    visible=False,
                    elem_id="hourglass",
                )
                thumb = gr.HTML(
                    value='<div id="thumb_overlay" class="mouse_overlay"><div class="status_icon">👍</div></div>',
                    visible=False,
                    elem_id="thumb",
                )
                bomb = gr.HTML(
                    value='<div id="bomb_overlay" class="mouse_overlay"><div class="status_icon"><span class="boom">💥</span></div></div>',
                    visible=False,
                    elem_id="bomb",
                )

                # Mouse tracking for overlays (thumb + bomb)
                gr.HTML(
                    value="""
<script>
(function(){
  if (window.__rag_mouse_overlay_init__) return;
  window.__rag_mouse_overlay_init__ = true;
  window.__ragMouse = {x: 0, y: 0};

  function moveOverlay(id, x, y) {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.left = (x + 16) + 'px';
    el.style.top  = (y + 16) + 'px';
  }

  document.addEventListener('mousemove', (e) => {
    window.__ragMouse = {x: e.clientX, y: e.clientY};
    moveOverlay('hourglass_overlay', e.clientX, e.clientY);
    moveOverlay('thumb_overlay', e.clientX, e.clientY);
    moveOverlay('bomb_overlay', e.clientX, e.clientY);
  });
})();
</script>
""",
                    visible=True,
                )

        with gr.Group(elem_classes=["panel"]):
            gr.Markdown("## Connection & model")

            with gr.Row():
                api_base = gr.Textbox(
                    label="API base URL",
                    value=DEFAULT_API_BASE,
                    placeholder="http://127.0.0.1:5000",
                    elem_classes=["user-input"],
                )
                endpoint = gr.Textbox(
                    label="Endpoint",
                    value=DEFAULT_ENDPOINT,
                    placeholder="answer",
                    elem_classes=["user-input"],
                )

            with gr.Row():
                llm_id = gr.Dropdown(
                    label="LLM model (llm_id = provider:model)",
                    choices=MODEL_CHOICES,
                    value=DEFAULT_LLM_ID,
                    allow_custom_value=True,
                    interactive=True,
                    elem_classes=["user-input", "user-dropdown"],
                )
                launch_compatible = gr.Checkbox(
                    label="Launch.py-compatible request (send only query + llm_id)",
                    value=True,
                    elem_classes=["user-input"],
                )

        with gr.Accordion("Advanced retrieval options", open=False, elem_classes=["panel"]):
            with gr.Row():
                embedding_id = gr.Textbox(
                    label="embedding_id", value=DEFAULT_EMBEDDING_ID, elem_classes=["user-input"]
                )
                version = gr.Textbox(label="version", value=DEFAULT_VERSION, elem_classes=["user-input"])

            with gr.Row():
                top_k = gr.Slider(1, 30, value=6, step=1, label="top_k", elem_classes=["user-input"])
                fetch_k = gr.Slider(5, 80, value=24, step=1, label="fetch_k", elem_classes=["user-input"])

            with gr.Row():
                search_type = gr.Dropdown(
                    choices=["mmr", "similarity"],
                    value="mmr",
                    label="search_type",
                    elem_classes=["user-input", "user-dropdown"],
                )
                mmr_lambda = gr.Slider(
                    0.0, 1.0, value=0.3, step=0.05, label="mmr_lambda", elem_classes=["user-input"]
                )

            with gr.Row():
                max_context_chars = gr.Slider(
                    2000, 50000, value=18000, step=500, label="max_context_chars", elem_classes=["user-input"]
                )
                timeout_s = gr.Slider(
                    5, 600, value=DEFAULT_TIMEOUT_S, step=5, label="timeout (seconds)", elem_classes=["user-input"]
                )

        last_query = gr.State("")
        last_request = gr.State("{}")
        last_raw = gr.State("{}")

        with gr.Tabs():
            with gr.Tab("Simple Q&A"):
                question = gr.Textbox(
                    label="Your question",
                    lines=4,
                    placeholder="Type your question here…",
                    elem_classes=["user-input"],
                )
                ask_btn = gr.Button("Ask", variant="primary")

                status = gr.State("")

                answer_md = gr.Markdown(elem_id="answer_box", elem_classes=["api-output"])
                with gr.Accordion("Sources (JSON)", open=False):
                    sources_json = gr.Code(language="json", elem_classes=["api-output"])
                with gr.Accordion("Raw API response (JSON)", open=False):
                    raw_json = gr.Code(language="json", elem_classes=["api-output"])

                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("## Download last exchange")
                    filename = gr.Textbox(
                        label="Download filename",
                        value="rag_exchange.json",
                        placeholder="e.g. my_run.json",
                        elem_classes=["user-input"],
                    )
                    dl_btn = gr.Button("Prepare download", variant="secondary")
                    dl_file = gr.File(label="Download", file_count="single")

                dl_btn.click(
                    fn=make_download_file,
                    inputs=[filename, last_query, last_request, last_raw],
                    outputs=[dl_file],
                )

                (
                    ask_btn.click(fn=_show_hourglass, inputs=[], outputs=[hourglass, thumb, bomb])
                    .then(fn=lambda: None, inputs=[], outputs=[], js=POSITION_HOURGLASS_JS)
                    .then(
                        fn=ask_api,
                        inputs=[
                            question,
                            api_base, endpoint, llm_id,
                            embedding_id, version,
                            top_k, fetch_k, search_type, mmr_lambda, max_context_chars,
                            timeout_s,
                            launch_compatible,
                        ],
                        outputs=[answer_md, sources_json, raw_json, status, last_request],
                    )
                    .then(fn=lambda q: q, inputs=[question], outputs=[last_query])
                    .then(fn=lambda raw: raw, inputs=[raw_json], outputs=[last_raw])
                    .then(
                        fn=_show_result_icon,
                        inputs=[status],
                        outputs=[hourglass, thumb, bomb],
                        js=AUTO_HIDE_JS,
                    )
                )

            with gr.Tab("Chat"):
                gr.Markdown(
                    "Each message is sent as a fresh `query` to the API. "
                    "The status icons above reflect each request."
                )

                chatbot = gr.Chatbot(label="Conversation", height=420, elem_classes=["api-output"])

                with gr.Row():
                    chat_msg = gr.Textbox(
                        label="Message",
                        placeholder="Ask something…",
                        lines=2,
                        elem_classes=["user-input"],
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear", scale=1)

                chat_status = gr.State("")

                (
                    send_btn.click(fn=_show_hourglass, inputs=[], outputs=[hourglass, thumb, bomb])
                    .then(fn=lambda: None, inputs=[], outputs=[], js=POSITION_HOURGLASS_JS)
                    .then(fn=lambda m: m, inputs=[chat_msg], outputs=[last_query])
                    .then(
                        fn=chat_step,
                        inputs=[
                            chat_msg,
                            chatbot,
                            api_base, endpoint, llm_id,
                            embedding_id, version,
                            top_k, fetch_k, search_type, mmr_lambda, max_context_chars,
                            timeout_s,
                            launch_compatible,
                        ],
                        outputs=[chatbot, chat_status, last_request, last_raw],
                    )
                    .then(
                        fn=_show_result_icon,
                        inputs=[chat_status],
                        outputs=[hourglass, thumb, bomb],
                        js=AUTO_HIDE_JS,
                    )
                    .then(fn=lambda: "", inputs=[], outputs=[chat_msg])
                )

                clear_btn.click(fn=lambda: [], inputs=[], outputs=[chatbot])

        gr.Markdown(
            "### Notes\n"
            "- The app calls `POST {API_BASE}/{endpoint}` with JSON including `query` and `llm_id`.\n"
            "- Your API also supports `embedding_id`, `version`, `top_k`, `fetch_k`, `search_type`, `mmr_lambda`, `max_context_chars`."
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
