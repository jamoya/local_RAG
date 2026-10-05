"""Client for the Local RAG Flask API.

Choose the model used by the API with `llm_id`.

Supported (per local_rag_api.py):
  - openai:<model>   e.g. openai:gpt-4o-mini
  - gemini:<model>   e.g. gemini:gemini-1.5-flash
  - ollama:<model>
  - mock:<anything>

Notes
-----
- OpenAI requires OPENAI_API_KEY in the *API server* environment.
- Gemini requires GOOGLE_API_KEY in the *API server* environment.

Examples
--------
python launch_modified.py --query "When automatic electricity systems will have to be available?" --llm-id openai:gpt-4o-mini
python launch_modified.py --query "When automatic electricity systems will have to be available?" --llm-id ollama:qwen3:30b

python launch_modified.py --query "What are the reported 'cross-media effects' of using 'wet scrubbers' for dust capture in cupola systems compared to dry systems?" --llm-id openai:gpt-4o-mini
python launch_modified.py --query "Summarize CBAM" --llm-id openai:gpt-4o-mini
python launch_modified.py --query "Summarize CBAM" --llm-id gemini:gemini-1.5-flash
python launch_modified.py --provider gemini --model gemini-1.5-pro --query "..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import requests


def build_llm_id(llm_id: Optional[str], provider: Optional[str], model: Optional[str]) -> str:
    """Return provider:model.

    Priority:
      1) --llm-id
      2) --provider + --model
      3) env LLM_ID
      4) default openai:gpt-4o-mini
    """
    if llm_id:
        return llm_id.strip()

    if provider and model:
        return f"{provider.strip()}:{model.strip()}"

    env_llm = os.environ.get("LLM_ID")
    if env_llm:
        return env_llm.strip()

    return "openai:gpt-4o-mini"


def extract_answer(payload: Any) -> str:
    """Best-effort extraction for different response shapes."""
    if isinstance(payload, dict):
        for k in ("answer", "context", "result", "text", "message"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return v
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Query the Local RAG API.")
    p.add_argument("--api", default=os.environ.get("RAG_API", "http://127.0.0.1:5050"),
                   help="API base URL (default: http://127.0.0.1:5050 or env RAG_API)")
    p.add_argument("--endpoint", default=os.environ.get("RAG_ENDPOINT", "answer"),
                   help="Endpoint name (default: answer or env RAG_ENDPOINT)")
    p.add_argument("--query", default=None,
                   help="Question text. If omitted, read from stdin.")
    p.add_argument("--llm-id", default=None,
                   help="provider:model, e.g. openai:gpt-4o-mini or gemini:gemini-1.5-flash")
    p.add_argument("--provider", choices=["openai", "gemini", "ollama", "mock"], default=None,
                   help="LLM provider (used if --llm-id not given)")
    p.add_argument("--model", default=None,
                   help="Model name (used if --llm-id not given)")
    p.add_argument("--timeout", type=int, default=int(os.environ.get("RAG_TIMEOUT", "120")),
                   help="Request timeout seconds (default: 120 or env RAG_TIMEOUT)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    query = args.query
    if query is None:
        query = sys.stdin.read().strip()
    if not query:
        print("ERROR: no query provided. Use --query or pipe text to stdin.", file=sys.stderr)
        return 2

    llm_id = build_llm_id(args.llm_id, args.provider, args.model)
    if ":" not in llm_id:
        print("ERROR: llm_id must be provider:model (e.g. openai:gpt-4o-mini)", file=sys.stderr)
        return 2

    url = args.api.rstrip("/") + "/" + args.endpoint.lstrip("/")

    body: Dict[str, Any] = {"query": query, "llm_id": llm_id}

    print(f"Querying {url} with LLM {llm_id} \nQuery: {query}")
    r = requests.post(url, json=body, timeout=args.timeout)
    r.raise_for_status()

    try:
        data = r.json()
    except Exception:
        print(r.text)
        return 0

    print(extract_answer(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())