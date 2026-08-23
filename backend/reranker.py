"""Optional cross-encoder reranking of retrieved chunks.

A reranker re-scores (query, chunk) pairs jointly, which is more accurate than
the bi-encoder similarity used for the initial vector search. It runs on the
candidates that retrieval already selected, so it is a pure reordering step.

reranker_id follows the same 'provider:model' convention as embedding_id and
llm_id:
  - none:                 disabled (the default)
  - ce:<hf-model>         sentence-transformers CrossEncoder,
                          e.g. ce:BAAI/bge-reranker-v2-m3
"""
from typing import Any, List

try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore

# Cross-encoders are expensive to load, so keep one instance per reranker_id.
_CACHE: dict = {}


def make_reranker(reranker_id: str):
    """Return a loaded reranker, or None when reranking is disabled."""
    provider, _, model = (reranker_id or "none").partition(":")
    if provider in ("none", ""):
        return None

    if provider != "ce":
        raise ValueError(f"Unsupported reranker provider: {provider}")
    if not model:
        raise ValueError("reranker_id 'ce:' requires a model name")

    if reranker_id not in _CACHE:
        if CrossEncoder is None:
            raise RuntimeError("sentence-transformers not installed.")
        _CACHE[reranker_id] = CrossEncoder(model)
    return _CACHE[reranker_id]


def rerank(reranker: Any, query: str, docs: List, top_k: int) -> List:
    """Reorder docs by cross-encoder score and keep the best top_k."""
    if reranker is None or not docs:
        return docs[:top_k]
    scores = reranker.predict([(query, d.page_content or "") for d in docs])
    order = sorted(range(len(docs)), key=lambda i: float(scores[i]), reverse=True)
    return [docs[i] for i in order[:top_k]]
