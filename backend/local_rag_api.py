from __future__ import annotations
"""
Local RAG Flask API

This file is designed to work with the libraries pinned in requirements.txt, but it also
includes lightweight fallbacks so the service can start even if some optional deps are missing.

Key improvements for question-answering accuracy on long technical PDFs:
- PDF text normalization (de-hyphenation, whitespace)
- Page-aware ingestion + optional page-range ingestion (page_start/page_end, 1-based)
- Chunking with separators tuned for regulatory text
- Heuristic multi-query expansion (acronyms, unit boosting, keyword focusing)
- Two-stage retrieval: vector candidates + lightweight lexical rerank (TF-IDF on candidates)
- Deterministic chunk IDs and source/page citations
"""

import os
import io
import re
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request

# ----------------------------
# Optional imports (LangChain / Chroma). We provide fallbacks for tests/offline.
# ----------------------------
try:
    from langchain_core.documents import Document  # type: ignore
except Exception:  # pragma: no cover
    @dataclass
    class Document:  # minimal compatible stand-in
        page_content: str
        metadata: Dict[str, Any]

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
except Exception:  # pragma: no cover
    RecursiveCharacterTextSplitter = None  # type: ignore

try:
    from langchain_chroma import Chroma  # type: ignore
except Exception:  # pragma: no cover
    Chroma = None  # type: ignore

try:
    import chromadb  # type: ignore
except Exception:  # pragma: no cover
    chromadb = None  # type: ignore

# PDF extraction
try:
    from langchain_community.document_loaders import PyPDFLoader  # type: ignore
except Exception:  # pragma: no cover
    PyPDFLoader = None  # type: ignore

try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore

# DOCX
try:
    import docx  # python-docx
except Exception:  # pragma: no cover
    docx = None  # type: ignore

# LLM backends (optional). We keep a "mock" backend for endpoint tests.
try:
    from langchain_ollama import ChatOllama  # type: ignore
except Exception:  # pragma: no cover
    ChatOllama = None  # type: ignore

try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
except Exception:  # pragma: no cover
    ChatGoogleGenerativeAI = None  # type: ignore

try:
    from langchain_anthropic import ChatAnthropic  # type: ignore
except Exception:  # pragma: no cover
    ChatAnthropic = None  # type: ignore

try:
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  # type: ignore
except Exception:  # pragma: no cover
    SystemMessage = None  # type: ignore
    HumanMessage = None  # type: ignore
    AIMessage = None  # type: ignore

# Embeddings (optional). For tests/offline we support "fake" and "tfidf".
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except Exception:  # pragma: no cover
    HuggingFaceEmbeddings = None  # type: ignore

try:
    from langchain_ollama import OllamaEmbeddings  # type: ignore
except Exception:  # pragma: no cover
    OllamaEmbeddings = None  # type: ignore

try:
    from langchain_core.embeddings import FakeEmbeddings  # type: ignore
except Exception:  # pragma: no cover
    try:
        from langchain_core.embeddings.fake import FakeEmbeddings  # type: ignore
    except Exception:  # pragma: no cover
        FakeEmbeddings = None  # type: ignore

# sklearn for fallback store + rerank
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Config
# ----------------------------
HOST = os.environ.get("RAG_HOST", "127.0.0.1")
PORT = int(os.environ.get("RAG_PORT", "5050"))

CHROMA_PATH = os.path.abspath(os.environ.get("CHROMA_DB_PATH", "./local_chroma_db"))
BASE_COLLECTION = os.environ.get("CHROMA_BASE_COLLECTION_NAME", "my_local_knowledge_base")
DEFAULT_VERSION = os.environ.get("CHROMA_COLLECTION_VERSION", "v1")

DEFAULT_EMBEDDING_ID = os.environ.get("EMBEDDING_ID", "tfidf:local")
DEFAULT_LLM_ID = os.environ.get("LLM_ID", "mock:any")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1600"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "240"))
LONG_DOC_CHUNK_SIZE = int(os.environ.get("LONG_DOC_CHUNK_SIZE", "2000"))
LONG_DOC_CHUNK_OVERLAP = int(os.environ.get("LONG_DOC_CHUNK_OVERLAP", "300"))
LONG_DOC_THRESHOLD_CHARS = int(os.environ.get("LONG_DOC_THRESHOLD_CHARS", "800000"))

DEFAULT_TOP_K = int(os.environ.get("TOP_K", "6"))
DEFAULT_FETCH_K = int(os.environ.get("FETCH_K", "24"))
DEFAULT_SEARCH_TYPE = os.environ.get("SEARCH_TYPE", "mmr")  # mmr | similarity
DEFAULT_MMR_LAMBDA = float(os.environ.get("MMR_LAMBDA", "0.3"))

DEFAULT_MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "18000"))

# Cross-encoder reranking. "none:" keeps retrieval exactly as it was.
DEFAULT_RERANKER_ID = os.environ.get("RERANKER_ID", "none:")

SUPPORTED_EXTS = {".pdf", ".txt", ".md", ".docx"}

WATCHED_FOLDER = os.path.abspath(
    os.environ.get("WATCHED_FOLDER", os.path.join(os.getcwd(), "watched_folder"))
)

app = Flask(__name__)

# ----------------------------
# Utilities
# ----------------------------
def _slug(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in s).strip("_")

def _collection_name(embedding_id: str, version: str) -> str:
    return f"{BASE_COLLECTION}__{version}__{_slug(embedding_id)}"

def _folder_of(source_path: str) -> str:
    """Absolute parent directory of a source file, the key retrieval is scoped by."""
    return str(Path(source_path).resolve().parent)

def _normalize_extracted_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate across line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _tokenize_for_lexical(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[/\-][A-Za-z0-9]+)*", (text or "").lower())

def doc_id(source_path: str, page: Optional[int], chunk: int) -> str:
    base = f"{source_path}|{page}|{chunk}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]

# ----------------------------
# Chunking
# ----------------------------
def choose_splitter(total_chars: int):
    if RecursiveCharacterTextSplitter is None:
        return None
    separators = ["\n\n", "\n", ". ", "; ", ": ", ", ", " ", ""]
    if total_chars >= LONG_DOC_THRESHOLD_CHARS:
        return RecursiveCharacterTextSplitter(
            chunk_size=LONG_DOC_CHUNK_SIZE,
            chunk_overlap=LONG_DOC_CHUNK_OVERLAP,
            separators=separators,
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
    )

def split_documents(docs: List[Document]) -> List[Document]:
    total_chars = sum(len(d.page_content or "") for d in docs)
    splitter = choose_splitter(total_chars)

    if splitter is None:
        # Very simple fallback
        chunks: List[Document] = []
        for d in docs:
            txt = d.page_content or ""
            step = max(200, CHUNK_SIZE - CHUNK_OVERLAP)
            for i in range(0, len(txt), step):
                piece = txt[i:i+CHUNK_SIZE]
                chunks.append(Document(page_content=piece, metadata=dict(d.metadata or {})))
        return chunks

    chunks: List[Document] = []
    for d in docs:
        chunks.extend(splitter.split_documents([d]))
    return chunks

# ----------------------------
# Document loading
# ----------------------------
def load_pdf_documents(tmp_path: str, source_name: str, source_path: str, page_start: Optional[int]=None, page_end: Optional[int]=None) -> List[Document]:
    start0 = max((page_start or 1) - 1, 0)
    end0 = (page_end - 1) if page_end is not None else None

    def within(i0: int) -> bool:
        if i0 < start0:
            return False
        if end0 is not None and i0 > end0:
            return False
        return True

    docs: List[Document] = []
    if PyPDFLoader is not None:
        loader = PyPDFLoader(tmp_path)
        for d in loader.load():
            md = dict(d.metadata or {})
            i0 = int(md.get("page", 0))
            if not within(i0):
                continue
            md["source"] = source_name
            md["source_path"] = source_path
            md["page"] = i0 + 1
            docs.append(Document(page_content=_normalize_extracted_text(d.page_content), metadata=md))
        return docs

    if PdfReader is None:
        raise RuntimeError("No PDF reader available (install pypdf or langchain-community).")

    reader = PdfReader(tmp_path)
    for i0, page in enumerate(reader.pages):
        if not within(i0):
            continue
        txt = _normalize_extracted_text(page.extract_text() or "")
        docs.append(Document(page_content=txt, metadata={"source": source_name, "source_path": source_path, "page": i0+1}))
    return docs

def load_docx(raw: bytes, source_name: str, source_path: str) -> List[Document]:
    if docx is None:
        raise RuntimeError("python-docx is not available.")
    f = io.BytesIO(raw)
    d = docx.Document(f)
    full = "\n".join(p.text for p in d.paragraphs)
    return [Document(page_content=_normalize_extracted_text(full), metadata={"source": source_name, "source_path": source_path, "page": None})]

def load_text(raw: bytes, source_name: str, source_path: str) -> List[Document]:
    txt = raw.decode("utf-8", errors="ignore")
    return [Document(page_content=_normalize_extracted_text(txt), metadata={"source": source_name, "source_path": source_path, "page": None})]

# ----------------------------
# Embeddings + Vector stores
# ----------------------------
def make_embeddings(embedding_id: str):
    """
    embedding_id formats:
      - tfidf:local               (always available fallback)
      - fake:anything             (if FakeEmbeddings exists)
      - hf:<huggingface model>    (requires langchain-huggingface)
      - ollama:<model>            (requires langchain-ollama and a running Ollama server)
      - openai:<model> / gemini:<model> (if corresponding deps installed)
    """
    if ":" not in embedding_id:
        raise ValueError("embedding_id must have a prefix like 'tfidf:' or 'hf:'")
    provider, model = embedding_id.split(":", 1)

    if provider == "tfidf":
        return None  # handled in fallback store

    if provider == "fake":
        if FakeEmbeddings is None:
            raise RuntimeError("FakeEmbeddings not available.")
        dim = int(os.environ.get("FAKE_EMB_DIM", "384"))
        return FakeEmbeddings(size=dim)

    if provider == "hf":
        if HuggingFaceEmbeddings is None:
            raise RuntimeError("langchain-huggingface not installed.")
        return HuggingFaceEmbeddings(model_name=model)

    if provider == "ollama":
        if OllamaEmbeddings is None:
            raise RuntimeError("langchain-ollama not installed.")
        return OllamaEmbeddings(model=model)

    raise RuntimeError(f"Embedding provider '{provider}' not available in this runtime. Use 'tfidf:local' or install deps.")

class _TfidfStore:
    """Lightweight in-memory store used as a fallback when Chroma/LangChain aren't installed."""
    def __init__(self):
        self._docs: List[Document] = []
        self._vectorizer = TfidfVectorizer(tokenizer=_tokenize_for_lexical, lowercase=False, min_df=1)
        self._X = None

    def _refit(self):
        corpus = [d.page_content or "" for d in self._docs]
        self._X = self._vectorizer.fit_transform(corpus) if corpus else None

    def add_documents(self, docs: List[Document]):
        self._docs.extend(docs)
        self._refit()

    def delete(self, source_path: str):
        self._docs = [d for d in self._docs if (d.metadata or {}).get("source_path") != source_path]
        self._refit()

    def similarity_search_with_score(self, query: str, k: int) -> List[Tuple[Document, float]]:
        if not self._docs or self._X is None:
            return []
        qv = self._vectorizer.transform([query])
        sims = cosine_similarity(qv, self._X).ravel()
        idx = sims.argsort()[::-1][:k]
        # score as 1 - sim distance-like to keep downstream robust
        return [(self._docs[i], float(1.0 - sims[i])) for i in idx]

    def similarity_search(self, query: str, k: int) -> List[Document]:
        return [d for d, _ in self.similarity_search_with_score(query, k)]

    def max_marginal_relevance_search(self, query: str, k: int, fetch_k: int, lambda_mult: float) -> List[Document]:
        # Simple MMR over cosine sims in TF-IDF space
        if not self._docs or self._X is None:
            return []
        qv = self._vectorizer.transform([query])
        sims = cosine_similarity(qv, self._X).ravel()
        candidates = sims.argsort()[::-1][:fetch_k].tolist()
        selected: List[int] = []
        cand_vecs = self._X[candidates]

        while candidates and len(selected) < k:
            if not selected:
                best = candidates[0]
                selected.append(best)
                candidates = [c for c in candidates if c != best]
                continue

            sel_vecs = self._X[selected]
            # diversity penalty: max similarity to selected
            div = cosine_similarity(self._X[candidates], sel_vecs).max(axis=1)
            rel = sims[candidates]
            mmr = lambda_mult * rel - (1 - lambda_mult) * div
            best_idx = int(mmr.argmax())
            best = candidates[best_idx]
            selected.append(best)
            candidates.pop(best_idx)

        return [self._docs[i] for i in selected]

# Cache stores per (embedding_id, version)
_STORE_CACHE: Dict[str, Any] = {}

_BACKFILLED: set = set()

def _backfill_folder_metadata(store: Any, cname: str) -> int:
    """Add the folder key to chunks written before it existed. Returns rows updated.

    Chroma merges metadata on update and leaves embeddings untouched, so this
    costs no re-embedding. Rows without the key are invisible to a folder
    filter, which is why this runs before any filtered read.
    """
    coll = getattr(store, "_collection", None)
    if coll is None:
        return 0

    updated = 0
    offset, page = 0, 10000
    while True:
        got = coll.get(include=["metadatas"], limit=page, offset=offset)
        ids = got.get("ids") or []
        metas = got.get("metadatas") or []
        stale_ids, stale_metas = [], []
        for cid, md in zip(ids, metas):
            md = md or {}
            if not md.get("folder") and md.get("source_path"):
                stale_ids.append(cid)
                stale_metas.append({"folder": _folder_of(md["source_path"])})
        if stale_ids:
            coll.update(ids=stale_ids, metadatas=stale_metas)
            updated += len(stale_ids)
        if len(ids) < page:
            break
        offset += page
    _BACKFILLED.add(cname)
    return updated

def get_store(embedding_id: str, version: str):
    key = f"{embedding_id}::{version}"
    if key in _STORE_CACHE:
        return _STORE_CACHE[key]

    cname = _collection_name(embedding_id, version)

    # Prefer LangChain Chroma if available
    if Chroma is not None:
        emb = make_embeddings(embedding_id)
        store = Chroma(collection_name=cname, persist_directory=CHROMA_PATH, embedding_function=emb)
        _STORE_CACHE[key] = (store, emb, cname, "chroma")
        if cname not in _BACKFILLED:
            _backfill_folder_metadata(store, cname)
        return _STORE_CACHE[key]

    # Fallback: in-memory TF-IDF store
    store = _TfidfStore()
    _STORE_CACHE[key] = (store, None, cname, "tfidf")
    return _STORE_CACHE[key]

def delete_by_source(store: Any, source_path: str):
    if hasattr(store, "_collection"):
        store._collection.delete(where={"source_path": source_path})
        return
    if hasattr(store, "delete"):
        store.delete(source_path)
        return
    raise RuntimeError("Store does not support delete")

# ----------------------------
# Retrieval improvements
# ----------------------------
def _build_query_variants(query: str) -> List[str]:
    q = (query or "").strip()
    if not q:
        return []
    variants = [q]

    expansions = {
        "bat-ael": "BAT-associated emission level",
        "bat-aepl": "BAT-associated environmental performance level",
        "tvoc": "total volatile organic compounds TVOC",
        "pcdd/f": "PCDD/F dioxins furans",
        "aox": "Adsorbable Organically Bound Halogens AOX",
        "hbc": "hot blast cupola HBC",
        "cbc": "cold blast cupola CBC",
        "eaf": "electric arc furnace EAF",
        "sps": "spark plasma sintering SPS",
        "otnoc": "other than normal operating conditions OTNOC",
        "cms": "chemicals management system CMS",
        "ems": "environmental management system EMS",
    }
    q_low = q.lower()
    for k, v in expansions.items():
        if k in q_low and v.lower() not in q_low:
            variants.append(f"{q} ({v})")

    toks = _tokenize_for_lexical(q)
    stop = {"what","which","are","is","the","a","an","of","for","to","and","in","on","under","from","with","using","according","based","list","describe"}
    keep = [t for t in toks if t not in stop]
    if len(keep) >= 6:
        variants.append(" ".join(keep[:18]))

    unit_hits = [t for t in toks if any(u in t for u in ["mg","nm","m3","mw","kwh","kg","ton","t/"])]
    if unit_hits:
        variants.append(q + " " + " ".join(unit_hits[:10]))

    out, seen = [], set()
    for v in variants:
        v2 = " ".join(v.split())
        if v2 and v2 not in seen:
            out.append(v2)
            seen.add(v2)
    return out

def _vector_candidates(store: Any, query: str, k: int) -> List[Tuple[Document, float]]:
    if hasattr(store, "similarity_search_with_score"):
        try:
            return store.similarity_search_with_score(query, k=k)
        except Exception:
            pass
    if hasattr(store, "similarity_search"):
        docs = store.similarity_search(query, k=k)
        return [(d, 1.0) for d in docs]
    raise RuntimeError("Store does not support retrieval")

def _hybrid_rerank(query: str, docs_and_scores: List[Tuple[Document, float]]) -> List[Document]:
    if not docs_and_scores:
        return []
    docs = [d for d, _ in docs_and_scores]
    raw = [float(s) for _, s in docs_and_scores]
    inv = [-r for r in raw]
    vmin, vmax = min(inv), max(inv)
    vec_sim = [(x - vmin) / (vmax - vmin) if vmax != vmin else 0.5 for x in inv]

    corpus = [d.page_content or "" for d in docs]
    vect = TfidfVectorizer(tokenizer=_tokenize_for_lexical, lowercase=False, min_df=1)
    X = vect.fit_transform(corpus)
    qv = vect.transform([query])
    lex = cosine_similarity(qv, X).ravel().tolist()
    lmin, lmax = min(lex), max(lex)
    lex_sim = [(x - lmin) / (lmax - lmin) if lmax != lmin else 0.0 for x in lex]

    blended = [0.55 * ls + 0.45 * vs for ls, vs in zip(lex_sim, vec_sim)]
    order = sorted(range(len(docs)), key=lambda i: blended[i], reverse=True)
    return [docs[i] for i in order]

def retrieve_docs(store: Any, query: str, top_k: int, fetch_k: int, search_type: str, mmr_lambda: float,
                  reranker_id: Optional[str] = None) -> List[Document]:
    variants = _build_query_variants(query)
    cand: List[Tuple[Document, float]] = []
    per_q = max(5, min(fetch_k, 40))
    for v in variants[:4]:
        cand.extend(_vector_candidates(store, v, k=per_q))

    uniq: Dict[str, Tuple[Document, float]] = {}
    for d, s in cand:
        md = d.metadata or {}
        key = md.get("id") or f"{md.get('source_path')}|{md.get('page')}|{md.get('chunk')}|{hash(d.page_content)}"
        if key not in uniq or s < uniq[key][1]:
            uniq[key] = (d, s)

    if search_type == "mmr" and hasattr(store, "max_marginal_relevance_search"):
        try:
            mmr_docs = store.max_marginal_relevance_search(query, k=top_k, fetch_k=max(fetch_k, top_k*4), lambda_mult=mmr_lambda)
            for d in mmr_docs:
                md = d.metadata or {}
                key = md.get("id") or f"{md.get('source_path')}|{md.get('page')}|{md.get('chunk')}|{hash(d.page_content)}"
                uniq.setdefault(key, (d, 0.0))
        except Exception:
            pass

    reranked = _hybrid_rerank(query, list(uniq.values()))

    # The hybrid blend picks the candidate pool; the cross-encoder, when
    # enabled, reorders that pool and makes the final top_k cut.
    from reranker import make_reranker, rerank
    rr = make_reranker(reranker_id if reranker_id is not None else DEFAULT_RERANKER_ID)
    if rr is None:
        return reranked[:top_k]
    return rerank(rr, query, reranked[:max(fetch_k, top_k)], top_k)

def format_context(docs: List[Document], max_chars: int) -> Tuple[str, List[Dict[str, Any]]]:
    blocks: List[str] = []
    sources: List[Dict[str, Any]] = []
    used = 0
    for d in docs:
        md = d.metadata or {}
        source = md.get("source", "unknown")
        page = md.get("page", None)
        chunk = md.get("chunk", None)
        header = f"--- Source: {source}"
        if page is not None:
            header += f", page {page}"
        if chunk is not None:
            header += f", chunk {chunk}"
        header += " ---\n"
        piece = header + (d.page_content or "").strip()
        if not piece.strip():
            continue
        if used + len(piece) > max_chars and blocks:
            break
        blocks.append(piece)
        used += len(piece)
        sources.append({"source": source, "page": page, "chunk": chunk, "source_path": md.get("source_path")})
    return "\n\n".join(blocks), sources

# ----------------------------
# LLM
# ----------------------------
def make_llm(llm_id: str):
    if ":" not in llm_id:
        raise ValueError("llm_id must have a prefix like 'ollama:' or 'openai:'")
    provider, model = llm_id.split(":", 1)

    if provider == "mock":
        class _Resp:
            def __init__(self, content: str):
                self.content = content
        class _MockLLM:
            def invoke(self, messages):
                user_msg = messages[-1].content if messages else ""
                return _Resp("MOCK_ANSWER: " + (user_msg or "")[:600])
        return _MockLLM()

    if provider == "ollama":
        if ChatOllama is None:
            raise RuntimeError("langchain-ollama not installed.")
        return ChatOllama(model=model, temperature=float(os.environ.get("OLLAMA_TEMPERATURE","0.2")))

    if provider == "openai":
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai not installed.")
        return ChatOpenAI(model=model, temperature=float(os.environ.get("OPENAI_TEMPERATURE","0.2")))

    if provider == "lmstudio":
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai not installed.")
        # LM Studio exposes an OpenAI-compatible server. The api_key is passed
        # explicitly so a real OPENAI_API_KEY in the environment is never sent
        # to the local server.
        return ChatOpenAI(
            model=model,
            base_url=os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
            api_key=os.environ.get("LMSTUDIO_API_KEY", "lm-studio"),
            temperature=float(os.environ.get("LMSTUDIO_TEMPERATURE", "0.2")),
        )

    if provider == "gemini":
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError("langchain-google-genai not installed.")
        return ChatGoogleGenerativeAI(model=model)

    if provider == "anthropic":
        if ChatAnthropic is None:
            raise RuntimeError("langchain-anthropic not installed.")
        return ChatAnthropic(
            model=model,
            max_tokens=int(os.environ.get("ANTHROPIC_MAX_TOKENS", "4096")),
        )

    raise ValueError(f"Unsupported llm provider: {provider}")

# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.get("/ready")
def ready():
    embedding_id = request.args.get("embedding_id", DEFAULT_EMBEDDING_ID)
    version = request.args.get("version", DEFAULT_VERSION)
    try:
        store, emb, cname, backend = get_store(embedding_id, version)
        return jsonify({"status": "ready", "collection": cname, "backend": backend, "embedding_id": embedding_id, "version": version})
    except Exception as e:
        return jsonify({"status": "not_ready", "error": str(e)}), 503

@app.get("/sources")
def sources():
    embedding_id = request.args.get("embedding_id", DEFAULT_EMBEDDING_ID)
    version = request.args.get("version", DEFAULT_VERSION)
    store, _, cname, backend = get_store(embedding_id, version)
    if hasattr(store, "_collection"):
        # Chroma's SQLite backend binds one variable per row, so a single large
        # get() raises "too many SQL variables" past ~32k chunks. Page instead.
        found = set()
        offset, page = 0, 10000
        while True:
            metas = store._collection.get(include=["metadatas"], limit=page, offset=offset).get("metadatas") or []
            found.update((m or {}).get("source_path") for m in metas if (m or {}).get("source_path"))
            if len(metas) < page:
                break
            offset += page
        srcs = sorted(found)
    else:
        # fallback store
        srcs = sorted({(d.metadata or {}).get("source_path") for d in getattr(store, "_docs", []) if (d.metadata or {}).get("source_path")})
    return jsonify({"sources": srcs, "collection": cname, "backend": backend, "embedding_id": embedding_id, "version": version})

@app.post("/delete")
def delete():
    payload = request.get_json(silent=True) or {}
    source_path = payload.get("source_path")
    if not source_path:
        return jsonify({"error": "source_path is required"}), 400
    embedding_id = payload.get("embedding_id", DEFAULT_EMBEDDING_ID)
    version = payload.get("version", DEFAULT_VERSION)
    store, _, cname, backend = get_store(embedding_id, version)
    delete_by_source(store, source_path)
    return jsonify({"status":"ok","deleted_source_path":source_path,"collection":cname,"backend":backend,"embedding_id":embedding_id,"version":version})

def _ingest_documents(docs: List[Document], source_path: str, embedding_id: str, version: str) -> Dict[str, Any]:
    """Chunk + upsert pre-loaded Documents. Shared by /ingest and /ingest_folder."""
    store, _, cname, backend = get_store(embedding_id, version)
    delete_by_source(store, source_path)

    folder = _folder_of(source_path)
    chunks = split_documents(docs)
    for idx, d in enumerate(chunks):
        md = dict(d.metadata or {})
        md["chunk"] = idx
        md["folder"] = folder
        md["id"] = doc_id(source_path, md.get("page"), idx)
        d.metadata = md

    if hasattr(store, "add_documents"):
        store.add_documents(chunks)
    elif hasattr(store, "_collection"):
        store.add_texts(
            [c.page_content for c in chunks],
            metadatas=[c.metadata for c in chunks],
            ids=[c.metadata["id"] for c in chunks],
        )
    else:
        raise RuntimeError("Store does not support add_documents")

    return {
        "collection": cname,
        "backend": backend,
        "embedding_id": embedding_id,
        "version": version,
        "pages_loaded": len(docs),
        "chunks_added": len(chunks),
        "total_chars": sum(len(d.page_content or "") for d in docs),
    }


def _ingest_local_file(source_path: str, embedding_id: str, version: str) -> Dict[str, Any]:
    """Load a file already on disk and ingest it into the active store."""
    p = Path(source_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"file not found: {source_path}")
    filename = p.name
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"unsupported extension '{ext}'")

    if ext == ".pdf":
        docs = load_pdf_documents(str(p), filename, str(p))
    elif ext == ".docx":
        docs = load_docx(p.read_bytes(), filename, str(p))
    else:
        docs = load_text(p.read_bytes(), filename, str(p))

    res = _ingest_documents(docs, str(p), embedding_id, version)
    res.update({"source_path": str(p), "filename": filename})
    return res


@app.post("/ingest")
def ingest():
    if "file" not in request.files:
        return jsonify({"error": "multipart form-data must include 'file'"}), 400
    missing = [k for k in ("embedding_id","version","source_path") if not (request.form.get(k) or "").strip()]
    if missing:
        return jsonify({"error":"missing required form fields","missing":missing}), 400

    f = request.files["file"]
    raw = f.read()
    filename = f.filename or "uploaded_file"
    source_path = request.form.get("source_path")
    embedding_id = request.form.get("embedding_id")
    version = request.form.get("version")

    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTS:
        return jsonify({"error": f"unsupported extension '{ext}'", "supported": sorted(SUPPORTED_EXTS)}), 400

    docs: List[Document] = []
    if ext == ".pdf":
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            page_start = request.form.get("page_start")
            page_end = request.form.get("page_end")
            ps = int(page_start) if page_start and page_start.strip().isdigit() else None
            pe = int(page_end) if page_end and page_end.strip().isdigit() else None
            docs = load_pdf_documents(tmp_path, filename, source_path, page_start=ps, page_end=pe)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    elif ext == ".docx":
        docs = load_docx(raw, filename, source_path)
    else:
        docs = load_text(raw, filename, source_path)

    res = _ingest_documents(docs, source_path, embedding_id, version)
    return jsonify({"status": "ok", "source_path": source_path, "filename": filename, **res})


@app.get("/config")
def config():
    """Return the server's current defaults so clients can auto-align."""
    return jsonify({
        "embedding_id": DEFAULT_EMBEDDING_ID,
        "version": DEFAULT_VERSION,
        "llm_id": DEFAULT_LLM_ID,
        "reranker_id": DEFAULT_RERANKER_ID,
        "watched_folder": WATCHED_FOLDER,
        "supported_extensions": sorted(SUPPORTED_EXTS),
    })


@app.post("/ingest_folder")
def ingest_folder():
    """Scan a folder and ingest new/changed files. Uses the folder_watcher fingerprint+state."""
    from folder_watcher import (
        scan_disk, state_file_path, load_state, save_state, file_fingerprint,
    )

    payload = request.get_json(silent=True) or {}
    folder = payload.get("folder") or WATCHED_FOLDER
    embedding_id = payload.get("embedding_id", DEFAULT_EMBEDDING_ID)
    version = payload.get("version", DEFAULT_VERSION)
    extensions = payload.get("extensions") or sorted(SUPPORTED_EXTS)

    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.is_dir():
        return jsonify({"error": f"folder not found or not a directory: {folder}"}), 400

    state_path = state_file_path(folder_path, version, embedding_id)
    state = load_state(state_path)
    files_state = state.setdefault("files", {})

    disk = scan_disk(folder_path, extensions, recursive=False)

    ingested: List[Dict[str, Any]] = []
    skipped_unchanged = 0
    errors: List[Dict[str, str]] = []

    for path_str in sorted(disk):
        p = Path(path_str)
        try:
            fp = file_fingerprint(p)
        except FileNotFoundError:
            continue
        prev = files_state.get(path_str)
        if prev and prev.get("mtime") == fp["mtime"] and prev.get("size") == fp["size"]:
            skipped_unchanged += 1
            continue
        try:
            r = _ingest_local_file(path_str, embedding_id, version)
            ingested.append({"source_path": r["source_path"], "chunks": r["chunks_added"]})
            files_state[path_str] = fp
            save_state(state_path, state)
            print(f"file {p.name} added to the database")
        except Exception as e:
            errors.append({"source_path": path_str, "error": f"{type(e).__name__}: {e}"})

    _, _, cname, backend = get_store(embedding_id, version)
    return jsonify({
        "status": "ok",
        "folder": str(folder_path),
        "embedding_id": embedding_id,
        "version": version,
        "collection": cname,
        "backend": backend,
        "scanned": len(disk),
        "ingested": ingested,
        "skipped_unchanged": skipped_unchanged,
        "errors": errors,
    })

@app.post("/retrieve")
def retrieve():
    payload = request.get_json(silent=True) or {}
    query = payload.get("query")
    if not query:
        return jsonify({"error":"query is required"}), 400
    embedding_id = payload.get("embedding_id", DEFAULT_EMBEDDING_ID)
    version = payload.get("version", DEFAULT_VERSION)
    top_k = int(payload.get("top_k", DEFAULT_TOP_K))
    fetch_k = int(payload.get("fetch_k", DEFAULT_FETCH_K))
    search_type = payload.get("search_type", DEFAULT_SEARCH_TYPE)
    mmr_lambda = float(payload.get("mmr_lambda", DEFAULT_MMR_LAMBDA))
    max_chars = int(payload.get("max_context_chars", DEFAULT_MAX_CONTEXT_CHARS))
    reranker_id = payload.get("reranker_id", DEFAULT_RERANKER_ID)

    store, _, cname, backend = get_store(embedding_id, version)
    docs = retrieve_docs(store, query, top_k=top_k, fetch_k=fetch_k, search_type=search_type, mmr_lambda=mmr_lambda,
                         reranker_id=reranker_id)
    context, sources = format_context(docs, max_chars)
    return jsonify({"context":context,"sources":sources,"collection":cname,"backend":backend,"embedding_id":embedding_id,"version":version,"reranker_id":reranker_id})


MAX_HISTORY_MESSAGES = 100  # 50 user/assistant exchanges

_ANAPHORA_RE = re.compile(
    r"\b(it|its|it's|they|them|their|this|that|these|those|he|she|his|her|one)\b",
    re.IGNORECASE,
)


def _retrieval_query(question: str, history) -> str:
    """Return the query to search on, expanded when it refers back to the chat.

    Retrieval runs on the question alone, so a follow-up like "How is it
    produced industrially?" would search on the pronoun and return documents
    about an unrelated subject -- the LLM still resolves "it" from history and
    presents that wrong context as an answer. Prepending the previous user turn
    restores the subject. Questions naming their own subject are left untouched
    so an unrelated follow-up is not diluted by the previous topic.
    """
    if not history or not _ANAPHORA_RE.search(question):
        return question
    for item in reversed(history[-MAX_HISTORY_MESSAGES:]):
        if not isinstance(item, dict):
            continue
        if str(item.get("role", "")).strip().lower() not in {"user", "human"}:
            continue
        previous = str(item.get("content") or "").strip()
        return f"{previous} {question}" if previous else question
    return question


def _build_chat_messages(
    system_text: str,
    history,
    user_text: str,
    *,
    cacheable_user_prefix: Optional[str] = None,
):
    """Build the message list passed to the LLM.

    When `cacheable_user_prefix` is provided, the system prompt and that prefix
    are emitted as structured content blocks tagged with Anthropic-style
    cache_control. langchain-anthropic forwards these to the Messages API;
    other providers ignore unknown fields. Use it for the retrieved context so
    re-runs of the same query against multiple Claude configs hit the cache.

    Falls back to lightweight shim objects with a .content attribute when
    langchain_core is not installed (matches the pre-existing behavior).
    """
    use_cache = cacheable_user_prefix is not None
    if SystemMessage is not None and HumanMessage is not None and AIMessage is not None:
        if use_cache:
            sys_content = [
                {"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            sys_content = system_text
        msgs = [SystemMessage(content=sys_content)]
        for item in (history or [])[-MAX_HISTORY_MESSAGES:]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", ""))
            if not content:
                continue
            if role in {"user", "human"}:
                msgs.append(HumanMessage(content=content))
            elif role in {"assistant", "ai"}:
                msgs.append(AIMessage(content=content))
        if use_cache:
            user_content = [
                {"type": "text", "text": cacheable_user_prefix, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": user_text},
            ]
        else:
            user_content = user_text
        msgs.append(HumanMessage(content=user_content))
        return msgs

    shim = lambda c: type("M", (), {"content": c})()
    msgs = [shim(system_text)]
    for item in (history or [])[-MAX_HISTORY_MESSAGES:]:
        if isinstance(item, dict) and item.get("content"):
            msgs.append(shim(str(item["content"])))
    combined = (cacheable_user_prefix + "\n\n" + user_text) if use_cache else user_text
    msgs.append(shim(combined))
    return msgs


@app.post("/answer")
def answer():
    payload = request.get_json(silent=True) or {}
    question = payload.get("query")
    if not question:
        return jsonify({"error":"query is required"}), 400
    embedding_id = payload.get("embedding_id", DEFAULT_EMBEDDING_ID)
    version = payload.get("version", DEFAULT_VERSION)
    llm_id = payload.get("llm_id", DEFAULT_LLM_ID)
    top_k = int(payload.get("top_k", DEFAULT_TOP_K))
    fetch_k = int(payload.get("fetch_k", DEFAULT_FETCH_K))
    search_type = payload.get("search_type", DEFAULT_SEARCH_TYPE)
    mmr_lambda = float(payload.get("mmr_lambda", DEFAULT_MMR_LAMBDA))
    max_chars = int(payload.get("max_context_chars", DEFAULT_MAX_CONTEXT_CHARS))
    reranker_id = payload.get("reranker_id", DEFAULT_RERANKER_ID)

    history = payload.get("history") or []
    if not isinstance(history, list):
        history = []

    store, _, cname, backend = get_store(embedding_id, version)
    docs = retrieve_docs(store, _retrieval_query(question, history), top_k=top_k, fetch_k=fetch_k,
                         search_type=search_type, mmr_lambda=mmr_lambda,
                         reranker_id=reranker_id)
    context, sources = format_context(docs, max_chars)

    system_text = (
        "You are a precise assistant doing retrieval-augmented generation (RAG) for a technical regulatory document. "
        "You have two sources of information: (1) the retrieved CONTEXT below, and (2) the prior conversation turns. "
        "For any FACTUAL/TECHNICAL claim about the documents, use ONLY the CONTEXT: "
        "copy exact numeric ranges, units, and conditional clauses; cite in-line like (Source: <name>, page <n>); "
        "if the CONTEXT is insufficient, reply 'Not found in the provided context.' Do not invent page numbers. "
        "For conversational or meta questions about the chat itself (the user's name, what they just said, "
        "clarifying the previous answer, pronoun/entity resolution), you MAY use the prior turns directly "
        "and no citations are needed. If multiple conditions apply to a factual answer, enumerate them (a), (b), etc."
    )
    cacheable_prefix = (
        "CONTEXT\n------\n"
        f"{context}"
    )
    user_text = (
        "\n\nQUESTION\n--------\n"
        f"{question}\n\n"
        "INSTRUCTIONS\n------------\n"
        "1) Answer with the exact values/thresholds/ranges and any stated applicability conditions.\n"
        "2) If the question asks to list items, provide a bullet list.\n"
        "3) End with a short 'Evidence' section listing the cited sources/pages.\n"
    )

    llm = make_llm(llm_id)

    if llm_id.startswith("anthropic:"):
        messages = _build_chat_messages(
            system_text, history, user_text, cacheable_user_prefix=cacheable_prefix
        )
    else:
        messages = _build_chat_messages(system_text, history, cacheable_prefix + user_text)
    resp = llm.invoke(messages)
    content = getattr(resp, "content", None)
    if content is None:
        answer_text = str(resp)
    elif str(content).strip():
        answer_text = content
    else:
        # Reasoning models can spend their whole budget before emitting any
        # content (common when a model is loaded with a small context window).
        # Report that instead of dumping the raw response object.
        usage = (getattr(resp, "response_metadata", None) or {}).get("token_usage", {})
        answer_text = (
            f"The model '{llm_id}' returned an empty answer, most likely hitting its token "
            f"limit before producing content. token_usage: {usage}"
        )

    return jsonify({"answer":answer_text,"sources":sources,"collection":cname,"backend":backend,"embedding_id":embedding_id,"version":version,"llm_id":llm_id,"reranker_id":reranker_id})

if __name__ == "__main__":
    os.makedirs(CHROMA_PATH, exist_ok=True)
    app.run(host=HOST, port=PORT, debug=True)
