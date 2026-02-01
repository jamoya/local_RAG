from __future__ import annotations

"""
Local RAG Flask API (LangChain >= 1.x compatible)

Goals:
- Flexible switching among embedding models and LLM backends (Ollama / OpenAI / Gemini)
- Accurate retrieval for lengthy PDFs (page-aware loading + chunking)
- Vector dimension compatibility enforced via versioned, dimension-tagged collections
- CRUD sync with filesystem watcher:
    - POST /ingest  (multipart: file, source_path?, embedding_id?, version?)
    - POST /delete  (json: source_path, embedding_id?, version?)
    - GET  /sources (query: embedding_id, version)  -> for reconcile mode
- Retrieval + Answering:
    - POST /retrieve (json: query, top_k?, fetch_k?, search_type?, mmr_lambda?, embedding_id?, version?)
    - POST /answer   (json: question, same retrieval params, llm_id?, max_context_chars?)

Test quickly:
    python local_rag_api.py
    curl http://127.0.0.1:5000/health
"""

import os
import io
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:  # pragma: no cover
    HuggingFaceEmbeddings = None  # type: ignore

# LLMs (optional, depending on what you install)
try:
    from langchain_ollama import ChatOllama
except Exception:  # pragma: no cover
    ChatOllama = None  # type: ignore

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover
    ChatGoogleGenerativeAI = None  # type: ignore

# PDF loading (prefer LangChain loader; fallback to pypdf)
try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:  # pragma: no cover
    PyPDFLoader = None  # type: ignore

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore


app = Flask(__name__)

# ----------------------------
# Defaults (override via env)
# ----------------------------
HOST = os.environ.get("RAG_HOST", "127.0.0.1")
PORT = int(os.environ.get("RAG_PORT", "5000"))

CHROMA_PATH = os.path.abspath(os.environ.get("CHROMA_DB_PATH", "./local_chroma_db"))
BASE_COLLECTION = os.environ.get("CHROMA_BASE_COLLECTION_NAME", "my_local_knowledge_base")
DEFAULT_VERSION = os.environ.get("CHROMA_COLLECTION_VERSION", "v1")

# Default embedding + LLM
DEFAULT_EMBEDDING_ID = os.environ.get("EMBEDDING_ID", "hf:BAAI/bge-large-en-v1.5")
DEFAULT_LLM_ID = os.environ.get("LLM_ID", "ollama:llama3.2")

# Chunking tuned for long technical PDFs
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1600"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "240"))
LONG_DOC_THRESHOLD_CHARS = int(os.environ.get("LONG_DOC_THRESHOLD_CHARS", "800000"))
LONG_DOC_CHUNK_SIZE = int(os.environ.get("LONG_DOC_CHUNK_SIZE", "2000"))
LONG_DOC_CHUNK_OVERLAP = int(os.environ.get("LONG_DOC_CHUNK_OVERLAP", "300"))

# Retrieval defaults (accuracy-focused)
DEFAULT_TOP_K = int(os.environ.get("TOP_K", "12"))
DEFAULT_FETCH_K = int(os.environ.get("FETCH_K", "40"))
DEFAULT_SEARCH_TYPE = os.environ.get("SEARCH_TYPE", "mmr")  # "similarity" or "mmr"
DEFAULT_MMR_LAMBDA = float(os.environ.get("MMR_LAMBDA", "0.3"))

# LLM inference params (Ollama)
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "1200"))
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_TOP_P = float(os.environ.get("OLLAMA_TOP_P", "0.9"))

# Context size passed to LLM
DEFAULT_MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "18000"))


# ----------------------------
# Helpers: model factories
# ----------------------------
def _slug(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in s).strip("_")


def _stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def make_embeddings(embedding_id: str):
    """
    embedding_id formats:
      - hf:<huggingface model id>   (requires langchain-huggingface + sentence-transformers)
      - openai:<model>              (requires langchain-openai)
      - gemini:<model>              (requires langchain-google-genai)
      - ollama:<model>              (embeddings model in ollama; requires langchain-ollama)
    """
    if ":" not in embedding_id:
        raise ValueError("embedding_id must have a prefix like 'hf:' or 'openai:'")

    provider, model = embedding_id.split(":", 1)

    if provider == "hf":
        if HuggingFaceEmbeddings is None:
            raise RuntimeError("langchain-huggingface is not installed. pip install -U langchain-huggingface sentence-transformers")
        return HuggingFaceEmbeddings(model_name=model)

    if provider == "openai":
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai is not installed. pip install -U langchain-openai")
        # OpenAI embeddings live in langchain_openai too; import lazily to avoid hard dependency.
        from langchain_openai import OpenAIEmbeddings  # type: ignore
        return OpenAIEmbeddings(model=model)

    if provider == "gemini":
        # Gemini embeddings
        from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
        return GoogleGenerativeAIEmbeddings(model=model)

    if provider == "ollama":
        if ChatOllama is None:
            raise RuntimeError("langchain-ollama is not installed. pip install -U langchain-ollama")
        # Ollama embeddings class lives in langchain_ollama
        from langchain_ollama import OllamaEmbeddings  # type: ignore
        return OllamaEmbeddings(model=model)

    raise ValueError(f"Unknown embedding provider prefix: {provider}")


def embedding_dimension(emb) -> int:
    v = emb.embed_query("dimension probe")
    if not isinstance(v, list) or not v:
        raise RuntimeError("Embedding provider returned invalid embedding vector.")
    return len(v)


def collection_name(base: str, version: str, embedding_id: str, dim: int) -> str:
    key = f"{base}|{version}|{embedding_id}|{dim}"
    return f"{base}__{version}__{_slug(embedding_id)}__d{dim}__{_stable_hash(key)}"


def make_llm(llm_id: str):
    """
    llm_id formats:
      - ollama:<model>
      - openai:<model>
      - gemini:<model>
    """
    if ":" not in llm_id:
        raise ValueError("llm_id must have a prefix like 'ollama:' or 'openai:'")
    provider, model = llm_id.split(":", 1)

    if provider == "ollama":
        if ChatOllama is None:
            raise RuntimeError("langchain-ollama is not installed. pip install -U langchain-ollama")
        return ChatOllama(
            model=model,
            temperature=OLLAMA_TEMPERATURE,
            top_p=OLLAMA_TOP_P,
            num_ctx=OLLAMA_NUM_CTX,
            num_predict=OLLAMA_NUM_PREDICT,
        )

    if provider == "openai":
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai is not installed. pip install -U langchain-openai")
        return ChatOpenAI(model=model, temperature=0.2)

    if provider == "gemini":
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError("langchain-google-genai is not installed. pip install -U langchain-google-genai")
        return ChatGoogleGenerativeAI(model=model, temperature=0.2)

    raise ValueError(f"Unknown LLM provider prefix: {provider}")


# ----------------------------
# Vector store access + validation
# ----------------------------
def get_store(embedding_id: str, version: str) -> Tuple[Chroma, Any, str, int]:
    emb = make_embeddings(embedding_id)
    dim = embedding_dimension(emb)
    cname = collection_name(BASE_COLLECTION, version, embedding_id, dim)

    store = Chroma(
        collection_name=cname,
        persist_directory=CHROMA_PATH,
        embedding_function=emb,
        collection_metadata={"dimension": dim, "embedding_id": embedding_id, "version": version},
    )

    # Dimension validation: if collection exists with a different dimension, fail fast.
    try:
        col = store._collection  # internal but stable enough for our sanity check
        meta = getattr(col, "metadata", None) or {}
        existing_dim = meta.get("dimension")
        if existing_dim is not None and int(existing_dim) != int(dim):
            raise RuntimeError(
                f"Embedding dimension mismatch for collection '{cname}': "
                f"collection dim={existing_dim}, current embedding dim={dim}. "
                f"Use a new CHROMA_COLLECTION_VERSION or delete the DB."
            )
    except Exception:
        # If we can't read metadata, we won't block; Chroma will error if incompatible at query/add time.
        pass

    return store, emb, cname, dim


def choose_splitter(total_chars: int) -> RecursiveCharacterTextSplitter:
    if total_chars >= LONG_DOC_THRESHOLD_CHARS:
        return RecursiveCharacterTextSplitter(chunk_size=LONG_DOC_CHUNK_SIZE, chunk_overlap=LONG_DOC_CHUNK_OVERLAP)
    return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def load_pdf_documents(tmp_path: str, source_name: str, source_path: str) -> List[Document]:
    docs: List[Document] = []
    if PyPDFLoader is not None:
        loader = PyPDFLoader(tmp_path)
        for d in loader.load():
            md = dict(d.metadata or {})
            # Normalize metadata keys
            md["source"] = source_name
            md["source_path"] = source_path
            # PyPDFLoader uses "page" key
            docs.append(Document(page_content=d.page_content, metadata=md))
        return docs

    if PdfReader is None:
        raise RuntimeError("Neither langchain-community PyPDFLoader nor pypdf is installed. pip install -U langchain-community pypdf")

    reader = PdfReader(tmp_path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        docs.append(Document(page_content=text, metadata={"source": source_name, "source_path": source_path, "page": i}))
    return docs


def load_text_document(raw: bytes, source_name: str, source_path: str) -> List[Document]:
    text = raw.decode("utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": source_name, "source_path": source_path, "page": None})]


def doc_id(source_path: str, page: Optional[int], chunk: int) -> str:
    base = f"{source_path}|{page}|{chunk}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]


def delete_by_source(store: Chroma, source_path: str) -> None:
    # Chroma supports deleting by metadata filter
    store._collection.delete(where={"source_path": source_path})


# ----------------------------
# Retrieval (LangChain 1.x compatible)
# ----------------------------
def retrieve_docs(
    store: Chroma,
    query: str,
    top_k: int,
    fetch_k: int,
    search_type: str,
    mmr_lambda: float,
) -> List[Document]:
    # as_retriever returns a VectorStoreRetriever which is a Runnable in LangChain 1.x
    kwargs: Dict[str, Any] = {}
    if search_type == "mmr":
        kwargs = {"k": top_k, "fetch_k": fetch_k, "lambda_mult": mmr_lambda}
    else:
        kwargs = {"k": top_k}

    retriever = store.as_retriever(search_type=search_type, search_kwargs=kwargs)

    # LangChain 1.x: prefer invoke()
    return retriever.invoke(query)


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
        text = d.page_content or ""
        piece = header + text.strip()
        if not piece.strip():
            continue
        if used + len(piece) > max_chars and blocks:
            break
        blocks.append(piece)
        used += len(piece)
        sources.append({"source": source, "page": page, "chunk": chunk, "source_path": md.get("source_path")})
    return "\n\n".join(blocks), sources


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
        store, emb, cname, dim = get_store(embedding_id, version)
        # Try a cheap call
        _ = embedding_dimension(emb)
        return jsonify({"status": "ready", "collection": cname, "dimension": dim, "embedding_id": embedding_id, "version": version})
    except Exception as e:
        return jsonify({"status": "not_ready", "error": str(e)}), 503


@app.get("/sources")
def sources():
    embedding_id = request.args.get("embedding_id", DEFAULT_EMBEDDING_ID)
    version = request.args.get("version", DEFAULT_VERSION)
    store, _, cname, dim = get_store(embedding_id, version)

    # Chroma doesn't have "distinct" query; get all metadatas and dedupe.
    # For large collections, this can be heavy; acceptable for reconcile intervals.
    data = store._collection.get(include=["metadatas"])
    metadatas = data.get("metadatas", []) or []
    uniq = sorted({(md or {}).get("source_path") for md in metadatas if (md or {}).get("source_path")})
    return jsonify({"count": len(uniq), "source_paths": uniq, "collection": cname, "dimension": dim})


@app.post("/delete")
def delete():
    payload = request.get_json(silent=True) or {}
    source_path = payload.get("source_path")
    if not source_path:
        return jsonify({"error": "source_path is required"}), 400

    embedding_id = payload.get("embedding_id", DEFAULT_EMBEDDING_ID)
    version = payload.get("version", DEFAULT_VERSION)

    store, _, cname, dim = get_store(embedding_id, version)
    delete_by_source(store, source_path)
    return jsonify({"status": "ok", "deleted_source_path": source_path, "collection": cname, "dimension": dim})


@app.post("/ingest")
def ingest():
    # Required: file + embedding_id + version + source_path.
    # We validate early so bad requests return quickly (and do not trigger
    # expensive model initialization with defaults).
    if "file" not in request.files:
        return jsonify({"error": "multipart form-data must include 'file'"}), 400

    missing = [k for k in ("embedding_id", "version", "source_path") if not (request.form.get(k) or "").strip()]
    if missing:
        return jsonify({"error": "missing required form fields", "missing": missing}), 400


    f = request.files["file"]
    raw = f.read()
    filename = f.filename or "uploaded_file"
    source_path = request.form.get("source_path")

    embedding_id = request.form.get("embedding_id")
    version = request.form.get("version")

    store, _, cname, dim = get_store(embedding_id, version)

    # Remove prior vectors for this source_path (upsert behavior)
    delete_by_source(store, source_path)

    ext = os.path.splitext(filename)[1].lower()

    # Load documents
    docs: List[Document] = []
    if ext == ".pdf":
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            docs = load_pdf_documents(tmp_path, filename, source_path)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    else:
        docs = load_text_document(raw, filename, source_path)

    total_chars = sum(len(d.page_content or "") for d in docs)
    splitter = choose_splitter(total_chars)

    chunks: List[Document] = []
    for d in docs:
        # Split each page/document while preserving metadata
        page_docs = splitter.split_documents([d])
        chunks.extend(page_docs)

    # Add chunk index metadata + deterministic IDs
    for idx, d in enumerate(chunks):
        md = dict(d.metadata or {})
        md["chunk"] = idx
        d.metadata = md

    ids = [doc_id((d.metadata or {}).get("source_path", source_path), (d.metadata or {}).get("page"), (d.metadata or {}).get("chunk", 0)) for d in chunks]

    store.add_documents(chunks, ids=ids)

    return jsonify({
        "status": "ok",
        "collection": cname,
        "dimension": dim,
        "embedding_id": embedding_id,
        "version": version,
        "source": filename,
        "source_path": source_path,
        "chunks_added": len(chunks),
        "total_chars": total_chars,
        "chunk_size": splitter._chunk_size,  # informative
        "chunk_overlap": splitter._chunk_overlap,
    })


@app.post("/retrieve")
def retrieve():
    payload = request.get_json(silent=True) or {}
    query = payload.get("query")
    if not query:
        return jsonify({"error": "query is required"}), 400

    embedding_id = payload.get("embedding_id", DEFAULT_EMBEDDING_ID)
    version = payload.get("version", DEFAULT_VERSION)

    top_k = int(payload.get("top_k", DEFAULT_TOP_K))
    fetch_k = int(payload.get("fetch_k", DEFAULT_FETCH_K))
    search_type = payload.get("search_type", DEFAULT_SEARCH_TYPE)
    mmr_lambda = float(payload.get("mmr_lambda", DEFAULT_MMR_LAMBDA))
    max_chars = int(payload.get("max_context_chars", DEFAULT_MAX_CONTEXT_CHARS))

    store, _, cname, dim = get_store(embedding_id, version)
    docs = retrieve_docs(store, query, top_k=top_k, fetch_k=fetch_k, search_type=search_type, mmr_lambda=mmr_lambda)
    context, sources = format_context(docs, max_chars)

    return jsonify({
        "context": context,
        "sources": sources,
        "collection": cname,
        "dimension": dim,
        "embedding_id": embedding_id,
        "version": version,
        "top_k": top_k,
        "fetch_k": fetch_k,
        "search_type": search_type,
        "mmr_lambda": mmr_lambda,
    })


@app.post("/answer")
def answer():
    payload = request.get_json(silent=True) or {}
    question = payload.get("query")
    if not question:
        return jsonify({"error": "query is required"}), 400


    embedding_id = payload.get("embedding_id", DEFAULT_EMBEDDING_ID)
    version = payload.get("version", DEFAULT_VERSION)

    llm_id = payload.get("llm_id", DEFAULT_LLM_ID)
    top_k = int(payload.get("top_k", DEFAULT_TOP_K))
    fetch_k = int(payload.get("fetch_k", DEFAULT_FETCH_K))
    search_type = payload.get("search_type", DEFAULT_SEARCH_TYPE)
    mmr_lambda = float(payload.get("mmr_lambda", DEFAULT_MMR_LAMBDA))
    max_chars = int(payload.get("max_context_chars", DEFAULT_MAX_CONTEXT_CHARS))

    store, _, cname, dim = get_store(embedding_id, version)
    docs = retrieve_docs(store, question, top_k=top_k, fetch_k=fetch_k, search_type=search_type, mmr_lambda=mmr_lambda)
    context, sources = format_context(docs, max_chars)

    system = (
        "You are a precise assistant doing retrieval-augmented generation (RAG). "
        "Answer ONLY using the provided CONTEXT. "
        "If the context does not contain the answer, say so. "
        "When referencing facts, cite them by mentioning Source + page."
    )

    user = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nReturn your answer."

    llm = make_llm(llm_id)
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])

    # resp may be AIMessage
    answer_text = getattr(resp, "content", None) or str(resp)

    return jsonify({
        "answer": answer_text,
        "sources": sources,
        "collection": cname,
        "dimension": dim,
        "embedding_id": embedding_id,
        "version": version,
        "llm_id": llm_id,
    })


if __name__ == "__main__":
    os.makedirs(CHROMA_PATH, exist_ok=True)
    app.run(host=HOST, port=PORT, debug=True)
