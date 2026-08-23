# Local RAG on this Mac — Model Selection & Setup Guide

_Generated 2026-08-09. Based on the models actually found installed in Ollama (`~/.ollama`) and LM Studio (`~/.lmstudio`) on this machine._

> **Implementation status (2026-08-09).** The recommendations below have been implemented in this repo, not in the standalone
> tutorial project of section 4. `bge-m3` is now the active embedder (`EMBEDDING_ID=ollama:qllama/bge-m3:latest`), LM Studio
> models are reachable via the `lmstudio:` LLM prefix, and cross-encoder reranking is available via `RERANKER_ID`
> (default `none:`). See `README.md` and `CLAUDE.md` for the shipped interfaces. Sections 4.2-4.6 remain a from-scratch
> tutorial and do **not** describe this codebase.

---

## 1. What's installed on this machine

**Ollama** (`~/.ollama/models`, ~52 GB total)

| Model | Size | Type / role |
|---|---|---|
| `qwen3:30b` | 17.3 GB | MoE generator (~3B active) |
| `gpt-oss:20b` (= `gpt-oss:latest`) | 12.8 GB | MoE reasoning generator |
| `gemma4:31b-mlx` | 17.4 GB | Dense generator (MLX quant) |
| `llama3.2:latest` | 1.9 GB | Small generator (~3B) |
| `deepseek-r1:1.5b` | 1.0 GB | Tiny reasoning distill |
| `nomic-embed-text:v1.5` | 262 MB | **Embedding model** |
| `qllama/bge-m3` | 605 MB | **Embedding model** |

**LM Studio** (`~/.lmstudio/models`)

| Model | Size | Type / role |
|---|---|---|
| `dolphin-2.9.1-llama-3-70b` (Q4_K_S) | 38 GB | Large uncensored generator |
| `Qwen3.6-27B` (Q4_K_M) | 16 GB | Multimodal generator (+mmproj) |
| `gpt-oss-20b-MXFP4-Q8` (MLX) | 11.4 GB | MoE reasoning generator |
| `Magistral-Small-2509` (MLX 4bit) | 13 GB | ~24B reasoning generator |
| `gemma-4-12B-it-QAT` (Q4_0) | 6.5 GB | Multimodal generator (+mmproj) |

> RAG is a two-part problem: a **retriever** (embedding model) finds relevant chunks, and a **generator** (LLM) writes the grounded answer. You have two dedicated embedders and many generators — so a strong stack can be built entirely from what's already here.

---

## 2. The 4 best local models for RAG (the selection)

| Pick | Model | Where | Role in the RAG pipeline |
|---|---|---|---|
| 1 | **`bge-m3`** (`qllama/bge-m3`) | Ollama | **Retriever / embeddings** |
| 2 | **`qwen3:30b`** (Qwen3-30B-A3B) | Ollama | **Primary generator** |
| 3 | **`gpt-oss:20b`** | Ollama | Reasoning / alternate generator |
| 4 | **`gemma-4-12B-it-QAT`** | LM Studio | Lightweight / fast generator |

### Line of thinking

RAG quality is bounded first by **retrieval**, then by the generator's **faithfulness to the retrieved context** (not raw size). I optimized for: retrieval strength, instruction-following & grounding, context length, and latency on Apple Silicon.

- **`bge-m3` as the retriever.** The single most impactful choice. It's the 2026 default production retriever: MIT-licensed, 100+ languages, 8K context, and it produces *dense + sparse + multi-vector* representations in one model — meaning it can drive hybrid search (semantic + keyword) from a single embedder. `nomic-embed-text` is a fine, lighter fallback, but bge-m3 retrieves better, which is where RAG lives or dies.
- **`qwen3:30b` (Qwen3-30B-A3B) as primary generator.** The consensus RAG sweet spot for local machines. It's a Mixture-of-Experts model with only ~3B active parameters, so it answers with the quality of a much larger model but at the *speed* of a small one — ideal when each query stuffs thousands of tokens of retrieved context into the prompt. Very long context and strong instruction-following make it stick to the sources.
- **`gpt-oss:20b` as the reasoning option.** MoE, efficient, and stronger at multi-step synthesis when a query needs the model to reason *over* several retrieved chunks (comparisons, "why", multi-hop). Good open-weight second opinion alongside Qwen.
- **`gemma-4-12B-it-QAT` as the fast/light generator.** QAT 4-bit at 6.5 GB leaves plenty of memory for the vector store + embedding model + KV cache. Good grounding, long context, and it's multimodal (the `mmproj` file) — useful if you later want to RAG over screenshots/diagrams. Pick this when you want snappy answers or to run everything comfortably alongside other apps.

### Why not the others
- **`dolphin-70b` (38 GB):** highest raw quality but heavy and slow per query; "uncensored" fine-tunes add nothing for grounded document QA. Keep as a "hard question" fallback, not the default.
- **`Magistral-Small` / `deepseek-r1:1.5b`:** reasoning-tuned models tend to *over-think* and drift from the sources in RAG, hurting faithfulness; r1:1.5b is also too small to synthesize reliably.
- **`llama3.2:3b`:** great for speed tests, but weaker grounding than Gemma-12B at a small memory saving.
- **`gemma4:31b-mlx`, `Qwen3.6-27B`:** solid but redundant given Qwen3-30B and Gemma-12B already cover the quality/speed spread. Qwen3.6-27B is a good multimodal alternative to Gemma if you need bigger.

**Recommended default stack:** `bge-m3` (retrieve) → `qwen3:30b` (generate), with `gemma-4-12B` for speed and `gpt-oss:20b` for hard reasoning queries.

---

## 3. Comparison with other openly accessible models (that run on this Mac)

Your local set is already very close to the 2026 state-of-the-art. Here's how it stacks up, plus a few worth downloading.

**Embedding / retrieval**

| Model | vs. your `bge-m3` | Verdict |
|---|---|---|
| **Qwen3-Embedding-0.6B/4B/8B** | Tops the MTEB leaderboard in 2026 (8B ~70.6, beats OpenAI & Google APIs); user-defined dims 32–1024 | **Installed (0.6B).** Use `EMBEDDING_ID=hf:Qwen/Qwen3-Embedding-0.6B` or `bash scripts/run_qwen3_emb.sh`. Emits 1024 dims, same width as bge-m3. |
| Nomic Embed v2 | You have v1.5; v2 is MoE and better on long docs | Minor upgrade |
| Snowflake Arctic-embed-l, mxbai-embed-large | Comparable English-only retrievers | No need — bge-m3 covers it |

**Rerankers** — was the biggest gap; now closed.

| Model | Why it matters | 
|---|---|
| **BGE-reranker-v2-m3** | The standard partner to bge-m3. Re-scores the top-K retrieved chunks before generation. **Installed and wired**: set `RERANKER_ID=ce:BAAI/bge-reranker-v2-m3`, or pass `reranker_id` per request. Costs ~6.5 s on first load, then ~0.6 s per query at `fetch_k=24`. |
| **Qwen3-Reranker (0.6B/4B)** | Newer, top-tier reranking; pairs with Qwen3-Embedding. **Not wired**: unlike BGE it is a causal LM scored on yes/no logits, not a `sentence-transformers` CrossEncoder, so the `ce:` provider cannot load it as-is. |

**Generators**

| Model | vs. your picks | Verdict |
|---|---|---|
| **Cohere Command-R (35B / 7B)** | The one open model *purpose-built for RAG*: returns **inline citations** to sources instead of hallucinating them; every other model needs prompt engineering | **Worth adding** if verifiable citations matter |
| Llama 3.3 70B / 3.1 8B | Strong, but Qwen3-30B-A3B matches quality at far higher speed | Optional |
| Phi-4 (14B), Mistral Small 3 (24B) | Good mid-size grounders | Comparable to what you have |

**Bottom line:** Your generators are already at/near SOTA (Qwen3-30B-A3B is *the* recommended local RAG model). To improve the pipeline, spend effort on retrieval, not generation: **add a reranker (`bge-reranker-v2-m3`)** and optionally **`Qwen3-Embedding-4B`** and **Command-R** for citations.

---

## 4. Running them locally with Python in a `uv` environment

This sets up a minimal but complete RAG pipeline: **ingest → embed → store → retrieve → generate**, using your local models through Ollama's OpenAI-compatible API.

### 4.1 Prerequisites

- **Ollama** running (it already is — the models are there). Start the server if needed:
  ```bash
  ollama serve            # runs on http://localhost:11434
  ollama list             # confirm bge-m3, qwen3:30b, gpt-oss:20b are present
  ```
- **`uv`** installed. If not:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- For the LM Studio model (`gemma-4-12B`): open LM Studio → **Developer** tab → **Start Server** (default `http://localhost:1234/v1`). Ollama covers picks 1–3; LM Studio only needed for pick 4.

### 4.2 Create the project and environment with `uv`

```bash
mkdir local-rag && cd local-rag

# create a project + pinned Python
uv init --python 3.12
uv venv

# add dependencies (uv resolves + installs into .venv)
uv add langchain langchain-community langchain-ollama \
       langchain-chroma chromadb ollama \
       pypdf unstructured tiktoken
```

`uv add` writes everything to `pyproject.toml` and locks it in `uv.lock`, so the environment is reproducible. Run any script with `uv run python …` (no manual `activate` needed).

> Prefer to pin exact versions? Use the `requirements.txt` in section 4.6 with `uv pip install -r requirements.txt`.

### 4.3 Put some documents in

```bash
mkdir docs
# drop your .pdf / .txt / .md files into ./docs
```

### 4.4 Build the index (ingest → embed → store)

Create `ingest.py`:

```python
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DOCS_DIR = "docs"
DB_DIR = "chroma_db"

# 1) Load documents (pdf + text/markdown)
docs = []
docs += DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader).load()
docs += DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader).load()
docs += DirectoryLoader(DOCS_DIR, glob="**/*.md",  loader_cls=TextLoader).load()
print(f"Loaded {len(docs)} document pages")

# 2) Chunk — good defaults for bge-m3 (8K context, but keep chunks tight for precision)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

# 3) Embed with your local bge-m3 (served by Ollama)
embeddings = OllamaEmbeddings(model="qllama/bge-m3")   # or "nomic-embed-text:v1.5"

# 4) Store in a local Chroma vector DB (persists to ./chroma_db)
Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
print(f"Index written to ./{DB_DIR}")
```

Run it:

```bash
uv run python ingest.py
```

### 4.5 Ask questions (retrieve → generate)

Create `ask.py`:

```python
import sys
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DB_DIR = "chroma_db"

embeddings = OllamaEmbeddings(model="qllama/bge-m3")
store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = store.as_retriever(search_kwargs={"k": 5})

# Primary generator: Qwen3-30B-A3B. Swap to "gpt-oss:20b" for hard reasoning.
llm = ChatOllama(model="qwen3:30b", temperature=0.1)

prompt = ChatPromptTemplate.from_template(
    "You are a precise assistant. Answer ONLY from the context below. "
    "If the answer isn't in the context, say you don't know. Cite the source snippets.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

def format_docs(docs):
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

question = " ".join(sys.argv[1:]) or "What are these documents about?"
print(chain.invoke(question))
```

Run it:

```bash
uv run python ask.py "What does the contract say about termination?"
```

**To use the LM Studio model (pick 4, `gemma-4-12B`)** instead of Ollama, point an OpenAI-compatible client at LM Studio's server:

```python
from langchain_openai import ChatOpenAI   # uv add langchain-openai
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",                     # any non-empty string
    model="gemma-4-12b-it-qat",              # name shown in LM Studio's server tab
    temperature=0.1,
)
```

### 4.6 Optional: pinned `requirements.txt`

```text
langchain
langchain-community
langchain-ollama
langchain-openai
langchain-chroma
chromadb
ollama
pypdf
unstructured
tiktoken
```

```bash
uv pip install -r requirements.txt
```

### 4.7 Adding a reranker (implemented — see `backend/reranker.py`)

The biggest quality win is a reranker between retrieval and generation.

**Correction:** an earlier draft of this section said `ollama pull bge-reranker-v2-m3`. That does not work — Ollama's
library serves generation and embedding models but has no reranker, and the URL returns 404. Rerankers are cross-encoders
and must come from Hugging Face:

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3')"
```

In this repo it is already wired. Retrieval keeps its hybrid blend as the candidate selector, then the cross-encoder
re-scores the top `fetch_k` and makes the final `top_k` cut:

```bash
# server-wide
export RERANKER_ID="ce:BAAI/bge-reranker-v2-m3"

# or per request
curl -s -X POST http://127.0.0.1:5050/answer \
  -H 'Content-Type: application/json' \
  -d '{"query":"...","llm_id":"ollama:qwen3:30b","reranker_id":"ce:BAAI/bge-reranker-v2-m3"}'
```

The default is `none:`, which leaves retrieval exactly as it was. Cost measured on this machine: ~6.5 s to load the model
once, then ~0.39 s at 12 candidates and ~0.64 s at 24.

---

## Quick reference — the stack

```
Documents ──► chunk ──► bge-m3 (embed) ──► Chroma (store)
Query ──► bge-m3 (embed) ──► Chroma (top-k) ──► [reranker] ──► qwen3:30b (answer)
                                                              └► fast: gemma-4-12B
                                                              └► reasoning: gpt-oss:20b
```

All local, all offline, all `uv run`.

As shipped in this repo:

| Stage | Setting |
|---|---|
| Embed | `EMBEDDING_ID=ollama:qllama/bge-m3:latest` (`scripts/run_bge_m3.sh`); alternative `hf:Qwen/Qwen3-Embedding-0.6B` |
| Rerank | `RERANKER_ID=ce:BAAI/bge-reranker-v2-m3` (default `none:`) |
| Generate | `llm_id=ollama:qwen3:30b`, `ollama:gpt-oss:20b`, or `lmstudio:google/gemma-4-12b-qat` |

Switching `EMBEDDING_ID` creates a *new* Chroma collection rather than mixing vectors, so changing embedder means
re-ingesting. Keep it consistent between the API and `folder_watcher.py`.
