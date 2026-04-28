# RAG exchange

## User query

```text
what's my name?
```

## Request sent to Flask API

```json
{
  "url": "http://127.0.0.1:5000/answer",
  "body": {
    "query": "what's my name?",
    "llm_id": "openai:gpt-4.1-mini",
    "history": [
      {
        "role": "user",
        "content": "what plastics are considered for the CBAM scope extension?"
      },
      {
        "role": "assistant",
        "content": "**Endpoint:** `http://127.0.0.1:5000/answer`  ·  **Latency:** `2.38s`\n\nThe plastics (polymers) considered for the CBAM scope extension include:\n\n- Polybutadiene  \n- Polyethylene glycol  \n- Polypropylene glycol  \n\nAdditionally, the study mentions the inclusion of precursors which are themselves polymers, such as polybutadiene for ABS and polyols as precursors to polyurethanes. The approach focuses on assessing the most popular \"pure\" polymers individually for inclusion in the CBAM. Industry concerns about partial coverage potentially shifting consumption to non-covered polymers are noted, but the assessment was made in an unbiased manner.\n\nEvidence:  \n- CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf, page 73, chunk 150  \n- Annex 4 - Chemicals Note FINAL.pdf, page 126, chunk 197"
      },
      {
        "role": "user",
        "content": "My name is Jose"
      },
      {
        "role": "assistant",
        "content": "**Endpoint:** `http://127.0.0.1:5000/answer`  ·  **Latency:** `2.52s`\n\nThe plastics (polymers) considered for the CBAM scope extension include:\n\n- Polybutadiene  \n- Polyethylene glycol (PEG)  \n- Polypropylene glycol (PPG)  \n\nThese polymers are assessed individually as the most popular \"pure\" polymers for potential inclusion in the CBAM. The assessment also considers precursors that are polymers themselves, such as polybutadiene for ABS and polyols as precursors to polyurethanes (PU). The approach aims to avoid partial coverage that could shift consumption to non-covered polymers.\n\nEvidence:  \n- CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf, page 73, chunk 150  \n- Annex 4 - Chemicals Note FINAL.pdf, page 126, chunk 197"
      }
    ]
  }
}
```

## Response from Flask API

```json
{
  "answer": "Your name is Jose.\n\nEvidence: Prior conversation turn.",
  "backend": "chroma",
  "collection": "my_local_knowledge_base__v1__hf_baai_bge_large_en_v1_5",
  "embedding_id": "hf:BAAI/bge-large-en-v1.5",
  "llm_id": "openai:gpt-4.1-mini",
  "sources": [
    {
      "chunk": 89,
      "page": 54,
      "source": "Annex 4 - Chemicals Note FINAL.pdf",
      "source_path": "/Users/jm/PythonProjects/_WORK/Local_RAG/watched_folder/Annex 4 - Chemicals Note FINAL.pdf"
    },
    {
      "chunk": 271,
      "page": 128,
      "source": "CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf",
      "source_path": "/Users/jm/PythonProjects/_WORK/Local_RAG/watched_folder/CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf"
    },
    {
      "chunk": 118,
      "page": 73,
      "source": "Annex 4 - Chemicals Note FINAL.pdf",
      "source_path": "/Users/jm/PythonProjects/_WORK/Local_RAG/watched_folder/Annex 4 - Chemicals Note FINAL.pdf"
    },
    {
      "chunk": 280,
      "page": 136,
      "source": "CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf",
      "source_path": "/Users/jm/PythonProjects/_WORK/Local_RAG/watched_folder/CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf"
    },
    {
      "chunk": 263,
      "page": 124,
      "source": "CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf",
      "source_path": "/Users/jm/PythonProjects/_WORK/Local_RAG/watched_folder/CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf"
    },
    {
      "chunk": 267,
      "page": 126,
      "source": "CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf",
      "source_path": "/Users/jm/PythonProjects/_WORK/Local_RAG/watched_folder/CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf"
    }
  ],
  "version": "v1"
}
```
