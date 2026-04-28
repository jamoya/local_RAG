# RAG exchange

## User query

```text
what products of the chemical industry are considered for the scope extension of the CBAM?
```

## Request sent to Flask API

```json
{
  "url": "http://127.0.0.1:5000/answer",
  "body": {
    "query": "what products of the chemical industry are considered for the scope extension of the CBAM?",
    "llm_id": "openai:gpt-4o"
  }
}
```

## Response from Flask API

```json
{
  "answer": "The products of the chemical industry considered for the scope extension of the CBAM include:\n\n- Methanol\n- Polyethylene\n- Polypropylene\n- PET (Polyethylene terephthalate)\n- Styrene\n- PVC (Polyvinyl chloride)\n- Olefins (particularly ethylene and propylene)\n- Aromatics\n- Other organic chemicals with high production volumes or known high emissions\n\nAdditionally, goods already in the existing CBAM scope, such as hydrogen, ammonia, nitric acid, and urea, were considered as \"at the beginning of the value chain,\" and goods produced from them were also assessed (Source: CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf, pages 57, 66).\n\n**Evidence:**\n- Source: CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf, pages 57, 66",
  "backend": "chroma",
  "collection": "my_local_knowledge_base__v1__hf_baai_bge_large_en_v1_5",
  "embedding_id": "hf:BAAI/bge-large-en-v1.5",
  "llm_id": "openai:gpt-4o",
  "sources": [
    {
      "chunk": 119,
      "page": 57,
      "source": "CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf",
      "source_path": "/Users/jm/PythonProjects/_WORK/Local_RAG/watched_folder/CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf"
    },
    {
      "chunk": 138,
      "page": 66,
      "source": "CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf",
      "source_path": "/Users/jm/PythonProjects/_WORK/Local_RAG/watched_folder/CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf"
    },
    {
      "chunk": 24,
      "page": 13,
      "source": "CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf",
      "source_path": "/Users/jm/PythonProjects/_WORK/Local_RAG/watched_folder/CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf"
    },
    {
      "chunk": 363,
      "page": 188,
      "source": "CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf",
      "source_path": "/Users/jm/PythonProjects/_WORK/Local_RAG/watched_folder/CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf"
    },
    {
      "chunk": 49,
      "page": 23,
      "source": "CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf",
      "source_path": "/Users/jm/PythonProjects/_WORK/Local_RAG/watched_folder/CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf"
    },
    {
      "chunk": 241,
      "page": 114,
      "source": "CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf",
      "source_path": "/Users/jm/PythonProjects/_WORK/Local_RAG/watched_folder/CBAM horizontal extension Final Report _CLEAN_31.10.2025.pdf"
    }
  ],
  "version": "v1"
}
```
