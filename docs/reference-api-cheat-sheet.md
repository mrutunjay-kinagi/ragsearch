# API Reference & Cheat Sheet

**Status:** Expanded in Slice C with detailed contracts, schemas, and examples.

Quick reference for core ragsearch APIs.

## Core Interfaces

### `setup()` – Initialize Engine

```python
engine = setup(
    data_path,
    llm_api_key,
    embedding_provider="cohere",
    llm_provider="cohere",
    use_chromadb=False,
    chromadb_sqlite_path=None,
    chromadb_collection_name=None,
)
```

**Returns:** `RagSearchEngine` instance.

See [Troubleshooting](./troubleshooting.md) for configuration help.

### `engine.search(query, top_k=5) -> List[Dict]`

Retrieve top-k most similar records to query.

**Output schema:**
```json
[
  {
    "metadata": { "...": "original columns excluding embedding" },
    "citation": {
      "record_id": 0,
      "source_path": "filename",
      "parser_name": "fallback/csv",
      "excerpt": "first 200 chars of text"
    },
    "similarity": 0.95
  }
]
```

### `engine.answer(query, top_k=5) -> Dict[str, Any]`

Answer a question grounded in retrieved sources.

**Output schema:**
```json
{
  "question": "user query",
  "answer": "LLM-generated answer",
  "results": [ "search results (see above)" ],
  "citations": [ "citation objects only" ],
  "context": "formatted context block passed to LLM"
}
```

---

**See also:** [Quickstart Guide](./quickstart.md) | [Cookbook](./cookbook-dataset-analytics.md) | [Troubleshooting](./troubleshooting.md)
