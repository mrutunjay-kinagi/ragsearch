# API Reference & Cheat Sheet

Complete contract and usage reference for ragsearch core APIs.

## Setup and Initialization

### `setup()`

Factory function to initialize a RAG engine with data and providers.

```python
from ragsearch import setup

setup(
  data_path: Path,
  llm_api_key: str,
  use_chromadb: bool = False,
  chromadb_sqlite_path: str = None,
  chromadb_collection_name: str = None,
  embeddings_dir: str = None,
  chunking_strategy: ChunkingStrategy = None,
  reranker: Reranker = None,
  observability_max_events: int = 1000,
  embedding_provider: str = "cohere",
  embedding_model_name: str = None,
  embedding_api_key: str = None,
  embedding_base_url: str = None,
  llm_provider: str = "cohere",
  llm_model_name: str = None,
  llm_base_url: str = None,
) -> RagSearchEngine
```

**Parameters:**

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `data_path` | Path | required | CSV, JSON, or Parquet file |
| `llm_api_key` | str | required | Required by the current setup contract |
| `embedding_provider` | str | "cohere" | Options: "cohere", "sentence_transformers", "openai", "ollama" |
| `llm_provider` | str | "cohere" | Options: "cohere", "openai", "ollama" |
| `use_chromadb` | bool | False | Use ChromaDB instead of FAISS |
| `embeddings_dir` | str | None | Directory for embedding manifests and cache files |
| `chunking_strategy` | ChunkingStrategy | None | Optional retrieval chunking strategy |
| `reranker` | Reranker | None | Optional result reranker |
| `observability_max_events` | int | 1000 | Max retained in-memory observability events |
| `embedding_model_name` | str | None | Provider-specific embedding model name |
| `embedding_api_key` | str | None | Optional embedding-provider API key override |
| `embedding_base_url` | str | None | Optional embedding-provider base URL |
| `llm_model_name` | str | None | Provider-specific chat model name |
| `llm_base_url` | str | None | Optional LLM base URL |

**Returns:** Initialized `RagSearchEngine` ready for queries.

**Notes:**
- `setup()` currently enforces `data_path` as a `Path` object.
- The parameter names above match the live public contract (`embedding_model_name`, `llm_model_name`, `embeddings_dir`).

**Example – Cohere (default):**
```python
from pathlib import Path

engine = setup(Path("data.csv"), llm_api_key="your-api-key")
```

**Example – Local embeddings:**
```python
from pathlib import Path

engine = setup(
  Path("data.csv"),
  llm_api_key="your-api-key",
  embedding_provider="sentence_transformers",
)
```

**Example – OpenAI provider:**
```python
from pathlib import Path

engine = setup(
  Path("data.csv"),
    llm_api_key="sk-...",
    embedding_provider="openai",
    llm_provider="openai",
  embedding_model_name="text-embedding-3-small",
  llm_model_name="gpt-4o-mini"
)
```

---

## Core Methods

### `engine.search(query, top_k=5)`

Retrieve top-k most similar records to a query.

**Parameters:**
- `query` (str): Search query
- `top_k` (int): Number of results to return [default: 5]

**Returns:** `List[Dict]` with structure:

```python
[
  {
    "metadata": {
      "...original DataFrame columns except 'embedding'..."
    },
    "citation": {
      "record_id": 0,                    # Engine-specific row identifier
      "source_path": "data.csv",        # Source identifier
      "excerpt": "Name: Smith | Age: 25 | ..."  # First 200 chars of text
    },
    "similarity": 0.95                 # Similarity score; higher means more similar
  },
  # ... more results
]
```

**Example:**
```python
results = engine.search("female passengers first class", top_k=5)
for result in results:
    print(f"Similarity: {result['similarity']:.2f}")
    print(f"Source: {result['citation']['source_path']}")
```

---

### `engine.answer(query, top_k=5)`

Generate a grounded answer to a question using retrieved sources.

**Parameters:**
- `query` (str): Question or prompt
- `top_k` (int): Number of sources to retrieve [default: 5]

**Returns:** `Dict[str, Any]` with structure:

```python
{
  "question": "female passengers first class?",
  "answer": "Based on the retrieved sources, female passengers in first...",
  "results": [
    # ... full search results (see above)
  ],
  "citations": [
    # ... citation objects only (see citation structure above)
  ],
  "context": "[1] source_path: ...\\n\\n[2]..."  # Formatted for LLM
}
```

**Example:**
```python
result = engine.answer("What happened to women on the Titanic?")
print(result["answer"])
for cit in result["citations"]:
    print(f"[{cit['record_id']}] {cit['excerpt'][:50]}...")
```

---

## Evaluation and Benchmarking

### `run_regression_gates(engine, cases, thresholds)`

Run deterministic evaluation gates against test cases.

```python
from ragsearch.evaluation import run_regression_gates, EvaluationThresholds

summary = run_regression_gates(
    engine=engine,
    cases=[
        {
            "query": "female passengers",
            "top_k": 5,
            "min_results": 1,
            "min_citations": 1
        },
        # ... more cases
    ],
    thresholds=EvaluationThresholds(min_results=1, min_citations=1)
)
```

**Returns:**
```python
{
    "total_cases": 3,
    "passed_cases": 3,
    "failed_cases": 0,
    "pass": True,
    "results": [
        {
            "query": "...",
            "top_k": 5,
            "observed_results": 5,
            "observed_citations": 5,
            "expected_min_results": 1,
            "expected_min_citations": 1,
          "passed": True
        },
        # ... one per case
    ]
}
```

---

## Engine Attributes

### Get Indexing Diagnostics

```python
diag = engine.indexing_diagnostics
print(f"Total records: {diag['total_records']}")
print(f"Embedded: {diag['embedded_records']}")
print(f"Reused (cached): {diag['reused_records']}")
print(f"New: {diag['new_records']}")
print(f"Changed: {diag['changed_records']}")
```

### Access Observability Events

```python
for event in engine.observability_events[-5:]:  # Last 5
    print(f"{event['stage']}: {event['event']}")
    print(f"Payload: {event['payload']}")
```

**Event stages:** `indexing`, `retrieval`, `generation`

---

## Common Patterns

### Batch Evaluation

```python
test_queries = [
    "female passengers",
    "first class",
    "survival outcomes"
]
for query in test_queries:
    result = engine.answer(query, top_k=3)
    print(f"{query}: {len(result['citations'])} sources")
```

### Save and Load Results

```python
import json
result = engine.answer("...")
json.dump(result, open("answer.json", "w"))
```

### Benchmark Tracking

The evaluation CLI prints a JSON summary to stdout. Dedicated benchmark runner scripts in this repository may persist summaries and history under `.benchmarks/`.

See [Benchmark Interpretation](./benchmark-interpretation.md) for details.

---

**See also:** [Quickstart Guide](./quickstart.md) | [Cookbook](./cookbook-dataset-analytics.md) | [Troubleshooting](./troubleshooting.md) | [Benchmark Guide](./benchmark-interpretation.md)
