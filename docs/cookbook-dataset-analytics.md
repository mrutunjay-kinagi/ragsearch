# Dataset Analytics Cookbook

Complete end-to-end workflow for RAG-powered dataset analysis, from setup through evaluation.

## Overview

This cookbook demonstrates how to:
1. **Setup** – Initialize engine with structured data (CSV/Parquet)
2. **Index** – Build embeddings and vector indexes with optional fallback parsing
3. **Retrieve** – Query and get ranked, grounded results
4. **Generate** – Get LLM answers with citations
5. **Evaluate** – Run regression gates and benchmark tracking

## Example: Analyzing a Recipe Dataset

### Step 1: Prepare Your Data

```python
import pandas as pd

# Load CSV data (or Parquet/JSON)
df = pd.read_csv("recipes.csv")
print(f"Loaded {len(df)} recipes")
print(df.head())

# Inspect columns
print(df.columns.tolist())
# Output: ['name', 'ingredients', 'steps', 'prep_time', 'servings', 'difficulty', ...]
```

### Step 2: Initialize the Engine

```python
from ragsearch import setup

# Setup with Cohere embeddings (default)
engine = setup(
    data_path="recipes.csv",
    llm_api_key="your-cohere-api-key",
    embedding_provider="cohere",
    llm_provider="cohere"
)
# Or use sentence-transformers for local embeddings:
# engine = setup("recipes.csv", embedding_provider="sentence_transformers")
```

### Step 3: Query and Answer

```python
# Single query
result = engine.answer(
    query="What are gluten-free pasta dishes with under 30 minutes prep?",
    top_k=5
)

print("Answer:", result["answer"])
print(f"\nCitations ({len(result['citations'])} sources):")
for i, cit in enumerate(result["citations"], start=1):
    print(f"  [{i}] {cit['excerpt'][:100]}...")
```

**Output:**
```
Answer: Based on the recipes in our database, here are some quick gluten-free 
pasta options: [1] shows a simple recipe under 20 minutes. [2] provides a more 
elaborate option still within the 30-minute window.

Citations (2 sources):
  [1] Gluten-Free Pasta Primavera: Fresh vegetables, olive oil, 3 ingredients. 
      Prep: 15 minutes...
  [2] Chickpea Pasta with Marinara: Gluten-free pasta, marinara sauce, 
      chickpeas. Prep: 25 minutes...
```

## Advanced Example: Batch Evaluation

### Run Multiple Queries and Track Results

```python
import json
from pathlib import Path

# Define test cases
test_cases = [
    {
        "query": "vegan appetizers under 10 minutes",
        "top_k": 5,
        "min_results": 1,
        "min_citations": 1
    },
    {
        "query": "desserts with chocolate",
        "top_k": 3,
        "min_results": 1,
        "min_citations": 1
    },
    {
        "query": "budget-friendly family dinners",
        "top_k": 5,
        "min_results": 2,  # Expect at least 2 relevant recipes
        "min_citations": 2
    }
]

# Run evaluation
from ragsearch.evaluation import run_regression_gates, EvaluationThresholds

summary = run_regression_gates(
    engine,
    test_cases,
    EvaluationThresholds(min_results=1, min_citations=1)
)

print(f"Passed: {summary['passed_cases']}/{summary['total_cases']}")
for result in summary["results"]:
    status = "✓" if result["pass"] else "✗"
    print(f"{status} {result['query']}: {result['observed_results']} results")
```

**Output:**
```
Passed: 3/3
✓ vegan appetizers under 10 minutes: 5 results
✓ desserts with chocolate: 3 results
✓ budget-friendly family dinners: 5 results
```

## Using Fallback Parsers for Unstructured Content

If your data includes unstructured files (HTML, PDF, DOCX, TXT):

```python
from ragsearch import setup

# Parser fallback chain: LiteParse → fallback parsers
# LiteParse used if available; fallback used otherwise
engine = setup(
    data_path="path/to/mixed_data.json",  # Can include file refs
    llm_api_key="...",
)

# The engine automatically handles parsing based on file type
```

## Observability and Debugging

### Access Indexing Diagnostics

```python
# Check what was indexed
diagnostics = engine.indexing_diagnostics
print(f"Total records: {diagnostics['total_records']}")
print(f"Embedded: {diagnostics['embedded_records']}")
print(f"Reused (cached): {diagnostics['reused_records']}")
print(f"New: {diagnostics['new_records']}")
print(f"Changed: {diagnostics['changed_records']}")
```

### View Observability Events

```python
# Access low-level events emitted during indexing/retrieval/generation
events = engine.observability_events
for event in events[-5:]:  # Last 5 events
    print(f"{event['stage']}: {event['event']}")
    print(f"  Payload: {event['payload']}")
```

## Benchmark Integration

### Persist Run Results

The `.benchmarks/` directory tracks all evaluation runs:

```python
from datetime import datetime
from pathlib import Path
import json

# Your benchmark results are automatically saved in:
# .benchmarks/runs/<timestamp>_<name>/summary.json
# .benchmarks/history/metrics.csv

# Review trends
import pandas as pd
metrics = pd.read_csv(".benchmarks/history/metrics.csv")
print(metrics.tail())

# Latest run
runs_dir = Path(".benchmarks/runs")
latest_run = sorted(runs_dir.glob("*"))[-1]
summary = json.loads((latest_run / "summary.json").read_text())
print(f"Run: {summary['dataset']}")
print(f"Result: {'PASS' if summary['pass'] else 'FAIL'}")
```

## Known Limitations

1. **Deterministic local stubs** – Example uses keyword-based embeddings; real providers offer better semantic understanding
2. **Source precision** – Answer quality depends on embedding model; source retrieval is keyword-based in examples
3. **Citation accuracy** – Answers can look good while citations point to wrong sources; always verify sources manually

---

**See also:** [Quickstart Guide](./quickstart.md) | [API Reference](./reference-api-cheat-sheet.md) | [Troubleshooting](./troubleshooting.md) | [Jupyter Notebook](./notebook-dataset-analytics.ipynb)
