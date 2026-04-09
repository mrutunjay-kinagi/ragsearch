# Troubleshooting Guide

Common issues, diagnosis, and solutions.

## Empty Results

**Problem:** `engine.search(query)` returns empty list.

**Decision Tree:**

```
Empty results?
├─ Is the engine initialized?  
│  └─ Check: engine.indexing_diagnostics['embedded_records'] > 0
│     └─ If 0: No data was indexed. Check data_path and DataFrame content.
│
├─ Does the data have text columns?
│  └─ Check: 'text' or 'combined_text' in engine.index_data.columns
│     └─ If No: Ensure DataFrame has textual columns before setup.
│
└─ Is the query covered by the embedding model?
   └─ Try simpler keywords or try different queries.
   └─ Real embedding models work better than keyword-based dummy models.
```

**Solutions:**

1. Verify indexing:
```python
print(f"Indexed rows: {len(engine.index_data)}")
print(engine.index_data[["text"]].head())
print(engine.indexing_diagnostics)
```

2. Try a different query with basic keywords.

3. Use a better embedding model:
```python
from pathlib import Path

engine = setup(
    Path("data.csv"),
    llm_api_key="...",
    embedding_provider="sentence_transformers",  # Better than dummy
)
```

---

## Citation Does Not Match Answer

**Problem:** Generated answer looks correct, but cited source seems wrong.

**Causes:**
1. Embedding model is weak (keyword-based, dummy) → retrieves false positives
2. Multiple similar records → top-k includes ambiguous matches
3. Query is ambiguous → embedding is not distinctive enough

**Why it happens:**
"Answer looks good" ≠ "Source is correct". LLMs can generate plausible text; source ranking is independent.

**Solutions:**

1. Use a stronger embedding model:
```python
# Production: Real semantic embeddings
from pathlib import Path

engine = setup(
    Path("data.csv"),
    llm_api_key="sk-...",
    embedding_provider="openai",
    embedding_model_name="text-embedding-3-small"
)
```

2. Lower `top_k` to reduce false positives:
```python
result = engine.answer(query, top_k=3)  # Instead of 5
```

3. Increase data quality and distinctiveness.

4. **Always verify sources manually** before trusting citations.

---

## Setup Failures

### "ModuleNotFoundError: No module named 'ragsearch'"

**Solution:**
```bash
# From repo root:
pip install -e .
# Or:
pip install /path/to/ragsearch
```

### "API key invalid" or provider not found

**Solution:**
```bash
# Check that required packages are installed:
pip install cohere           # For Cohere
pip install openai           # For OpenAI
pip install sentence-transformers  # For local embeddings
pip install ollama           # For Ollama
```

### "No data found in the provided DataFrame"

**Solution:**
Ensure your CSV/Parquet file loads and has data:
```python
import pandas as pd
df = pd.read_csv("your_file.csv")
print(len(df), df.columns)
# Then pass to setup()
```

---

## Slow Performance

**Problem:** Indexing or search is slow.

**Solutions:**

1. Reduce batch size for memory-constrained environments:
```python
from pathlib import Path

engine = setup(Path("data.csv"), llm_api_key="...")
```

2. Use FAISS (default) instead of ChromaDB for faster in-memory search.

3. For large datasets (>10k rows), consider:
   - Sampling: Index a subset first
   - Pagination: Process in chunks
   - Faster embeddings: sentence-transformers over API-based

---

## Evaluation Gates Failing

**Problem:** `run_regression_gates()` returns `pass=False`.

**Debug:**
```python
summary = run_regression_gates(engine, cases, thresholds)
for result in summary['results']:
    if not result['pass']:
        print(f"Failed: {result['query']}")
        print(f"  Expected {result['expected_min_results']} results, got {result['observed_results']}")
        print(f"  Expected {result['expected_min_citations']} citations, got {result['observed_citations']}")
```

**Solutions:**
1. Lower thresholds in `EvaluationThresholds`
2. Improve data quality or embedding model
3. Adjust test cases to match data coverage

---

## Known Limitations

1. **Keyword embeddings** (demo model) are brittle; use real embeddings in production
2. **Source accuracy** depends on embedding model; always verify manually
3. **ChromaDB persistence** requires explicit save calls; check `.benchmarks/` for artifacts

---

**See also:** [API Reference](./reference-api-cheat-sheet.md) | [Quickstart Guide](./quickstart.md) | [Benchmark Interpretation](./benchmark-interpretation.md)
