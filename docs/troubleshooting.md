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

(citation-does-not-match-answer)=
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

### `NotFoundError: model 'large' not found` with the default Cohere provider

The Cohere adapters do not send a model name, and Cohere has retired the defaults it falls back to (`large` for embeddings, `command-r` for chat). `embedding_model_name` and `llm_model_name` are currently ignored for Cohere, so they cannot be used as a workaround. Until [#82](https://github.com/mrutunjay-kinagi/ragsearch/issues/82) is fixed, use another provider:

```bash
pip install openai
```

```python
from pathlib import Path
from ragsearch import setup

engine = setup(
    Path("data.csv"),
    llm_api_key="sk-...",
    embedding_provider="openai",
    llm_provider="openai",
)
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

1. Reduce input size for memory-constrained environments:
```python
from pathlib import Path

engine = setup(Path("data.csv"), llm_api_key="...")
```
    For very large files, pre-split source data and index in smaller chunks.

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
    if not result['passed']:
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
3. **Benchmark artifacts** in `.benchmarks/` are produced by the benchmark runner scripts, not by search calls
4. **ChromaDB mode:** `search()` and `answer()` currently fail when `use_chromadb=True`; use the default FAISS backend ([#76](https://github.com/mrutunjay-kinagi/ragsearch/issues/76))
5. **Default Cohere models retired:** see the Setup Failures entry above ([#82](https://github.com/mrutunjay-kinagi/ragsearch/issues/82))
6. **One chunk per document by default:** unstructured files are indexed as a single chunk unless you pass a `chunking_strategy` ([#77](https://github.com/mrutunjay-kinagi/ragsearch/issues/77))
7. **Numeric columns** in CSV/JSON/Parquet are not included in the indexed text ([#78](https://github.com/mrutunjay-kinagi/ragsearch/issues/78))
8. **Parsing gaps (built-in fallback parser, used when Node.js/LiteParse is not available):** DOCX tables are skipped and scanned PDFs (no text layer) yield no text, since there is no OCR ([#83](https://github.com/mrutunjay-kinagi/ragsearch/issues/83))
9. **No prompt token budget:** `answer()` sends the full text of every retrieved chunk; very large chunks or a high `top_k` can exceed the model's context window ([#88](https://github.com/mrutunjay-kinagi/ragsearch/issues/88))

---

**See also:** [API Reference](./reference-api-cheat-sheet.md) | [Quickstart Guide](./quickstart.md) | [Benchmark Interpretation](./benchmark-interpretation.md)
