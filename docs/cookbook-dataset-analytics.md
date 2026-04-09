# Dataset Analytics Cookbook

**Status:** Expanded in Slice B with full examples and Jupyter notebook.

This guide covers an end-to-end RAG workflow for structured dataset analysis:

1. **Setup** – Initialize engine with your CSV/Parquet data
2. **Indexing** – Build embeddings and vector indexes
3. **Retrieval** – Query and retrieve relevant records
4. **Generation** – Get LLM-generated answers grounded in data
5. **Evaluation** – Validate quality and track metrics

See [Slice B implementation branch](./cookbook-dataset-analytics.md) for detailed walkthrough and interactive Jupyter notebook.

## Quick Example

```python
# (Detailed examples coming in Slice B)
from ragsearch import setup
engine = setup("your_data.csv", llm_api_key="...")
result = engine.answer("Your question about the data?", top_k=5)
print(result["answer"])
print(result["citations"])
```

---

**See also:** [Quickstart Guide](./quickstart.md) | [API Reference](./reference-api-cheat-sheet.md)
