# Troubleshooting Guide

**Status:** Expanded in Slice C with decision flows and detailed solutions.

Common issues and solutions.

## Empty Results

**Problem:** `engine.search(query)` returns empty list.

**Possible causes:**
1. Embedding model doesn't cover query keywords → Try simpler query or adjust embedding model
2. Vector DB not initialized → Check engine logs for initialization errors
3. Data has no textual content → Ensure DataFrame has "text" or "combined_text" column

**Solution:**
```python
# Debug: Check what was indexed
print(f"Indexed rows: {len(engine.index_data)}")
print(engine.index_data[["text", "combined_text"]].head())
```

## Citation Mismatch

**Problem:** Answer looks correct but citation points to wrong source.

**Causes:**
- Query embedding is ambiguous (happens with keyword-based or weak semantic embeddings)
- Multiple similar records exist; top-k includes false positives

**Solution:**
- Use real embedding model (Cohere, OpenAI, etc) instead of dummy
- Increase data quality and distinctiveness
- Lower `top_k` to reduce false positives

## Setup Failures

**Problem:** `setup()` raises an error.

See [API Reference](./reference-api-cheat-sheet.md) → `setup()` for configuration details.

---

**See also:** [Quickstart Guide](./quickstart.md) | [API Reference](./reference-api-cheat-sheet.md)
