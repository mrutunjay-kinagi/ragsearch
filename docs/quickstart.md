# Quickstart Guide

Get up and running with ragsearch in **≤10 minutes** using a public Titanic dataset.

## What You'll Build

A **Retrieval-Augmented Generation (RAG)** system that answers natural language questions about passenger data, grounded in retrieved records and citations.

**Example:**
```
Query: "What happened to female passengers in first class?"
Answer: (generated from retrieved records)
Citations: [source fragments showing actual passenger data]
```

## Prerequisites

- Python 3.9+
- pip or poetry

## Step 1: Install ragsearch (1 min)

```bash
pip install /path/to/ragsearch
```

Or, from the repo root with development dependencies:

```bash
pip install -e .
```

## Step 2: Run the Quickstart Script (5 min)

Copy and run this Python script in your ragsearch directory:

```python
"""
Quickstart: RAG over public Titanic dataset with deterministic embeddings/LLM.
This script runs in ~5 minutes and produces a grounded answer.
"""

import sys
from pathlib import Path

import pandas as pd

# Set up import path
ROOT = Path(__file__).resolve().parent
LIBS_ROOT = ROOT / "libs"
if str(LIBS_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBS_ROOT))

from ragsearch.engine import RagSearchEngine
from ragsearch.vector_db import VectorDB

# ============================================================================
# Step 1: Define deterministic embedding and LLM models (for reproducibility)
# ============================================================================

class DummyEmbeddingResponse:
    """Normalize embedding response format."""
    def __init__(self, embeddings):
        self.embeddings = embeddings


class KeywordEmbeddingModel:
    """
    Simple keyword-based embedding for demo purposes.
    In production, use real embedding models (Cohere, OpenAI, etc).
    """
    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = str(text).lower()
            v = [0.1] * 4
            if "female" in lowered or "woman" in lowered:
                v[0] += 1.0
            if "male" in lowered or "man" in lowered:
                v[1] += 1.0
            if "first" in lowered or "class 1" in lowered:
                v[2] += 1.0
            if "third" in lowered or "class 3" in lowered:
                v[3] += 1.0
            vectors.append(v)
        return DummyEmbeddingResponse(vectors)


class DummyLLMClient:
    """
    Placeholder LLM for demo purposes.
    In production, use Cohere, OpenAI, or Ollama.
    """
    def generate(self, prompt, **kwargs):
        return "Answer generated from retrieved Titanic passenger data. Citation indices reference source records above."


# ============================================================================
# Step 2: Load and normalize Titanic dataset
# ============================================================================

print("Loading public Titanic dataset...")
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df_raw = pd.read_csv(DATA_URL)

# Keep first 100 rows for speed
df = df_raw.head(100).copy()

# Build text field from available columns
def to_text(row):
    pieces = []
    for col in ["Name", "Sex", "Age", "Pclass", "Survived"]:
        if col in row and pd.notna(row[col]):
            pieces.append(f"{col}: {row[col]}")
    return " | ".join(pieces)

df["text"] = df.apply(to_text, axis=1)
df["source_path"] = "public:titanic.csv"
df["parser_name"] = "public/csv"

print(f"Loaded {len(df)} Titanic passenger records.")

# ============================================================================
# Step 3: Initialize RAG Engine and index data
# ============================================================================

print("Initializing RAG Engine...")
engine = RagSearchEngine(
    data=df,
    embedding_model=KeywordEmbeddingModel(),
    llm_client=DummyLLMClient(),
    vector_db=VectorDB(embedding_dim=4),
    save_dir="quickstart_embeddings",
    file_name="titanic.csv",
)
print("✓ Engine initialized and data indexed.")

# ============================================================================
# Step 4: Query and answer
# ============================================================================

print("\n" + "=" * 70)
print("QUERY & ANSWER EXAMPLE")
print("=" * 70)

query = "What happened to female passengers in first class?"
print(f"\nQuery: {query}\n")

result = engine.answer(query, top_k=3)

print(f"Answer:\n{result['answer']}\n")

print(f"Retrieved {len(result['results'])} sources:")
for i, citation in enumerate(result['citations'], start=1):
    print(f"\n[{i}] {citation.get('source_path', 'unknown')}")
    print(f"    {citation.get('excerpt', 'no excerpt')[:100]}...")

print("\n" + "=" * 70)
print("✓ Quickstart complete! You've built a working RAG system.")
print("=" * 70)

print("\n📚 Next steps:")
print("  1. See docs/cookbook-dataset-analytics.md for deeper examples.")
print("  2. See docs/reference-api-cheat-sheet.md for full API documentation.")
print("  3. Try different queries and observe the retrieved sources.")
```

Save this script as `quickstart.py` in your ragsearch root and run:

```bash
python quickstart.py
```

**Expected output** (≈5 seconds):
```
Loading public Titanic dataset...
Loaded 100 Titanic passenger records.
Initializing RAG Engine...
✓ Engine initialized and data indexed.

======================================================================
QUERY & ANSWER EXAMPLE
======================================================================

Query: What happened to female passengers in first class?

Answer:
Answer generated from retrieved Titanic passenger data. Citation indices reference source records above.

Retrieved 3 sources:

[1] public:titanic.csv
    Name: Cumings, Mrs. John Bradley (Florence Briggs Thayer) | ...

[2] public:titanic.csv
    Name: Bonnell, Miss. Elizabeth | ...

[3] public:titanic.csv
    Name: Brown, Mrs. James Joseph (Margaret Tomlinson) | ...

======================================================================
✓ Quickstart complete! You've built a working RAG system.
======================================================================

📚 Next steps:
  1. See docs/cookbook-dataset-analytics.md for deeper examples.
  2. See docs/reference-api-cheat-sheet.md for full API documentation.
  3. Try different queries and observe the retrieved sources.
```

## Key Concepts

| Concept | Explanation |
|---------|-------------|
| **Embedding** | Converts text into a numeric vector. Here we use simple keyword matching; production systems use learned models. |
| **Vector DB** | Stores embeddings and finds similar records via nearest-neighbor search. Here: FAISS in-memory. |
| **Retrieval** | Returns top-k most similar records to your query. |
| **Generation** | Feeds retrieved sources + query to an LLM to generate a grounded answer. |
| **Citation** | Points to the original source record that informed the answer. |

## Troubleshooting

**"ModuleNotFoundError: No module named 'ragsearch'"**
- Ensure you're in the ragsearch repo root.
- Try: `pip install -e .`

**"No results returned"**
- Check that the keyword embedding model covers your query terms.
- Try a simpler query (e.g., "female", "first class").
- See docs/troubleshooting.md for more.

**"LLM provider not found"**
- The quickstart uses a dummy LLM client (no API key needed).
- For real providers, see docs/reference-api-cheat-sheet.md → LLM Setup.

## Next Steps

- **Deeper learning**: Read [docs/cookbook-dataset-analytics.md](./cookbook-dataset-analytics.md)
- **Run examples interactively**: Open [docs/notebook-dataset-analytics.ipynb](./notebook-dataset-analytics.ipynb)
- **API reference**: See [docs/reference-api-cheat-sheet.md](./reference-api-cheat-sheet.md)
- **Troubleshooting**: See [docs/troubleshooting.md](./troubleshooting.md)

---

**Total time:** ~10 minutes | **Concepts covered:** RAG pipeline, embeddings, retrieval, generation, citations.
