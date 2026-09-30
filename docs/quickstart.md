# Quickstart Guide

Build a working RAG system over a small set of insurance claims in a few minutes, using the same `setup()` / `answer()` calls you would use in production.

## What You'll Build

A **Retrieval-Augmented Generation (RAG)** system that answers questions about insurance claims, grounded in the retrieved claims and citing them:

```
Question: Why was the windshield claim denied?
Answer:   The windshield claim was denied because the policy has no glass coverage [1].
Sources:  [1] record 2: CLM-1003 | auto | denied | Windshield cracked by road debris on the highway; ...
```

## Prerequisites

- Python 3.10+
- A Cohere API key ([create one here](https://dashboard.cohere.com/api-keys)), exported as `COHERE_API_KEY`:

```bash
export COHERE_API_KEY="your-cohere-api-key"
```

## Step 1: Install ragsearch

```bash
pip install ragsearch
```

## Step 2: Run the quickstart script

Save this as `quickstart.py` in an empty folder. It writes the sample data to `insurance_claims.csv` (the same file ships in the repository as [`samples/quickstart/insurance_claims.csv`](https://github.com/mrutunjay-kinagi/ragsearch/blob/master/samples/quickstart/insurance_claims.csv)), indexes it, and answers a question.

```python
"""Quickstart: RAG over a small set of insurance claims with ragsearch."""

import os
from pathlib import Path

from ragsearch import setup

SAMPLE_CSV = """\
claim_id,policy_type,status,description
CLM-1001,auto,approved,"Rear-end collision at a red light on Main Street; bumper and trunk damage, no injuries reported."
CLM-1002,home,under review,"Burst pipe in the upstairs bathroom flooded the kitchen ceiling; plumber invoice and photos attached."
CLM-1003,auto,denied,"Windshield cracked by road debris on the highway; policy has no glass coverage."
CLM-1004,health,approved,"Emergency room visit for a sprained ankle after a fall at work; X-ray showed no fracture."
CLM-1005,home,approved,"Hailstorm damaged roof shingles and two skylights; contractor estimate submitted."
CLM-1006,auto,under review,"Side-swiped in a parking garage by an unidentified vehicle; driver-side doors dented."
CLM-1007,travel,approved,"Checked luggage lost on a connecting flight; airline confirmed the bag was not recovered."
CLM-1008,health,denied,"Cosmetic dental whitening procedure; excluded from the dental plan."
"""

data_path = Path("insurance_claims.csv")
data_path.write_text(SAMPLE_CSV, encoding="utf-8")

# Index the claims: parse, embed with Cohere, and store the vectors in FAISS.
engine = setup(data_path, os.environ["COHERE_API_KEY"])
print(f"Indexed {engine.ingestion_diagnostics['indexing']['total_records']} claims.")

# Retrieve the most relevant claims and generate a grounded, cited answer.
question = "Why was the windshield claim denied?"
result = engine.answer(question, top_k=3)

print(f"\nQuestion: {question}")
print(f"Answer: {result['answer']}\n")
print("Sources:")
for number, citation in enumerate(result["citations"], start=1):
    print(f"[{number}] record {citation['record_id']}: {citation['excerpt'][:90]}")
```

Run it:

```bash
python quickstart.py
```

**Example output** (from a real run; the answer text and lower-ranked sources can vary):
```
Indexed 8 claims.

Question: Why was the windshield claim denied?
Answer: The windshield claim was denied because the policy has no glass coverage [1].

Sources:
[1] record 2: CLM-1003 | auto | denied | Windshield cracked by road debris on the highway; policy has no
[2] record 7: CLM-1008 | health | denied | Cosmetic dental whitening procedure; excluded from the dental
[3] record 4: CLM-1005 | home | approved | Hailstorm damaged roof shingles and two skylights; contractor
```

The first run also creates an `embeddings/` folder: a cache that lets later runs skip re-embedding unchanged claims.

## Key Concepts

| Concept | Explanation |
|---------|-------------|
| **Embedding** | Converts text into a numeric vector. `setup()` uses Cohere's `embed-v4.0` by default. |
| **Vector DB** | Stores embeddings and finds similar records with nearest-neighbor search (FAISS, in memory). |
| **Retrieval** | `search()` returns the top-k records most similar to your question. |
| **Generation** | `answer()` sends the retrieved sources and your question to an LLM (Cohere's `command-a-03-2025` by default). |
| **Citation** | Every result points back to the source record it came from. |

## Troubleshooting

**`KeyError: 'COHERE_API_KEY'`**
- Export your key first: `export COHERE_API_KEY="your-cohere-api-key"`.

**`TypeError: 'module' object is not callable`**
- You have ragsearch 0.1.x. Upgrade: `pip install --upgrade ragsearch` (this quickstart needs 0.2.0 or newer).

**`ModelNotFoundError`**
- The configured model name is unknown to your provider or has been retired. The message says which parameter to change; see [troubleshooting](./troubleshooting.md).

**Using another provider (OpenAI, Ollama, or an OpenAI-compatible server)**
- See the README sections [Changing the LLM provider](https://github.com/mrutunjay-kinagi/ragsearch#changing-the-llm-provider) and [OpenAI-compatible endpoints](https://github.com/mrutunjay-kinagi/ragsearch#openai-compatible-endpoints).

## Next Steps

- **Your own data**: pass any CSV, JSON, Parquet, PDF, DOCX, HTML, Markdown or text file to `setup()`.
- **Deeper learning**: read [cookbook-dataset-analytics.md](./cookbook-dataset-analytics.md).
- **Run examples interactively**: open [notebook-dataset-analytics.ipynb](./notebook-dataset-analytics.ipynb).
- **API reference**: see [reference-api-cheat-sheet.md](./reference-api-cheat-sheet.md).
- **Troubleshooting**: see [troubleshooting.md](./troubleshooting.md).
