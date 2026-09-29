# ragsearch

`ragsearch` is a Python library for building Retrieval-Augmented Generation (RAG) applications that answer natural language questions over structured and unstructured data. It embeds your records, indexes them in a vector database (FAISS by default), retrieves the most relevant ones for a question, and asks an LLM for an answer grounded in those sources.

> **Known issues:** a few features are still being fixed. See [Known limitations](./docs/troubleshooting.md#known-limitations) before you start.

## What makes ragsearch different

One library covers both structured data (CSV, JSON, Parquet) and unstructured documents (PDF, DOCX, HTML, text), behind the same `setup()` / `search()` / `answer()` calls. Every search result and every answer carries **citations** back to the source record, file and parser, so you can check what the model saw. It also ships an **evaluation harness** (regression gates you can run in CI) and **diagnostics** for ingestion, indexing, retrieval and generation, so you can see what happened inside a run.

---

## Start here

👉 **[Quickstart Guide](./docs/quickstart.md)**: build a working RAG system over a small insurance-claims file in a few minutes. You need `pip install ragsearch` and a [Cohere API key](https://dashboard.cohere.com/api-keys).

Then explore:
- 📖 [Dataset Analytics Cookbook](./docs/cookbook-dataset-analytics.md): end-to-end workflow examples with a Jupyter notebook.
- 🔍 [API Reference & Cheat Sheet](./docs/reference-api-cheat-sheet.md): full parameter reference and usage patterns.
- ❓ [Troubleshooting Guide](./docs/troubleshooting.md): common issues and how to solve them.
- 📊 [Benchmark Interpretation](./docs/benchmark-interpretation.md): understanding evaluation results and metrics.

---

## Features
- Structured (CSV/JSON/Parquet) and unstructured (PDF/DOCX/HTML/Markdown/text) ingestion through one `setup()` call.
- Citations on every search result and answer.
- Pluggable providers: Cohere (default), OpenAI and [OpenAI-compatible servers](#openai-compatible-endpoints), Ollama, and local sentence-transformers embeddings.
- FAISS for fast in-memory vector search (ChromaDB support is being reworked, see [#76](https://github.com/mrutunjay-kinagi/ragsearch/issues/76)).
- Built-in PDF, DOCX and HTML parsing (pypdf, python-docx, beautifulsoup4), included in the default install.
- Incremental indexing: unchanged records reuse cached embeddings across runs.
- Evaluation harness, diagnostics and observability events built in.
- A simple web interface and HTTP API.

## Installation

```bash
pip install ragsearch
```

> The fixes described in this README are newer than the latest PyPI release (0.1.5). Until the next release, install from GitHub:
> `pip install "git+https://github.com/mrutunjay-kinagi/ragsearch@develop"`

All dependencies are installed automatically, including the document parsers for PDF (pypdf), DOCX (python-docx) and HTML (beautifulsoup4). Node.js is not needed.

> **LiteParse is not supported yet.** LlamaIndex's LiteParse (npm `@llamaindex/liteparse`) has a different command-line interface from the one ragsearch was written for, so ragsearch does not use it automatically. Proper support (including OCR for scanned PDFs) is planned in [#102](https://github.com/mrutunjay-kinagi/ragsearch/issues/102).

## Basic usage

The examples below run from a clone of this repository, using the small sample files in [`samples/quickstart/`](./samples/quickstart/), and read your Cohere key from the `COHERE_API_KEY` environment variable. Run them in order, as one script or one notebook.

### Step 1: Prepare your data
Structured files (CSV/JSON/Parquet) are loaded with pandas. Unstructured files are parsed with the built-in parsers.

**Example data** ([`samples/quickstart/insurance_claims.csv`](./samples/quickstart/insurance_claims.csv), first rows):
```csv
claim_id,policy_type,status,description
CLM-1001,auto,approved,"Rear-end collision at a red light on Main Street; bumper and trunk damage, no injuries reported."
CLM-1002,home,under review,"Burst pipe in the upstairs bathroom flooded the kitchen ceiling; plumber invoice and photos attached."
CLM-1003,auto,denied,"Windshield cracked by road debris on the highway; policy has no glass coverage."
```

### Step 2: Initialize `ragsearch`

```python
import os
from pathlib import Path

from ragsearch import setup

data_path = Path("samples/quickstart/insurance_claims.csv")
llm_api_key = os.environ["COHERE_API_KEY"]

rag_engine = setup(data_path, llm_api_key)
```

### Step 3: Run a search query

```python
results = rag_engine.search("Which claims involve water damage?", top_k=3)

for result in results:
    print("Metadata:", result["metadata"])
    print("Citation:", result["citation"])
    print("Similarity:", result["similarity"])

# citation fields:
# - record_id: row/chunk index in the indexed dataset
# - source_path: source file path when available
# - parser_name: parser used during ingestion when available
# - excerpt: up to 200 chars of the matched chunk (for display; answer() sends the full chunk to the LLM)
```

### Step 4: Generate a grounded answer

```python
response = rag_engine.answer("Why was the windshield claim denied?", top_k=3)

print(response["answer"])
for citation in response["citations"]:
    print(citation)
```

Answer response fields:
- `question`: original query string
- `answer`: generated response text
- `results`: full retrieval results, including `metadata`, `citation`, and `similarity`
- `citations`: citations of the sources the LLM actually received, in the same order as the `[n]` markers
- `context`: the numbered sources supplied to the LLM, each containing the full text of the retrieved chunk (not the 200-char `excerpt`)
- `context_sources`: how many of the retrieved `results` were sent to the LLM
- `context_truncated`: `True` when sources were left out, or cut, to stay within the context budget
- `context_tokens_estimate`: estimated tokens of `context` (about 3.5 characters per token, erring high)

**Context budget.** `answer()` adds sources in score order until the next one would exceed `max_context_tokens` estimated tokens (default 3000). If even the first source is too large, a truncated prefix of it is sent. Set it per engine with `setup(..., max_context_tokens=...)`, or per call with `answer(..., max_context_tokens=...)`. `None` on the engine disables the cap. The default suits Ollama's default 4096-token context window with room for the instructions and the answer. Raise it for large-context models such as Cohere's Command A (256K).

### Unstructured files (PDF/DOCX/HTML/Markdown/text)

Use the same `setup()` call:

```python
letter_engine = setup(Path("samples/quickstart/claim_letter.txt"), llm_api_key)

print(letter_engine.answer("What caused the water damage?", top_k=1)["answer"])
```

Parser behavior:
- The built-in parser handles `.txt`, `.md`, `.html`, `.htm`, `.pdf` and `.docx`. Other types raise `UnsupportedFileTypeError`.
- Advanced: setting the `RAGSEARCH_LITEPARSE_CLI` environment variable to a CLI that accepts `<cli> --json <file>` and prints `{"documents": [...]}` or `{"text": ...}` makes ragsearch try it first, and fall back to the built-in parser if it fails. The official LiteParse CLI does not use this interface yet ([#102](https://github.com/mrutunjay-kinagi/ragsearch/issues/102)).

## Diagnostics and incremental indexing

Ingestion diagnostics: after `setup()`, `rag_engine.ingestion_diagnostics` reports per-file fields:
- `source_path`: input file path used for setup.
- `selected_parser`: `structured/pandas`, `liteparse`, or `fallback`.
- `status`: `success` or `recovered_with_fallback`.
- `failure_reason`: empty string on success; the primary parser's error message when fallback recovery was used.
- `observability`: setup metrics (`stage`, `event`, and `metrics.setup_latency_ms`, `metrics.loaded_records`, `metrics.selected_parser`, `metrics.fallback_recovered`).

Incremental indexing (FAISS backend):
- `setup()` persists an embedding manifest in the embeddings directory (`embeddings_dir`, default `embeddings/`) and reuses cached embeddings for unchanged records.
- New or changed records are re-embedded, while unchanged records are skipped for embedding generation.
- The manifest records which embedding model and vector dimension produced the cache. If either changes, or the cache was written by ragsearch 0.1.5 or earlier (which stored neither), `setup()` logs a warning and re-embeds every record once instead of reusing incompatible vectors.
- `rag_engine.ingestion_diagnostics["indexing"]` reports:
    - `manifest_version`: manifest schema version.
    - `manifest_path`: on-disk manifest file path.
    - `total_records`: total records considered for indexing.
    - `embedded_records`: records embedded in the current run.
    - `reused_records`: records reused from manifest cache.
    - `new_records`: records seen for the first time.
    - `changed_records`: previously-seen records with changed content hash.
    - `cache_invalidated_reason`: why the cache was discarded and everything re-embedded (empty when the cache was reusable).

Logging: ragsearch does not configure logging. To see its progress messages, configure logging in your application, for example `logging.basicConfig(level=logging.INFO)`.

Observability events:
- `rag_engine.observability_events` stores structured events emitted during indexing, retrieval and generation.
- Retrieval events include `query`, `top_k`, `results_count`, `latency_ms`.
- Generation events include `query`, `top_k`, `results_count`, `citations_count`, `latency_ms`.
- Configure `observability_max_events` in `setup(...)` to cap retained in-memory events for long-lived processes.

## Retrieval quality hooks

- `chunking_strategy`: controls how each record is split before embedding and indexing.
- `reranker`: post-processes retrieval results before they are returned.
- Defaults: unstructured files (PDF, DOCX, HTML, Markdown, text) are split into overlapping windows of 150 words with 30 words of overlap (`FixedWordChunkingStrategy(words_per_chunk=150, overlap_words=30)`); structured files (CSV, JSON, Parquet) keep one chunk per row (`RowChunkingStrategy`); no reranking (`NoOpReranker`).
- Changing the chunking strategy or its settings invalidates the embedding cache: the next `setup()` logs a warning and re-embeds every record once.

```python
from ragsearch.chunking import FixedWordChunkingStrategy


class ReverseReranker:
    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        return list(reversed(results))


tuned_engine = setup(
    data_path,
    llm_api_key,
    chunking_strategy=FixedWordChunkingStrategy(words_per_chunk=120, overlap_words=20),
    reranker=ReverseReranker(),
)
```

Keep hooks at their defaults unless you are tuning retrieval quality: chunking changes alter indexed record boundaries and retrieval behavior.

## Evaluation

Use `ragsearch.evaluation.run_regression_gates(...)` to run pass/fail gates over a fixed set of questions. Thresholds are configurable with `EvaluationThresholds(min_results=..., min_citations=...)`.

```python
from ragsearch.evaluation import EvaluationThresholds, load_cases, run_regression_gates

cases = load_cases(Path("samples/quickstart/eval_cases.json"))
summary = run_regression_gates(
    rag_engine,
    cases,
    EvaluationThresholds(min_results=1, min_citations=1),
)

print(summary["pass"], summary["passed_cases"], summary["failed_cases"])
```

Evaluation CLI, with the engine factory in [`samples/quickstart/build_engine.py`](./samples/quickstart/build_engine.py):

```bash
PYTHONPATH=samples/quickstart python -m ragsearch.evaluation \
    --engine-factory build_engine.build_engine \
    --cases samples/quickstart/eval_cases.json \
    --summary-only
```

## Web interface and HTTP API

`run()` starts a Flask server on port 8080 in a background thread and returns immediately. In a script, keep the main thread alive, or requests fail once the script ends:

```python
import threading

rag_engine.run()
threading.Event().wait()  # serve until you press Ctrl+C
```

Then open http://localhost:8080/, enter a query, and click **Submit**.

HTTP API:
- `POST /query` returns search results. It returns metadata only by default; set `include_details=true` in the request JSON to also get `citation` and `similarity`.
- `POST /answer` returns the same payload as `rag_engine.answer(...)`.

## Advanced usage

### Changing the embedding model

Embedding providers, selected with `setup()` parameters:
- `embedding_provider="cohere"` (default)
- `embedding_provider="sentence_transformers"` (local; `pip install sentence-transformers`)
- `embedding_provider="openai"` (`pip install openai`)
- `embedding_provider="ollama"` (`pip install ollama`)

Optional provider settings:
- `embedding_model_name`: provider-specific model id (Cohere default: `embed-v4.0`)
- `embedding_api_key`: embedding provider key. It defaults to `llm_api_key` only when the embedding and LLM providers are the same, and is required when they differ (except for the keyless `ollama` and `sentence_transformers`), so one provider's key is never sent to another.
- `embedding_base_url`: custom endpoint URL (OpenAI-compatible server or Ollama host)

Example (requires `pip install openai` and an OpenAI key in `OPENAI_API_KEY`):
```python
openai_embeddings_engine = setup(
    data_path,
    llm_api_key,
    embedding_provider="openai",
    embedding_model_name="text-embedding-3-small",
    embedding_api_key=os.environ["OPENAI_API_KEY"],
)
```

Custom embedding models must provide `embed(texts=[...])` and return an object with an `embeddings` attribute holding a non-empty sequence of numeric vectors.

`setup()` probes the embedding model to find the vector dimension. If the probe fails (an invalid response shape, a provider or network error), `setup()` raises `EmbeddingProbeError` (a `RuntimeError`) explaining what failed, instead of guessing a dimension. An unknown or retired model name raises `ModelNotFoundError` naming the parameter to change.

### Changing the LLM provider

LLM providers:
- `llm_provider="cohere"` (default)
- `llm_provider="openai"` (`pip install openai`), also used for [OpenAI-compatible servers](#openai-compatible-endpoints)
- `llm_provider="ollama"` (`pip install ollama`)

Optional provider settings:
- `llm_model_name`: provider-specific chat model id (Cohere default: `command-a-03-2025`)
- `llm_base_url`: custom endpoint URL (OpenAI-compatible server or Ollama host)

Example with OpenAI for both chat and embeddings (requires `pip install openai` and an OpenAI key in `OPENAI_API_KEY`):
```python
openai_engine = setup(
    data_path,
    os.environ["OPENAI_API_KEY"],
    llm_provider="openai",
    llm_model_name="gpt-4o-mini",
    embedding_provider="openai",
)
```

**Mixing providers.** Give each provider its own key. For example, an OpenAI LLM with Cohere embeddings (the default embedding provider):
```python
mixed_engine = setup(
    data_path,
    os.environ["OPENAI_API_KEY"],
    llm_provider="openai",
    llm_model_name="gpt-4o-mini",
    embedding_api_key=os.environ["COHERE_API_KEY"],
)
```

When the providers differ and `embedding_api_key` is missing, `setup()` raises a `ValueError` instead of sending your OpenAI key to Cohere. The one exception is an LLM provider that needs no key (`ollama`): `llm_api_key` is then still used for the embedding provider, with a `DeprecationWarning`. Pass `embedding_api_key` instead, because this fallback will be removed.

Custom LLM clients must implement `generate(prompt, **kwargs)` and return a string.

### OpenAI-compatible endpoints

Any server that speaks the OpenAI API works through `llm_provider="openai"` plus `llm_base_url`, and, for embeddings, `embedding_provider="openai"` plus `embedding_base_url`. The OpenAI SDK sends requests to `<base_url>/chat/completions` and `<base_url>/embeddings`. Examples:
- [LM Studio](https://lmstudio.ai/docs/app/api/endpoints/openai): serves both endpoints at `http://localhost:1234/v1` by default.
- [vLLM](https://docs.vllm.ai/en/latest/serving/online_serving/): `vllm serve <model>` exposes `/v1/chat/completions` for chat models and `/v1/embeddings` for embedding models.
- Hosted gateways and proxies that expose the OpenAI API.

ragsearch requires a non-empty API key. Local servers usually ignore it, so pass any placeholder.

Example with LM Studio serving a chat model and an embedding model (requires `pip install openai`). Replace the two model identifiers with the ones your server lists under `GET /v1/models`:
```python
local_engine = setup(
    data_path,
    "lm-studio",  # placeholder key, as in LM Studio's own examples
    llm_provider="openai",
    llm_model_name="your-chat-model-identifier",
    llm_base_url="http://localhost:1234/v1",
    embedding_provider="openai",
    embedding_model_name="your-embedding-model-identifier",
    embedding_base_url="http://localhost:1234/v1",
)
```

### Using the ChromaDB backend

> **Currently broken ([#76](https://github.com/mrutunjay-kinagi/ragsearch/issues/76)):** in ChromaDB mode, `search()` and `answer()` fail with `'NoneType' object has no attribute 'search'`, and `data_path` is not indexed. Use the default FAISS backend until this is fixed. The code below shows the intended API and does not work yet.

`use_chromadb=True` takes a ChromaDB persistence directory (`chromadb_sqlite_path`, passed to `chromadb.PersistentClient(path=...)`) and a collection name:

```python
chroma_engine = setup(
    data_path,
    llm_api_key,
    use_chromadb=True,
    chromadb_sqlite_path="chroma_data",
    chromadb_collection_name="insurance_claims",
)
```

As a stopgap, `chroma_engine.chromadb_search(query, top_k)` queries an existing collection directly. It uses the collection's own embedding function (Chroma's default is all-MiniLM-L6-v2, downloaded on first use) and returns Chroma's raw result dict, not ragsearch results with citations.

### Customizing the web interface
Edit `index.html` in `libs/ragsearch/templates` to adjust the UI layout.

## Troubleshooting
- **`ModelNotFoundError: Cohere embedding model '...' is not available`** (or `chat model`): the provider does not recognise the model name, or has retired it. Set `embedding_model_name` or `llm_model_name` (whichever the message names) to a current model from [Cohere's model list](https://docs.cohere.com/docs/models).
- **`AssertionError: d == self.d`**: vector dimensions are normally inferred automatically. If this appears with a custom provider, check that your embed response contains consistent numeric vectors in `response.embeddings`.
- **`ValueError: Embedding response must contain an 'embeddings' attribute`**: your custom embedding model does not follow the embedding contract; return an object with an `embeddings` sequence.
- **DOCX content missing / scanned PDF gives `NoDataFoundError`**: with the built-in parser, DOCX files are read paragraphs-only, so tables are skipped, and PDFs without a text layer yield no text because there is no OCR ([#83](https://github.com/mrutunjay-kinagi/ragsearch/issues/83)). Convert table-heavy DOCX files to PDF or text, and OCR scans before ingesting.

### Parser pipeline troubleshooting

| Error | Typical Cause | Resolution |
| --- | --- | --- |
| `ParserUnavailableError: Cannot parse PDF files: the 'pypdf' package is not installed` (or `python-docx`, `beautifulsoup4`) | Incomplete installation: these are ragsearch dependencies | Run the `pip install ...` command from the message, or reinstall ragsearch |
| `ParserUnavailableError: LiteParse CLI not found` | `RAGSEARCH_LITEPARSE_CLI` is not set, or points to a path that does not exist | Unset it to use the built-in parser, or fix the path (see [#102](https://github.com/mrutunjay-kinagi/ragsearch/issues/102)) |
| `ParseTimeoutError` | Large/complex document exceeded parse timeout | In the default `setup()` flow, a timeout may be recovered automatically by the fallback parser for supported types; otherwise retry with a smaller file and inspect parser logs |
| `ParseCorruptError` | Corrupt file or invalid parser output payload | In the default `setup()` flow, corruption may be recovered automatically by the fallback parser for supported types; if both parsers fail, the primary LiteParse error is raised |
| `UnsupportedFileTypeError` | Extension not supported by the active parser backend | Convert to a supported format: `.txt/.md/.html/.htm/.pdf/.docx` |
| `NoDataFoundError` | File parsed but content was empty/whitespace only | Verify the source file contains readable text |

More in the [Troubleshooting Guide](./docs/troubleshooting.md).

### Performance notes

- Structured files (CSV/JSON/Parquet) are typically fastest because they bypass parser dispatch.
- Unstructured parsing performance depends on document size and parser backend.
- Very large PDF/DOCX files can hit parser timeouts; split them where possible.
- Empty or whitespace-only parsed documents are dropped before indexing.

## Development

```bash
poetry install
poetry run pytest
```

Changes are recorded in [CHANGELOG.md](./CHANGELOG.md).

## Contributing
Contributions are welcome: open an issue, a feature request, or a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
