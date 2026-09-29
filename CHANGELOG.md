# Changelog

All notable changes to ragsearch are documented here. Each pull request adds its own entry under
**Unreleased**; the section is renamed to the version number when a release is cut.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- **PDF, DOCX and HTML files work after a plain `pip install ragsearch`.** The parsers
  `pypdf` (>= 6.16.2), `python-docx` (>= 0.8.11) and `beautifulsoup4` (>= 4.9.0) are now regular
  dependencies. They used to be optional, and without them every PDF and DOCX failed. If one is
  missing at runtime anyway, the error names the `pip install` command. ([#103](https://github.com/mrutunjay-kinagi/ragsearch/pull/103), [#100](https://github.com/mrutunjay-kinagi/ragsearch/issues/100))
- **No more failed LiteParse call before every document.** ragsearch invoked
  `npx --no-install @run-llama/liteparse`, a package that does not exist on npm, whenever
  Node.js was installed. That cost about 1.4 s per file before falling back, and broke `.doc` and
  image files with an npm 404. ([#103](https://github.com/mrutunjay-kinagi/ragsearch/pull/103), [#100](https://github.com/mrutunjay-kinagi/ragsearch/issues/100))

- **API keys are no longer sent to the wrong provider.** `embedding_api_key` defaulted to
  `llm_api_key` even when the two providers differed, so, for example, an OpenAI key was sent to
  Cohere. Separately, Cohere embeddings always used `llm_api_key` and ignored `embedding_api_key`, so
  an OpenAI LLM with Cohere embeddings could not work. Each provider now gets its own key, and
  mixed providers work. ([#101](https://github.com/mrutunjay-kinagi/ragsearch/pull/101))
- **Importing ragsearch no longer configures the root logger.** `ragsearch.vector_db` called
  `logging.basicConfig(level=INFO)` at import time, overriding the host application's logging
  setup. It now logs through a module-level logger. ([#101](https://github.com/mrutunjay-kinagi/ragsearch/pull/101))

- **Default Cohere setup works again.** The Cohere adapters now send an explicit model:
  `embed-v4.0` for embeddings and `command-a-03-2025` for chat. Before, Cohere fell back to its
  retired `large` embedding model and every default `setup()` failed. `embedding_model_name` and
  `llm_model_name` are now passed through to Cohere instead of being ignored. Embed requests send
  `input_type="search_document"`, which Cohere's v3+ embedding models require.
  ([#82](https://github.com/mrutunjay-kinagi/ragsearch/issues/82),
  [#90](https://github.com/mrutunjay-kinagi/ragsearch/pull/90))
- **`from ragsearch import setup` returns the `setup()` function.** It returned the
  `ragsearch.setup` module, so every README example failed with
  `TypeError: 'module' object is not callable`. `from ragsearch import RagSearchEngine` is also
  reliable regardless of import order.
  ([#91](https://github.com/mrutunjay-kinagi/ragsearch/issues/91),
  [#93](https://github.com/mrutunjay-kinagi/ragsearch/pull/93))
- **Embedding cache no longer reuses vectors from a different model.** The cache manifest (now
  schema version 2) records the embedding model and vector dimension. When either changes, or the
  cache predates this release, `setup()` logs a warning and re-embeds every record. Before, it
  failed with an empty `Embedding indexing failed` error, or with a same-dimension model it
  silently mixed vectors from two models. The reason is reported in
  `ingestion_diagnostics["indexing"]["cache_invalidated_reason"]`.
  ([#95](https://github.com/mrutunjay-kinagi/ragsearch/pull/95))

### Added

- **Answer context budget.** `answer()` adds retrieved sources in score order up to
  `max_context_tokens` estimated tokens (default 3000; set it on `setup()` or per `answer()` call,
  `None` disables it). The response gains `context_sources`, `context_truncated` and
  `context_tokens_estimate`. `FixedWordChunkingStrategy` gains `overlap_words`. ([#118](https://github.com/mrutunjay-kinagi/ragsearch/pull/118), [#77](https://github.com/mrutunjay-kinagi/ragsearch/issues/77), [#88](https://github.com/mrutunjay-kinagi/ragsearch/issues/88))

- `EmbeddingProbeError` (a `RagSearchError` and `RuntimeError`), raised when `setup()` cannot
  determine the embedding dimension. ([#101](https://github.com/mrutunjay-kinagi/ragsearch/pull/101))

- **A quickstart that uses the real API.** `docs/quickstart.md` now runs the real
  `setup()` / `answer()` path on a small insurance-claims sample. It needs only
  `pip install ragsearch` and a Cohere key, and no longer depends on demo embedding and LLM
  stand-ins or a repository checkout. The samples ship in `samples/quickstart/` (claims CSV, a
  claim letter, evaluation cases and an engine factory for the evaluation CLI). The demo
  stand-ins moved to test fixtures, and a test runs the quickstart script offline.
  ([#99](https://github.com/mrutunjay-kinagi/ragsearch/pull/99))
- README section on **OpenAI-compatible endpoints** (LM Studio, vLLM, gateways) via
  `llm_provider="openai"` + `llm_base_url` and `embedding_provider="openai"` +
  `embedding_base_url`. ([#99](https://github.com/mrutunjay-kinagi/ragsearch/pull/99))

- `ModelNotFoundError` (a `RagSearchError`), raised when a provider reports an unknown or retired
  model. The message names the parameter to change (`embedding_model_name` or `llm_model_name`).
  ([#90](https://github.com/mrutunjay-kinagi/ragsearch/pull/90))

### Changed

- **Unstructured files are chunked by default.** PDF, DOCX, HTML, Markdown and text files are
  split into 150-word windows with 30 words of overlap, instead of one chunk per file. That used to
  send whole documents to the LLM: about 57,700 estimated tokens per question for an 80-page PDF.
  Structured files keep one chunk per row. The chunking settings are part of the embedding-cache
  identity (manifest schema v3), so the first `setup()` after upgrading re-embeds cached
  unstructured files once, with a warning. Pass `chunking_strategy=RowChunkingStrategy()` to keep
  the old behaviour. ([#118](https://github.com/mrutunjay-kinagi/ragsearch/pull/118), [#77](https://github.com/mrutunjay-kinagi/ragsearch/issues/77), [#88](https://github.com/mrutunjay-kinagi/ragsearch/issues/88))
- **`answer()["citations"]` lists only the sources sent to the LLM.** When the context budget
  leaves sources out, they stay in `results` but not in `citations`, so citation numbers always
  match the `[n]` markers in the answer. ([#118](https://github.com/mrutunjay-kinagi/ragsearch/pull/118), [#77](https://github.com/mrutunjay-kinagi/ragsearch/issues/77), [#88](https://github.com/mrutunjay-kinagi/ragsearch/issues/88))

- **LiteParse is no longer used automatically.** The official LiteParse CLI
  (`@llamaindex/liteparse`) has a different interface from the one ragsearch expects, so support is
  deferred to [#102](https://github.com/mrutunjay-kinagi/ragsearch/issues/102). ragsearch uses a
  LiteParse-compatible CLI only when `RAGSEARCH_LITEPARSE_CLI` is set. `.doc`, `.png` and `.jpg`
  files are therefore unsupported by default. The README no longer tells users to run
  `npx --yes @run-llama/liteparse`, which would have downloaded and executed whatever gets
  published under that unclaimed name. ([#103](https://github.com/mrutunjay-kinagi/ragsearch/pull/103), [#100](https://github.com/mrutunjay-kinagi/ragsearch/issues/100))

- **`setup()` fails fast when the embedding-dimension probe fails.** It used to log a warning
  and fall back to dimension 4096, which only suited Cohere's retired `large` model and otherwise
  failed later with `AssertionError: d == self.d`. It now raises `EmbeddingProbeError` naming the
  provider and the cause. ([#101](https://github.com/mrutunjay-kinagi/ragsearch/pull/101))
- **`embedding_api_key` is required when the embedding and LLM providers differ** (except for the
  keyless `ollama` and `sentence_transformers` embedding providers). `setup()` raises a
  `ValueError` instead of sending `llm_api_key` to a different provider. Same-provider setups
  (such as the Cohere default) keep defaulting to `llm_api_key`. ([#101](https://github.com/mrutunjay-kinagi/ragsearch/pull/101))
- **ragsearch no longer prints INFO logs by default.** Applications that want them should
  configure logging themselves, for example `logging.basicConfig(level=logging.INFO)`. ([#101](https://github.com/mrutunjay-kinagi/ragsearch/pull/101))

- **README rewritten so every example runs as written** (given a key). It adds a "what makes
  ragsearch different" summary, a known-issues link, a short sample-data excerpt instead of the raw
  recipe CSV, a clear "currently broken" label on the ChromaDB example (#76), and no internal
  jargon. `README.rst` is removed; `README.md` is the only README and the package long
  description. The Sphinx landing page no longer embeds the README.
  ([#99](https://github.com/mrutunjay-kinagi/ragsearch/pull/99))

- **Upgrading re-embeds existing caches once.** Caches written by 0.1.5 or earlier carry no model
  metadata, so the first `setup()` after upgrading re-embeds them (one embedding API pass per
  data file). ([#95](https://github.com/mrutunjay-kinagi/ragsearch/pull/95))
- **`ragsearch.setup` as an attribute is now the `setup()` function, not the module.** As a
  result, `import ragsearch.setup as m` binds the function, and pytest's string-form
  `monkeypatch.setattr("ragsearch.setup.X", ...)` fails with `AttributeError`. To reach the
  module, use `unittest.mock.patch("ragsearch.setup.X")` or
  `importlib.import_module("ragsearch.setup")`. `from ragsearch.setup import setup` is unchanged.
  ([#93](https://github.com/mrutunjay-kinagi/ragsearch/pull/93))
- A wrong embedding model name is no longer masked by `setup()`'s fallback to a 4096-dimension
  index. It raises `ModelNotFoundError` instead.
  ([#90](https://github.com/mrutunjay-kinagi/ragsearch/pull/90))

### Deprecated

- Reusing `llm_api_key` for the embedding provider when `llm_provider` needs no key (`ollama`)
  and `embedding_api_key` is not given. It still works, but emits a `DeprecationWarning`. Pass
  `embedding_api_key` instead; the fallback will be removed in a future release. ([#101](https://github.com/mrutunjay-kinagi/ragsearch/pull/101))
