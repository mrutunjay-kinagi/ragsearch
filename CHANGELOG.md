# Changelog

All notable changes to ragsearch are documented here. Each pull request adds its own entry under
**Unreleased**; the section is renamed to the version number when a release is cut.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

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

- `ModelNotFoundError` (a `RagSearchError`), raised when a provider reports an unknown or retired
  model. The message names the parameter to change (`embedding_model_name` or `llm_model_name`).
  ([#90](https://github.com/mrutunjay-kinagi/ragsearch/pull/90))

### Changed

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
