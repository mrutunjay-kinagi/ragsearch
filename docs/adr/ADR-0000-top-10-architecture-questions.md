# ADR-0000 — Top-10 Prioritized Architecture & Technical Questions for ragsearch v1

**Status:** Informational / Decision Backlog  
**Date:** 2026-03-29  
**Author:** Architect Agent  
**Relates to:** [Issue #21](https://github.com/mrutunjay-kinagi/ragsearch/issues/21), [Epic #19](https://github.com/mrutunjay-kinagi/ragsearch/issues/19)

---

## Context

This document captures the 10 most critical architecture and technical questions that must be answered before ragsearch v1 can be considered enterprise-grade. Each question is anchored to the agreed constraints:

- **local-first** — no mandatory cloud API; all components must have a viable local alternative
- **LiteParse required** — `@run-llama/liteparse` is the default text-extraction layer for unstructured documents
- **Chroma default / FAISS opt-in** — ChromaDB (persistent local SQLite) is the default vector backend
- **Incremental indexing via full-file SHA-256** — source files are fingerprinted by their full SHA-256 digest; unchanged files are not re-embedded
- **Multi-root** — the engine can index multiple directories, files, and glob patterns in one instance
- **Chat sessions persisted** — multi-turn conversations survive process restarts
- **`--frozen` pins to index revision** — a revision hash makes the index immutable and reproducible

Questions are ordered by dependency risk (blocking order), with the recommended ADR slug for each.

---

## Top-10 Questions

### Q1 — Vector-Backend Abstraction: Chroma as Default, FAISS as Opt-in

**Why it matters**  
`setup.py` today hard-codes `VectorDB(embedding_dim=4096)` (FAISS, in-memory) while Cohere
`embed-english-v3.0` outputs 1024-dim vectors — a dimension mismatch that crashes at search
time. The agreed contract flips the default to Chroma (persistent, local SQLite) and makes FAISS
a compile-time opt-in. Without a clean abstract protocol every other question below is blocked.

**Options**

| Option | Description | Trade-off |
|--------|-------------|-----------|
| A | `VectorBackend` Protocol + concrete Chroma & FAISS impls, selected via factory | Clean separation; extra module |
| B | Single class with mode enum | Lower churn but coupling grows with each backend |
| C | Plugin entry-points (`ragsearch.vector_backends`) | Maximum extensibility; over-engineered for v1 |

**Recommendation** — Option A.  
Define `VectorBackend` protocol in a new `ragsearch/backends/` subpackage. `ChromaBackend`
(default, persists to `.ragsearch/chroma.sqlite3`) and `FAISSBackend` (opt-in, in-memory or
mmap). Factory reads config key `vector_backend = "chroma" | "faiss"`. Embedding dimension is
always sourced from the embedding model adapter — never hardcoded.

**ADR to create:** `ADR-0001-vector-backend-abstraction`  
*Must capture:* interface contract, default/opt-in selection mechanism, migration path from
current `VectorDB`, embedding dimension source of truth.

---

### Q2 — LiteParse as Required Document-Parsing Layer

**Why it matters**  
The library today only ingests CSV / JSON / Parquet. Issue #18 mandates `@run-llama/liteparse`
as the default text-extraction pipeline for unstructured documents (PDF, DOCX, HTML, images).
Every ingestion path must pass through a parser before chunking and embedding. Coupling pandas
I/O directly to `setup()` prevents this.

**Options**

| Option | Description | Trade-off |
|--------|-------------|-----------|
| A | `DocumentParser` protocol + `LiteParseAdapter` as default | Pluggable, testable |
| B | Call LiteParse inline in `setup()` | Fast but un-testable and inflexible |
| C | Pre-processing CLI step that converts to CSV | Separates concerns but breaks streaming ingestion |

**Recommendation** — Option A.  
New `ragsearch/parsers/` subpackage. `LiteParseAdapter` wraps LiteParse and returns an iterator
of `ParsedDocument(text: str, metadata: dict)`. `setup()` dispatches on file extension:
structured files (CSV/JSON/Parquet) → pandas path; everything else → LiteParse path. Both paths
produce `ParsedDocument` objects consumed by the chunker.

**ADR to create:** `ADR-0002-document-parsing-pipeline`  
*Must capture:* LiteParse as required runtime dependency, `DocumentParser` protocol, dispatch
table for file-type routing, handling of parsing failures (partial ingestion vs. hard stop).

---

### Q3 — Incremental Indexing via Full-File SHA-256 Fingerprinting

**Why it matters**  
Today `setup()` re-embeds the entire dataset on every start-up — expensive for large corpora and
incompatible with the `--frozen` flag. Incremental indexing using the **full-file SHA-256**
digest lets the engine skip unchanged source files, making cold starts near-instant after the
first run.

**Options**

| Option | Description | Trade-off |
|--------|-------------|-----------|
| A | Manifest file (`.ragsearch/index.manifest.json`) keyed by `{root_path: sha256}` | Simple, human-readable |
| B | Embed manifest inside Chroma as a reserved metadata collection | Single store; harder to inspect |
| C | Git-object-like content-addressed store | Powerful; far beyond v1 scope |

**Recommendation** — Option A.  
On each ingestion run: compute `sha256(file_bytes)` for every source file; compare to manifest;
re-embed only changed/new files; tombstone deleted files in the backend; atomically write the new
manifest. Manifest format:

```json
{
  "version": 1,
  "created_at": "<iso8601>",
  "roots": [
    {
      "path": "/abs/path/to/root",
      "sha256": "<hex>",
      "indexed_at": "<iso8601>",
      "backend_ids": ["chroma-doc-id-1", "..."]
    }
  ]
}
```

**ADR to create:** `ADR-0003-incremental-indexing`  
*Must capture:* SHA-256 unit (full file, not chunk), manifest schema and atomic write strategy,
handling of renames (delete + re-index), interaction with `--frozen` flag (manifest is immutable
when frozen).

---

### Q4 — Multi-root Ingestion

**Why it matters**  
`setup(data_path: Path, ...)` accepts exactly one file. Enterprise use requires indexing multiple
directories, mixed file types, and glob patterns in a single engine instance.

**Options**

| Option | Description | Trade-off |
|--------|-------------|-----------|
| A | `setup(roots: list[Path | str | glob])` | Natural; one call |
| B | Config-file-driven roots (`ragsearch.toml` `[[roots]]` stanzas) | Reproducible; extra config layer |
| C | Mutating `add_root()` / `remove_root()` API | Incremental; complex lifecycle |

**Recommendation** — Option A with a thin config-file layer on top (B).  
`setup(roots=[...])` accepts `Path`, `str`, and glob patterns. Config file (`ragsearch.toml`)
can declare roots; CLI `--root` flag appends to the list. Each root carries optional metadata
(labels, priority) passed through to document metadata for filtering at query time.

**ADR to create:** `ADR-0004-multi-root-ingestion`  
*Must capture:* root resolution order, deduplication of overlapping roots, per-root metadata
propagation, interaction with SHA-256 manifest (manifest keyed per resolved root).

---

### Q5 — `--frozen` Flag and Index Revision Pinning

**Why it matters**  
Reproducibility and auditability require the ability to pin searches to a specific, immutable
snapshot of the index — identical results regardless of when or where the query runs. `--frozen`
must prevent any writes to the index while still allowing reads.

**Options**

| Option | Description | Trade-off |
|--------|-------------|-----------|
| A | Revision = SHA-256 of manifest snapshot; lock file refuses mutating ingestion | Lightweight, portable |
| B | Chroma native snapshots / export | Vendor-specific, limited portability |
| C | Full copy-on-write of Chroma DB directory per revision | Safe; storage-heavy |

**Recommendation** — Option A.  
Each successful ingestion writes an immutable revision file
(`.ragsearch/revisions/<revision_hash>.json`) containing the manifest snapshot and a pointer to
the Chroma collection name (or FAISS mmap path). `--frozen <revision_hash>` loads that revision
read-only; any `setup()` call with `frozen=True` raises `IndexFrozenError` on attempted writes.

**ADR to create:** `ADR-0005-index-revision-pinning`  
*Must capture:* revision hash computation, revision file schema, `IndexFrozenError` semantics,
interaction with multi-root and incremental indexing.

---

### Q6 — Chat Session Persistence

**Why it matters**  
Single-turn search (`engine.search(query)`) is insufficient for conversational use. Sessions must
persist across process restarts, support multi-turn history, and be independently reviewable or
exportable.

**Options**

| Option | Description | Trade-off |
|--------|-------------|-----------|
| A | SQLite-backed session store (`.ragsearch/sessions.sqlite`) | Zero extra deps; atomic writes |
| B | JSON file per session (`.ragsearch/sessions/<uuid>.json`) | Simpler; no atomic updates |
| C | In-memory sessions with optional pluggable backend | Flexible; violates "persisted" constraint |

**Recommendation** — Option A.  
`SessionManager` class owns the SQLite store. A `Session` holds `id`, `created_at`, `metadata`,
and an ordered list of `Turn(role, content, retrieved_chunks, timestamp)`. `engine.chat(
session_id, message)` appends to the session, passes the last-N turns as context to the LLM,
and returns a `Turn`. Sessions survive process restart; old sessions are queryable.

**ADR to create:** `ADR-0006-chat-session-persistence`  
*Must capture:* SQLite schema, `Turn` data model (including retrieved chunk references for
citations), context window management strategy (last-N turns, token counting), session lifecycle
(create / resume / archive / delete).

---

### Q7 — Embedding Model Abstraction for Local-First Operation

**Why it matters**  
`engine.py` and `setup.py` are hard-wired to `cohere.Client`. Local-first means Cohere calls
must be optional. Users on air-gapped machines need sentence-transformers or Ollama embeddings
out of the box.

**Options**

| Option | Description | Trade-off |
|--------|-------------|-----------|
| A | `EmbeddingModel` Protocol with adapters for Cohere, sentence-transformers, Ollama | Clean; minimal deps |
| B | LangChain/LlamaIndex embedding abstraction | Reuses ecosystem; heavy dependency |
| C | `isinstance` union checks | Anti-pattern; breaks extensibility |

**Recommendation** — Option A.  
Minimal `EmbeddingModel` protocol (structural subtyping via `typing.Protocol`). Ship
`CohereEmbeddingAdapter`, `SentenceTransformerAdapter` (local default), and
`OllamaEmbeddingAdapter`. `setup()` selects adapter from config key
`embedding_model = "sentence-transformers/all-MiniLM-L6-v2"` (local default) or
`"cohere:embed-english-v3.0"`. Each adapter exposes `embedding_dim` so the vector backend never
needs it hardcoded.

**ADR to create:** `ADR-0007-embedding-model-abstraction`  
*Must capture:* `EmbeddingModel` protocol signature, adapter naming convention, local default
model selection, how `embedding_dim` is propagated to the vector backend, batch size handling.

---

### Q8 — LLM / Generative Model Abstraction for Local-First Operation

**Why it matters**  
`engine.py` uses `cohere.Client` for generation as well as embedding. Local-first support
requires Ollama / LlamaCpp / vLLM as first-class options. Decoupling generation from embedding
also allows mixing providers (local embed + cloud LLM, or vice-versa).

**Options**

| Option | Description | Trade-off |
|--------|-------------|-----------|
| A | `LLMClient` protocol with per-provider adapters | Thin; requires one adapter per provider |
| B | OpenAI-compatible `/v1/chat/completions` as universal wire format | Works for Ollama, vLLM without custom adapters |
| C | LangChain `BaseLLM` | Reuses ecosystem; large dependency tree |

**Recommendation** — Option B layered on A.  
Define `LLMClient` protocol (A) but make the default adapter an OpenAI-compat HTTP client (B)
so Ollama, vLLM, and any OpenAI-compatible local server work with `base_url` config only. Cohere
gets its own adapter. Config:

```toml
[llm]
provider = "ollama"
model = "llama3.2"
base_url = "http://localhost:11434"
```

**ADR to create:** `ADR-0008-llm-abstraction`  
*Must capture:* `LLMClient` protocol, OpenAI-compat adapter spec, Cohere adapter, prompt
template ownership (engine vs. adapter), streaming support plan, timeout/retry policy.

---

### Q9 — CLI Interface and Configuration Schema

**Why it matters**  
There is currently no CLI and no declarative config format. The `--frozen` flag, multi-root, and
session management all require a CLI. A config file (`ragsearch.toml`) enables reproducible,
version-controlled configurations — critical for local-first, team workflows.

**Options**

| Option | Description | Trade-off |
|--------|-------------|-----------|
| A | `typer`-based CLI + `pydantic`-based config model loaded from `ragsearch.toml` | Type-safe; auto-generates help |
| B | `argparse` + manual dict validation | Lighter; verbose and error-prone |
| C | Click + Dynaconf | Powerful; over-engineered for v1 |

**Recommendation** — Option A.  
Top-level commands:

```
ragsearch index [--root <path>]... [--frozen <rev>]
ragsearch search "<query>" [--session <id>] [--top-k N]
ragsearch chat [--session <id>]
ragsearch sessions list|show|delete [<id>]
ragsearch revisions list|pin|unpin [<rev>]
```

Config hierarchy: CLI flags > `ragsearch.toml` > env vars > built-in defaults. Config schema
documented via Pydantic model and shipped as JSON Schema for editor support.

**ADR to create:** `ADR-0009-cli-and-config-schema`  
*Must capture:* command taxonomy, config file location resolution (`.ragsearch/config.toml` or
`$RAGSEARCH_CONFIG`), `--frozen` flag semantics in CLI, Pydantic config model as source of
truth, backward-compatibility policy for config keys.

---

### Q10 — Observability, Error Hierarchy, and Structured Logging

**Why it matters**  
All modules call `logging.basicConfig(...)` independently — a library anti-pattern that conflicts
with host-application logging config. There is no structured error hierarchy, no ingestion
metrics, and no query latency tracking. Enterprise adoption requires predictable, filterable logs
and machine-readable error codes.

**Options**

| Option | Description | Trade-off |
|--------|-------------|-----------|
| A | `structlog` + typed `ragsearch.errors` exception hierarchy | Zero config for library consumers |
| B | `loguru` | Simpler API; non-standard for library use |
| C | Standard `logging` with `NullHandler` + dataclass error context | Zero new runtime deps |

**Recommendation** — Option C for the library core (no `basicConfig` calls, one `NullHandler`
per module), with Option A available as an optional extras install
(`pip install ragsearch[observability]`). Exception hierarchy:

```
RagSearchError
├── IndexFrozenError
├── BackendError
├── ParsingError
├── EmbeddingError
└── SessionError
```

Emit structured `INFO` log events at ingestion start/finish (file count, embedding count,
elapsed time); `DEBUG` per batch; `WARNING` on SHA-256 manifest conflicts.

**ADR to create:** `ADR-0010-observability-and-error-hierarchy`  
*Must capture:* NullHandler best practice for library packages, `RagSearchError` hierarchy,
structured event schema (ingestion / query / session events), optional `structlog` integration,
metrics surface (counters vs. histograms, no mandatory external sink for local-first).

---

## Priority & Dependency Order

| Priority | # | Question | Blocks |
|----------|---|----------|--------|
| 1 | Q7 | Embedding model abstraction | Q1, Q3 |
| 2 | Q1 | Vector backend abstraction (Chroma default) | Q3, Q5 |
| 3 | Q2 | LiteParse integration | Q3, Q4 |
| 4 | Q3 | Incremental indexing / SHA-256 | Q5 |
| 5 | Q4 | Multi-root ingestion | Q3, Q9 |
| 6 | Q5 | `--frozen` / revision pinning | Q9 |
| 7 | Q8 | LLM abstraction | Q6 |
| 8 | Q6 | Chat session persistence | Q9 |
| 9 | Q9 | CLI & config schema | — |
| 10 | Q10 | Observability & error hierarchy | — |

**Suggested ADR authoring order:**
`0007 → 0001 → 0002 → 0003 → 0004 → 0005 → 0008 → 0006 → 0009 → 0010`

Each ADR lives at `docs/adr/ADR-XXXX-<slug>.md` and must be linked back to this issue before
implementation begins on the relevant component.
