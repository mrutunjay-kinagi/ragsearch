# Prioritized Backlog — Enterprise-Grade RAG Library
<!-- Product Owner Agent deliverable for Epic #19 -->
<!-- Generated: 2026-03-28 -->

## Context

This backlog was produced by the **Product Owner Agent** in response to Epic #19.
It covers all open issues (#1, #8, #9, #18, #20–#23), grades them by urgency and
value, writes acceptance criteria for the next three executable issues, and lists
net-new issues that are needed but not yet created.

---

## Target Personas

| Persona | Description | Primary Need |
|---------|-------------|-------------|
| **Platform Engineer** | Builds internal search/Q&A tools on top of company data | Reliable ingestion of PDFs, source attribution, stable API |
| **Data Scientist** | Rapid prototyping of RAG pipelines | Easy plug-in of custom embeddings/LLMs, clean Python API |
| **Enterprise Adopter** | Deploys in regulated/financial contexts (e.g., stock market data, legal docs) | Audit trail, citations, multi-tenancy, observability |
| **Open-Source Contributor** | Extends or maintains the library | Clear interfaces, ADRs, contribution guidelines |

---

## Prioritized Backlog

### 🔴 NOW — Sprint 0 (Foundation & Governance)

These are sequencing blockers. Nothing else can proceed well without them.

| # | Issue | Why NOW |
|---|-------|---------|
| 1 | **#20** — Review current codebase | All architecture decisions and planning depend on a shared understanding of what exists |
| 2 | **#22** — Assign agent roles / RACI | Defines who owns each workstream; prevents duplicate or conflicting work |
| 3 | **#23** — Define agent operating guidelines | Establishes ADR format, discussion rules, and decision-capture process |

### 🟡 NEXT — Sprint 1 (Highest-Value Features)

These unlock enterprise adoption and directly address user pain points.

| # | Issue | Why NEXT |
|---|-------|---------|
| 4 | **#18** — Integrate liteparse for unstructured document parsing | Enterprise users ingest PDFs/DOCX/HTML; current library only handles DataFrames |
| 5 | **#8** — Add citations to responses | Response trust is non-negotiable for enterprise; users must know which chunks were cited |
| 6 | **#21** — Gather & prioritize top 10 arch questions | Needed before making irreversible design decisions (interfaces, plugin system) |
| 7 | **NEW-1** — Define abstract interfaces (Extractor/Chunker/Embedder/Retriever) | Required to make #18 and future backends slot in cleanly; architectural prerequisite |

### 🟢 LATER — Sprint 2+ (Scale & Extensibility)

Valuable but can follow after Sprint 1 foundations are solid.

| # | Issue | Why LATER |
|---|-------|---------|
| 8 | **NEW-2** — Multi-backend LLM support (OpenAI, vLLM, HuggingFace) | Removes Cohere lock-in; critical for broad adoption but needs interface work first |
| 9 | **NEW-3** — CI/CD quality gates (coverage ≥ 80%, linting, type-checking) | Needed before the library is "enterprise-grade" but not blocking feature dev |
| 10 | **NEW-4** — ADR-001: Establish ADR framework and document first decisions | Captures outcomes from #23; first ADR covers extraction strategy |
| 11 | **#1** — Exception handling / structured exception hierarchy | Improves reliability; can be done incrementally alongside feature work |
| 12 | **NEW-5** — Async ingestion pipeline for large document sets | Needed for production-scale use; depends on interface design (NEW-1) |
| 13 | **#9** — Product use case: PDF → financial data mart (Indian stock market) | Becomes a reference cookbook/sample once liteparse and citations are in |
| 14 | **NEW-6** — Quickstart guide + API reference documentation | DX polish; needed before public 1.0 announcement |

---

## Acceptance Criteria — Next 3 Issues to Execute

### Issue #20 — Review current codebase

**Problem statement:** No shared baseline understanding of the codebase exists.
All agents are making decisions without knowing what is already built or what gaps
need to be closed.

**Target users:** All agents (Architect, Dev, QA, PO, Maintainer, DX).

**Proposed solution:** The Architect Agent produces a written summary directly in
issue #20 covering: entrypoints, module responsibilities, data flow, test coverage
snapshot, and a prioritised gap list.

```
Acceptance Criteria:
- [ ] A comment is posted on #20 listing all Python entrypoints (modules, CLI, notebooks).
- [ ] Each core module (engine, utils, vector_db, config) has a one-paragraph
      responsibility summary.
- [ ] A "Gaps" table lists at least 5 enterprise-grade capabilities that are absent
      or incomplete (e.g., unstructured ingestion, citation, multi-LLM, async, observability).
- [ ] Each gap is tagged with severity: Blocker / High / Medium / Low.
- [ ] The summary notes the current LLM/embedding provider lock-in (Cohere-only).
- [ ] Test coverage percentage is reported (from existing test suite).
- [ ] The deliverable is posted within the agreed 48-hour SLA from kickoff.

Non-goals:
- No code changes are made in this issue.
- No design decisions are finalised here; that is for #21 and ADRs.

Testability notes:
- Review is a written artifact; "done" = comment with all checklist items present.
- QA Agent confirms coverage stat is accurate by running `pytest --cov` against the repo.
```

---

### Issue #18 — Integrate liteparse as default unstructured document parser

**Problem statement:** `RagSearchEngine` accepts only `pd.DataFrame` input.
Enterprise users need to ingest PDFs, DOCX, HTML, and other unstructured formats.
Issue #9 (stock market PDF use case) is completely blocked without this.

**Target users:** Platform Engineers, Data Scientists, Enterprise Adopters.

**Proposed solution:** Add a new `DocumentLoader` / `Parser` layer that uses
`liteparse` (run-llama/liteparse) to extract text from unstructured files and
convert them into a `pd.DataFrame` suitable for the existing pipeline.

```
Acceptance Criteria:
- [ ] A new module `ragsearch/document_loader.py` (or equivalent) is added.
- [ ] `DocumentLoader.load(path: str | Path) -> pd.DataFrame` accepts at minimum:
      PDF, DOCX, HTML, plain-text files.
- [ ] liteparse is added as a dependency in `pyproject.toml`.
- [ ] `RagSearchEngine` can be initialised from a loaded DataFrame (existing API
      is unchanged — no breaking changes).
- [ ] A sample script under `samples/` demonstrates loading a PDF and running a query.
- [ ] Unit tests cover: successful load of each supported format, graceful error on
      unsupported format, empty document handling.
- [ ] Extraction failures raise a typed exception (e.g., `DocumentLoadError`) — not
      a generic `Exception`.
- [ ] README is updated with a "Loading unstructured documents" section.

Non-goals:
- Audio, image, or video ingestion are out of scope for this issue.
- Real-time/streaming ingestion is out of scope.
- No changes to the embedding or retrieval pipeline.

Risks / assumptions:
- liteparse API may change; pin to a specific version.
- Large PDFs may hit memory limits; document a `max_pages` safeguard.

Testability notes:
- Golden test fixture: include a small PDF (≤ 2 pages) and DOCX in `tests/fixtures/`.
- Integration test: load fixture → index → query → assert ≥ 1 result returned.
```

---

### Issue #8 — Add citations to responses

**Problem statement:** When `search()` returns results, users have no way to know
which chunk of which document was used. Enterprise contexts (legal, financial,
compliance) require source attribution; without it the library cannot be trusted
in production.

**Target users:** Enterprise Adopters, Platform Engineers.

**Proposed solution:** Augment the `search()` return value with a `citations` list.
Each citation contains the source document identifier, chunk index / page number,
and the verbatim text excerpt that was matched. If an LLM is used to synthesise an
answer, the LLM prompt should be updated to request in-line citations.

```
Acceptance Criteria:
- [ ] `search()` returns a list of result dicts; each dict gains a `citations` key.
- [ ] Each citation contains at minimum:
      `source` (file name / document ID), `chunk_id` (int), `excerpt` (str ≤ 300 chars).
- [ ] When `chromadb_search()` is used, citations reference the ChromaDB document ID.
- [ ] Existing callers that ignore the `citations` key continue to work (non-breaking).
- [ ] A `CitationFormatter` utility can render citations as Markdown footnotes.
- [ ] Unit tests cover: citation present in single-document result, multiple citations
      for multi-document result, citation absent when no relevant document found.
- [ ] Integration test: ingest a known document, query for known content, assert the
      citation `source` matches the ingested document name.
- [ ] API docs / docstrings are updated to describe the new `citations` field.

Non-goals:
- Storing citation history / audit log is out of scope (defer to observability issue).
- Cross-document citation deduplication is out of scope.

Testability notes:
- Use `sample_data.csv` in existing fixtures; enrich with a `source_doc` column.
- Deterministic test: seed data + fixed query → assert exact citation source.
```

---

## Missing Issues — Proposed for Creation

The following issues do not exist yet but are necessary for the enterprise roadmap.
Each is described with enough detail to be filed directly.

---

### NEW-1 · Define abstract interfaces: Extractor, Chunker, Embedder, Retriever

**Labels:** `enhancement`, `architecture`

**Description:**
The current codebase has no formal interface/protocol layer. `RagSearchEngine` is
tightly coupled to `CohereClient` and FAISS/ChromaDB. Before integrating liteparse
(#18) or adding multi-LLM support, we need stable abstract interfaces so that each
component can be swapped independently.

Define Python `Protocol` (or `ABC`) classes for:
- `Extractor` — takes a raw document path, returns text chunks
- `Chunker` — takes a long text, returns a list of chunk strings
- `Embedder` — takes a list of strings, returns a list of float vectors
- `Retriever` — takes a query vector + top_k, returns ranked results

Each interface should live in a new `ragsearch/interfaces.py` module.
Concrete implementations (Cohere, FAISS, ChromaDB, liteparse) should implement these.

**Acceptance Criteria:**
- `interfaces.py` added with typed Protocol/ABC definitions for all four concepts.
- Existing `CohereClient` and FAISS/ChromaDB wrappers annotated to satisfy interfaces.
- No breaking changes to the public `RagSearchEngine` API.
- Unit tests verify that existing implementations satisfy the protocols.

---

### NEW-2 · Add multi-backend LLM and embedding support (OpenAI, vLLM, HuggingFace)

**Labels:** `enhancement`

**Description:**
Currently `RagSearchEngine` accepts only `cohere.Client` objects. Enterprise
teams use OpenAI, Azure OpenAI, vLLM self-hosted endpoints, and HuggingFace
sentence-transformers. Remove the Cohere lock-in by accepting any object that
satisfies the `Embedder` and `LLMClient` protocols defined in NEW-1.

Deliver at minimum:
- `OpenAIEmbedder` wrapper implementing `Embedder`
- `VLLMEmbedder` wrapper implementing `Embedder`
- `HuggingFaceEmbedder` wrapper implementing `Embedder`
- Updated `RagSearchEngine.__init__` to accept any `Embedder` instance

**Acceptance Criteria:**
- At least two non-Cohere embedders are implemented and tested.
- `RagSearchEngine` accepts any `Embedder`-compliant object.
- Migration guide in docs shows Cohere → OpenAI swap.

**Dependencies:** NEW-1 (interfaces must be defined first)

---

### NEW-3 · CI/CD quality gates: coverage threshold, linting, and type-checking

**Labels:** `enhancement`, `ci`

**Description:**
There are no automated quality gates beyond running tests. An enterprise-grade
library needs:
- Minimum 80% line coverage enforced in CI (fail PR if below threshold)
- `ruff` or `flake8` linting (no errors allowed)
- `mypy` static type-checking (strict mode, no new untyped public APIs)
- Pre-commit hooks for local enforcement

**Acceptance Criteria:**
- `.github/workflows/ci.yml` (or equivalent) runs on every PR: test, lint, type-check.
- PR fails if coverage < 80%.
- PR fails on any `ruff`/`flake8` error.
- `mypy` runs on `libs/ragsearch/` and reports no errors on new code.
- `pre-commit` config added to repo root.
- README updated with "Contributing" section referencing the quality gates.

---

### NEW-4 · ADR-001: Establish Architectural Decision Record framework

**Labels:** `documentation`, `architecture`

**Description:**
Agent operating guidelines (#23) reference ADRs but there is no ADR directory,
template, or tooling in the repo. This issue establishes the framework so that
decisions made in technical Q&A cycles (#21) can be captured and found later.

Deliverables:
- Create `docs/adr/` directory with a `README.md` explaining the format.
- Add `docs/adr/template.md` (MADR or Nygard-style).
- Author `docs/adr/ADR-001-document-extraction-strategy.md` capturing the
  liteparse decision from #18.

**Acceptance Criteria:**
- `docs/adr/` exists with README and template.
- ADR-001 is complete and covers: context, decision, consequences, alternatives considered.
- ADR-001 cross-links to #18 and #21.

---

### NEW-5 · Async ingestion pipeline for large document sets

**Labels:** `enhancement`

**Description:**
Production workloads ingest thousands of documents. The current pipeline is fully
synchronous. Add an async ingestion mode using `asyncio` (and optionally a task
queue) so that large batch jobs do not block the event loop.

**Acceptance Criteria:**
- `async def ingest(documents: list[Path]) -> None` added to `RagSearchEngine`
  (or a separate `AsyncIngestionPipeline` class).
- Existing synchronous API remains unchanged.
- A benchmark is included showing async path handles 1,000-document batch
  without blocking.
- Unit tests for async path using `pytest-asyncio`.

**Dependencies:** NEW-1 (interfaces), #18 (liteparse integration)

---

### NEW-6 · Quickstart guide and API reference documentation

**Labels:** `documentation`

**Description:**
The current README is brief and example-thin. Developers evaluating the library
need a clear quickstart (≤ 5 minutes to first answer) and a full API reference.

Structure:
1. **Quickstart** — install, ingest a PDF, run a query (based on #18 + #8)
2. **Guides** — structured data (CSV), unstructured data (PDF/DOCX/HTML)
3. **API Reference** — auto-generated from docstrings (Sphinx/MkDocs)
4. **Recipes** — stock market PDF use case (#9), customer support RAG

**Acceptance Criteria:**
- `docs/quickstart.md` walks user from `pip install` to first query in < 20 lines.
- `docs/guides/` contains at minimum the structured and unstructured data guides.
- `make docs` (or equivalent) builds HTML API reference without errors.
- All public functions/classes have complete docstrings (args, returns, raises).

---

## Roadmap Summary

```
Now:  #20 (codebase review) → #22 (roles) → #23 (guidelines)
Next: #18 (liteparse) + #8 (citations) + #21 (arch questions) + NEW-1 (interfaces)
Later: NEW-2 (multi-LLM) + NEW-3 (CI gates) + NEW-4 (ADRs) + #1 (exceptions)
       + NEW-5 (async) + #9 (stock market demo) + NEW-6 (docs)

Risks:
- liteparse API stability (pin version; abstract behind Extractor interface)
- Cohere dependency is deep; NEW-1 must precede NEW-2 to avoid rework
- No CI gates today means quality debt can accumulate rapidly

Open questions:
- Will vLLM be self-hosted only, or also support cloud APIs?
- Is multi-tenancy (per-user namespaces in ChromaDB) a Sprint-2 or Sprint-3 concern?
- Should the Flask UI be a separate optional package or stay in core?
```
