# ragsearch 0.2.0 — Release Plan for Claude Code

This file is the single source of instructions for the 0.2.0 release. It replaces earlier chat instructions. Re-read it at the start of every session.

## Goal

Ship **0.2.0, "It works"** to PyPI.

**Done means:** a stranger runs `pip install ragsearch` in a fresh environment, copies any README example, and it works, including on an ordinary multi-page PDF. CI is green on every supported Python version.

## Decisions already made (don't reopen)

- **Scope is lean.** Only the work items below. Everything else goes to 0.2.1 or 0.3.0.
- **Chunking and a context size limit (#77, #88) are in 0.2.0.** Since #85, `answer()` sends full text; with one chunk per document, a real multi-page PDF can overflow the model's context window.
- **Drop Python 3.9** (end-of-life since October 2025). Require Python 3.10+.
- **Accept ADR-0009** (read-only SQL for aggregation).
- **The web server binds to localhost by default.**
- **No Trusted Publishing, docs-site automation, cookbook rewrite, or ROADMAP.md in 0.2.0.** File them as 0.2.1 issues.

## Ground rules (every work item)

1. One work item = one issue = one PR, targeting `develop`. Never `master`, except the final release PR in Item 8.
2. Reproduce before fixing: a failing test first.
3. Run the full suite before every PR. No new failures. Current baseline after #103: 171 passed, 1 skipped, 2 xfailed.
4. Preserve public contracts (`setup()`, `search()`, `answer()`, `/query`, `/answer`). Behaviour changes get a changelog entry explaining them.
5. Check external facts (versions, APIs, tools) in official docs, never memory. Cite them in the PR.
6. Every PR adds its own CHANGELOG entry under **Unreleased**.
7. After every push, verify the remote SHA matches local. Never pipe commands in a way that hides exit codes.
8. Never delete branches, change repo settings, tag, or publish. Prepare these and ask.
9. **After each work item: stop, report (what changed, PR link, test results, surprises), and wait for my go-ahead.** Merge a PR only when I say so, with a regular merge commit.

---

## Item 1: Merge #103 and issue housekeeping

No code changes.

1. Merge #103 into `develop` (regular merge commit). Verify the remote SHA; run the full suite on `develop` (expect 171 passed).
2. Create milestones **0.2.0**, **0.2.1**, and **0.3.0**. Don't change existing milestones; report what they contain.
3. Close as fixed in develop, each with a comment linking the fixing PR and "Ships in 0.2.0", assigned to 0.2.0: #82 (#90), #91 (#93), #80 (#99), #100 (#103), and the Task 3 issue if one exists (#101).
4. #79: comment that root logging and key handling were fixed by #101, and the 0.0.0.0 bind moves to the new issue in step 5. If nothing else remains, close it; otherwise list what's left.
5. Split the localhost default out of #98 into a new issue in 0.2.0 (Item 3). The rest of #98 stays, in 0.3.0.
6. Create 0.2.0 issues for Items 4–8 below. Assign #77 and #88 to 0.2.0.
7. Create 0.2.1 issues: PyPI Trusted Publishing; automatic docs-site deployment; cookbook and notebook rewrite; ROADMAP.md; refresh of the #84 epic; opening issue creation to the public with a bug-report template.
8. Assign to 0.3.0: #76, #78, #83, #92, #102, and the remainder of #98. Leave #94, #96, #97 unassigned.
9. Report the full list of open issues with milestones, and flag anything not covered here.

## Item 2: Default chunking and context size limit (#77 + #88, one PR)

1. **Check first** whether any parser already splits documents by page or section. Report what you find.
2. **Chunking (#77):** for unstructured inputs, default to a word-window chunker with overlap (extend `FixedWordChunkingStrategy`, or add a recursive paragraph → sentence strategy). Keep row-level chunking as the default for structured inputs. Propose the default chunk size and overlap, and justify them with retrieval results on the sample eval cases in `samples/quickstart/`.
3. **Cache key:** make the chunking strategy and its settings part of the incremental-indexing cache identity, so changing them triggers a safe re-index with a clear warning (the same pattern as #95).
4. **Size limit (#88):** add a context budget to answer assembly (propose the parameter name and default). Add chunks in score order until the budget is reached, and report in the `answer()` response when truncation happened. An approximate token count is fine; don't add a heavy tokenizer dependency.
5. **Tests:** a multi-page document produces several chunks; a question about late-document content retrieves the right chunk; oversized context is trimmed to the budget; a chunking-settings change triggers re-indexing. Include the DOCX claim document as a regression case (its content is in tables, #83, so record the current answer rather than expecting a perfect one).
6. **Real-world check:** in a fresh virtualenv with only the built wheel, run `answer()` on a genuinely long PDF (at least 50 pages, e.g. a public-domain report) with default settings. It must answer without a context-window error.
7. Update README and docs where chunking defaults are described.

## Item 3: Web server listens on localhost by default

1. `run()` defaults to host `127.0.0.1` instead of `0.0.0.0`, and accepts `host` and `port` parameters.
2. Document the risk of binding to other interfaces without authentication: anyone who can reach the port can spend the owner's API credits and read the indexed data.
3. Changelog entry explaining the behaviour change, and how to restore network access deliberately.
4. The rest of #98 (blocking behaviour, `create_app()`, the script-exit bug) stays in 0.3.0.

## Item 4: Drop Python 3.9

1. Require Python `>=3.10` in `pyproject.toml`; regenerate the lock file.
2. Remove the 3.9-only pins (e.g. `click` 8.1.8) and re-check the resolved versions for known advisories (OSV), including `soupsieve`.
3. Update docs and any version references. Changelog entry.

## Item 5: CI on pull requests

A new workflow, separate from `python-publish.yml`:

1. **Tests:** run the full suite on every pull request and on pushes to `develop` and `master`, across every Python version the package supports (3.10+).
2. **Packaging check:** build the wheel, install it in a clean environment with no repo checkout, and run the quickstart on a CSV, a PDF, and a DOCX using the offline test models. (Most of this cycle's biggest bugs, #82, #91, and #100, would have been caught by this.)
3. **Docs build** must pass.
4. Add a CI status badge to the README once the workflow exists.

## Item 6: Publish workflow checks

In `python-publish.yml`:

1. Fail early if the release tag (without the leading `v`) doesn't match the version in `pyproject.toml`.
2. Restrict or remove the manual `workflow_dispatch` trigger so it can't publish from an arbitrary branch.
3. Add a path to publish release candidates to **TestPyPI**.
4. Keep the existing token for now (Trusted Publishing is a 0.2.1 issue).

## Item 7: Metadata and minimal docs update

PyPI metadata is frozen at publish time, so this must land before the release. **Describe only what 0.2.0 actually does**: no claims about abstention, trust guarantees, scalability, or ChromaDB (still broken, #76).

1. `pyproject.toml`: a one-line description (e.g. *"RAG over CSVs and documents, with citations, ingestion diagnostics, and a built-in evaluation harness."*); fix the malformed author email (no spaces inside the address); add project URLs (homepage, documentation, changelog, issues), keywords, and classifiers (Python 3.10+, license, development status). Verify with a build and `twine check`.
2. Update the package docstring in `libs/ragsearch/__init__.py`.
3. Mark ADR-0009 as **accepted**.
4. Run the dataset analytics cookbook and notebook against `develop`. Don't rewrite them (that's 0.2.1); if they break or rely on aggregate questions RAG can't answer, add a short note at the top saying so.
5. Propose new GitHub "About" text and topics for me to apply.

## Item 8: Release preparation (prepare only; I publish)

1. Write `docs/release-checklist.md`.
2. Open a PR on `develop` that bumps the version to `0.2.0` and moves the changelog's Unreleased entries under a `0.2.0` heading (date to be filled at release).
3. Give me exact commands for the TestPyPI rehearsal: publish `0.2.0rc1`, install it in a fresh virtualenv, run every README example and the quickstart.
4. After the rehearsal passes, open the release PR from `develop` into `master`.
5. List the final steps for me: merge the release PR, tag `v0.2.0`, publish the GitHub Release with the changelog as notes, and confirm the package on PyPI.

---

## Out of scope for 0.2.0

- **0.2.1:** Trusted Publishing, docs-site deployment, cookbook rewrite, ROADMAP.md, epic refresh, public issue creation.
- **0.3.0:** numeric columns (#78), ChromaDB (#76), DOCX tables (#83), query embeddings (#92), LiteParse (#102), the rest of the web server (#98), retrieve-wide reranking.
- **Unassigned:** #94, #96, #97.

If you discover something that seems to belong in 0.2.0, file it, explain why, and ask. Don't add it on your own.
