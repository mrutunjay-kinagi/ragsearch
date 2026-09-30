# ADR-0010: Default chunking for unstructured files and an answer context budget

- Status: accepted (2026-09-29)
- Date: 2026-09-29
- Related issues: #77, #88

## Context

Every parser yields one document per file: the built-in PDF parser joins all pages with newlines and records no page boundaries, and DOCX and HTML are likewise one text each. With the default `RowChunkingStrategy`, each file was a single chunk. Since #85, `answer()` sends each retrieved chunk's full text to the LLM, so every question about an 80-page PDF sent the whole document: about 57,700 estimated tokens for NIST SP 800-63B. That is enough to overflow smaller context windows (Ollama defaults to 4096 tokens), and it's expensive everywhere. Retrieval could also only point at "the whole file".

## Decision

1. **Unstructured files are chunked by default.** When `setup()` gets no `chunking_strategy` for a PDF, DOCX, HTML, Markdown or text file, it uses `FixedWordChunkingStrategy(words_per_chunk=150, overlap_words=30)`. Structured files (CSV, JSON, Parquet) keep `RowChunkingStrategy`, and `RagSearchEngine`'s own default stays row-level. `FixedWordChunkingStrategy` gains `overlap_words` (default 0, so existing explicit use is unchanged).
2. **Chunking is part of the embedding-cache identity.** The manifest (schema v3) records `describe_chunking_strategy()`, the class name plus public scalar settings. A change re-embeds everything once with a warning, as for model changes (ADR-0006 amendment 1). Manifests without the field are treated as row chunking, the only behaviour that existed before.
3. **`answer()` has a context budget.** `max_context_tokens` on `setup()` and `RagSearchEngine` (default 3000 estimated tokens; `None` disables it), overridable per call. Sources are added in score order until the next would exceed the budget. If the first source alone is too big, a truncated prefix of it is sent. The response reports `context_sources`, `context_truncated` and `context_tokens_estimate`, and `citations` lists only the sources that were sent.
4. **Token estimate:** `ceil(len(text) / 3.5)`, with no tokenizer dependency. Measured with Cohere's tokenizer (`command-a-03-2025`) on the claim documents and the NIST PDF: 3.55–4.87 characters per token, 4.27 overall. So 3.5 errs on the high side (it over-counts), which keeps a budget conservative.

## Evidence for 150 words / 30 overlap

Document: NIST SP 800-63B (80 pages, public domain). There are 14 questions whose answers sit on known pages from 5 to 80, 7 of them on page 57 or later. A question counts as a hit when a retrieved chunk contains the answer's evidence phrase (whitespace-insensitive). Embeddings: Cohere `embed-v4.0`, used exactly as the library uses them.

| words / overlap | chunks | hit@1 | hit@3 | hit@5 | late-page hit@5 (of 7) | est. tokens, top 5 |
|---|---|---|---|---|---|---|
| 100 / 0 | 274 | 9 | 9 | 10 | 6 | 986 |
| 100 / 20 | 342 | 7 | 10 | 11 | 6 | 1006 |
| 150 / 0 | 183 | 8 | 10 | 11 | 6 | 1458 |
| **150 / 30** | 228 | **8** | **11** | **12** | **7** | **1506** |
| 200 / 0 | 137 | 8 | 9 | 10 | 6 | 2006 |
| 200 / 40 | 171 | 6 | 10 | 11 | 6 | 1935 |
| 250 / 0 | 110 | 6 | 10 | 12 | 7 | 2389 |
| 250 / 50 | 137 | 7 | 10 | 11 | 6 | 2473 |
| 300 / 0 | 92 | 7 | 11 | 11 | 5 | 2875 |
| 300 / 60 | 114 | 5 | 11 | 13 | 7 | 3066 |
| 400 / 0 | 69 | 5 | 9 | 11 | 5 | 3915 |
| 400 / 80 | 86 | 5 | 10 | 10 | 4 | 3870 |
| 500 / 0 | 55 | 4 | 12 | 13 | 6 | 5153 |
| 500 / 100 | 69 | 5 | 10 | 12 | 6 | 4853 |
| whole document (0.1.x) | 1 | – | – | – | – | 57723 |

- hit@1 falls as chunks grow (9 at 100 words to 4 at 500 words), and context cost grows in proportion.
- 150/30 gives near-best hit@1, the second-best hit@3 and hit@5, and all 7 late-page answers in the top 5, at about 1,500 tokens for `top_k=5`. 300/60 and 500/0 reach hit@5 = 13 only at double or more the context, with hit@1 of 5 and 4.
- Repeating a configuration moved hit@1 by one question (embedding calls aren't bit-identical), so differences of ±1 are noise.

The quickstart sample eval cases (`samples/quickstart/eval_cases.json`) cover the claims CSV. That file stays on row chunking, so it can't inform unstructured chunk sizes.

## Budget default (3000)

At 150-word chunks, `top_k=5` needs about 1,500 estimated tokens, and `top_k=10` about 3,000, so the budget rarely binds by default. It stops oversized custom chunks or large `top_k` values from overflowing small windows. 3000 plus the instructions, question and answer fits Ollama's default 4096-token context (https://docs.ollama.com/faq). Large-context models such as Cohere Command A (256K, https://docs.cohere.com/docs/command-a) can raise it.

## Limitation of the evidence

The defaults (150 words, 30 overlap, budget 3000) were tuned on **one document**, the NIST SP 800-63B PDF, a long, well-structured technical standard. They should be re-checked on a wider mix of document types (short letters, forms, contracts, scanned or table-heavy files, multi-document corpora) in the 0.3.0 retrieval benchmark before being treated as general-purpose. Whole-document questions over small documents can also lose completeness with top-k chunks. See #120.

## Consequences

- Answers cite specific passages instead of whole files, and prompts stay small.
- Indexing an unstructured file makes more embedding calls: 228 chunks for the 80-page PDF, about 1.25× the words because of overlap.
- Upgrading re-embeds cached unstructured files once (the chunking identity changes); structured files keep their cache.
- Page numbers are still not recorded (the PDF parser joins pages). Page-level citations belong with the parser work in #83/#102.
