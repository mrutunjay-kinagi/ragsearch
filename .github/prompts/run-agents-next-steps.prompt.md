---
description: "Use when orchestrating the next multi-agent steps across issues #19, #21, #22, #23, #18, #1, and #8 for ragsearch."
---

# RAGSearch Multi-Agent Next Steps

Use this prompt to run your agents in the right order and produce closure plus implementation momentum.

## How To Use
1. Start with Product Owner and Architect to finalize planning outputs.
2. Use Maintainer to validate process and closure quality.
3. Move to Backend Dev + QA for implementation on issue #18.
4. Keep UX Docs in the loop for user-facing impact.

## Prompt 1: Product Owner (planning closure)
Post a final planning summary for issues #21, #22, and #23 and recommend close/keep-open status for each. Include:
- accepted outcome,
- unresolved items,
- concrete next tasks by role,
- closure recommendation.
Then post a single rollup checkpoint on #19.

## Prompt 2: Architect (technical decision pack)
For issue #18, produce a short architecture decision pack:
- LiteParse integration boundary,
- fallback behavior when LiteParse is unavailable,
- config surface,
- error taxonomy,
- testing strategy,
- migration/backward compatibility notes.
Format as ADR-short template.

## Prompt 3: Maintainer (governance gate)
Review the proposed closure comments and decision pack for policy compliance:
- branch target develop,
- issue linkage,
- test-first expectations,
- release-risk callout.
Return a go/no-go with required edits.

## Prompt 4: Backend Dev (implementation slice)
Create a test-first implementation plan for issue #18 with 2-3 PR slices max:
- exact files to touch,
- tests to add first,
- smallest reversible change in each PR,
- rollback plan.
Do not start coding until QA reviews the test strategy.

## Prompt 5: QA (test strategy gate)
Review the Backend Dev plan and produce:
- required tests,
- fixtures/mocks strategy,
- deterministic constraints,
- pass/fail gates before PR merge.

## Prompt 6: UX Docs (docs impact)
Prepare docs plan for issue #18:
- prerequisites (Node/LiteParse/etc.),
- quickstart updates,
- troubleshooting section,
- examples that must run before merge.

## Done Criteria
- #21, #22, #23 have final accepted summary comments.
- #19 has one rollup checkpoint comment.
- #18 has an approved test-first implementation plan and role-owned tasks.
- #1 has a scoped exception-handling test task queued.
- #8 is sequenced after ingestion baseline stabilization.
