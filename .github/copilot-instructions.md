# RAGSearch Multi-Agent Operating Instructions

Use these rules for all agent-driven work in this repository.

## Roles In Scope
- Architect
- Backend Dev
- QA
- Maintainer
- Product Owner
- UX Docs

## Workflow Contract
1. Start from an issue with clear acceptance criteria.
2. Plan first in issue comments if requirements or design are unclear.
3. Keep changes small and reversible.
4. Link PRs to issues and include a test plan.
5. Only target the develop branch for pull requests.
6. Do not merge with failing tests or missing required review feedback.

## Delivery Gates
- Product Owner: issue is clear and prioritized.
- Architect: interface and trade-offs are documented for non-trivial changes.
- Backend Dev: implementation and tests are complete.
- QA: test verdict and risk level are explicit.
- Maintainer: CI/release/readability checks pass.
- UX Docs: docs impact is addressed for user-facing changes.

## Comment Templates
Use concise issue comments with these sections:
- Summary
- Decision or Proposal
- Evidence or Test Notes
- Risks
- Next tasks with owners

## Current Epic Priority (Issue 19)
1. Close planning-loop issues: #21, #22, #23 with final accepted summaries.
2. Post one rollup checkpoint comment on #19.
3. Begin implementation slice on #18 (LiteParse integration) with test-first plan.
4. Address exception-handling hardening in #1 with tests.
5. Schedule citations (#8) immediately after ingestion baseline stabilizes.
