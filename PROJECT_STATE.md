---
step: "Phase 4 - real runs not started; bridge (GitHub <-> web) being added"
next_action: "Place real LLM-HypatiaX-REPRO beside this repo (A-016), run `source scripts/env.sh smoke`, then smoke-run exp1 and ingest"
blockers: [A-016, A-014, A-006, A-007, A-012, A-020]
updated: 2026-10-09
---
# PROJECT_STATE

Paste or upload this file (or the Pages dashboard link) at the start of every agent session.
The YAML header above is machine-read by `scripts/export_state.py` -> `docs/state.json` -> dashboard.

## Rules
- Real numbers enter the repo only through `scripts/ingest_results.py` (never hand-typed).
- Agent changes arrive as a PR from a `agent/*` branch or as a `.patch` for `git am`.
- Update `step`, `next_action`, `blockers`, `updated` at the end of every session.

## Log
- 2026-10-09: added state export, Pages dashboard, issue sync, results-branch workflow.
