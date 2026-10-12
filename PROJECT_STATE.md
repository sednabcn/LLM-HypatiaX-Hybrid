---
step: "Phase 1 done (validator + tests); waiting on 1.2 schema approval and Phase 0 decisions"
next_action: "Approve or change the schema in cases/P1_market_risk.yml header (1.2); then decide 0.1-0.4 in config/variants.yml and config/anchor.yml"
blockers: [A-016, A-014, A-006, A-007, A-012, A-020]
updated: 2026-10-12
---
# PROJECT_STATE

Paste or upload this file (or the Pages dashboard link) at the start of every agent session.
The YAML header above is machine-read by `scripts/export_state.py` -> `docs/state.json` -> dashboard.

## Rules
- Real numbers enter the repo only through `scripts/ingest_results.py` (never hand-typed).
- Agent changes arrive as a PR from a `agent/*` branch or as a `.patch` for `git am`.
- Update `step`, `next_action`, `blockers`, `updated` at the end of every session.

## Log
- 2026-10-12: validator, tests and P1 schema header rebuilt in this zip (scripts/validate_cases.py). Bridge files (export_state.py, dashboard, workflows) are NOT in this zip; re-apply the patch if you still want them.
- 2026-10-09: added state export, Pages dashboard, issue sync, results-branch workflow.
