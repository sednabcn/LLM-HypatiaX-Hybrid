# PLAYBOOK - Anchor paper ("Hybrid"): 150 formulas, 5 protocols, all variants on the SAME cases

Follow the steps in order. A step is finished only when its "Done when" check passes. Do not start a step
whose predecessor is unfinished. After every step add one line to PROCESS_LOG.md (what, minutes, what broke):
that log becomes the app specification.

## Rules (never broken)
1. You decide what is true (thresholds, mappings, which source is right). The agent only proposes.
2. Nothing synthetic reaches the paper: the banner must read REAL before any number is used.
3. The analysis plan is committed BEFORE any real run. Deviations are logged, never silent.
4. Two variants are compared only if their comparability key matches: case set hash, split, seed, threshold,
   engine version, LLM model + temperature, config hash.
5. Every change goes through a branch and a PR. Case sets are frozen by hash; edits create a new version.
6. Contamination wording: "reduced by construction and measured by a probe", never "uncontaminated".
7. Session start: paste PROJECT_STATE.md and the case files you changed.

## Phase 0 - Decisions (you; blocking)
| # | Decision | Done when |
|---|---|---|
| 0.1 | Variant mapping: paper names (EHSDeFi, HLLMNN, SymLLM, HDSv50_2, ensemble) -> V1..V5 and the baselines; which class is V5 (retry loop, physics fallback, or HDSv50_2) | config/variants.yml updated, A-001 and A-022 closed |
| 0.2 | Confirm the 5 protocols and the 10/10/10 tier rule; confirm what "validation models" covers in P1-P3 | written in cases/README or PROCESS_LOG |
| 0.3 | Pin LLM model and temperature; seeds; PySR settings; threshold(s) | config/anchor.yml committed |
| 0.4 | Compute plan: where the runs execute and the API budget cap | a number in anchor.yml |

## Phase 1 - Case tooling (agent) - DONE in this patch
1.1 scripts/validate_cases.py + tests on synthetic planted defects. Done when: `python -m pytest rsc/experiments/tests` is green.
1.2 You review the schema (cases/P1_market_risk.yml header) and approve or change fields. Done when: you say "schema approved".

## Phase 2 - Analysis plan (agent drafts, you approve)
2.1 config/analysis_plan.yml: metrics, tie tolerances (0, 1e-6, 1e-3, 0.01), thresholds, tests (Cliff's delta, Friedman/Nemenyi, Wilcoxon-Holm,
    bootstrap over FAMILIES), sensitivity axes, ranking-fragility (Kendall tau). Done when: committed and tagged `plan-v1`.
2.2 Planted-effect test: sensitivity.py recovers known differences in synthetic data. Done when: test green.

## Phase 3 - Cases, one protocol at a time (P1, then P2, P3, P4, P5)
3.1 Agent drafts 30 cases (10 canonical, 10 perturbed, 10 composed) with sources.
3.2 Sources verified by search (not memory); you review every formula as the domain expert; set source_verified: true.
3.3 `python scripts/validate_cases.py` -> 0 errors, 0 warnings explained.
3.4 `python scripts/validate_cases.py --freeze P1` writes the hash. Done when: cases/FROZEN.json has the protocol.
Do not start P2 before P1 is frozen.

## Phase 4 - Contamination measurements (needs API key, small cost)
4.1 Recall probe on frozen cases: ask for the formula from the name only, check equivalence in SymPy, store recall per case.
4.2 Export both views (`--export-views`): semantic and anonymous. Done when: no leak reported.

## Phase 5 - Pipeline readiness
5.1 Schema columns added: threshold, llm_model, temperature, config hash, case-set hash, comparability key.
5.2 Adapters for the real result formats (protocol_core_*, noise/sample sweeps, DeFi seed files). Closes A-010, A-011, A-013.
5.3 Data-generator check: inputs are generated independently (correlation of generated inputs ~ 0). Catches the Reserve-Ratio class of bug.
5.4 Retro-test: reproduce chosen numbers of the HypatiaX paper from its existing result files (21/30, 23/30 vs 30/30, 51/60 vs 49/60,
    67/74 vs 44/74). Done when: each matches, or the difference is logged as an audit item.

## Phase 6 - Smoke and bridge
6.1 Smoke: 2 cases, seed 42, small PySR budget, capped API calls. Done when: result JSONs exist and ingest cleanly under the real guards.
6.2 Bridge control: re-run ONE already-published method next to the new ones. Done when: it matches its old numbers (then old runs may be reused)
    or it does not (then everything is re-run).

## Phase 7 - Pilot: P1 only, 1 seed, all applicable variants
Done when: ingest has no unmapped methods, banner REAL, release gate has no data problems, cost per case is known.

## Phase 8 - Full runs, per protocol
5 seeds x 2 splits. After EACH protocol: ingest, audit, update PROJECT_STATE.md, commit.

## Phase 9 - Analysis exactly as locked in plan-v1
Per-protocol tables first (a ranking, or "no separable winner", or "not applicable" for each cell). Pooled results are secondary and
shown with both case-weighted and domain-weighted views.

## Phase 10 - Short paper
3 tables, 3 figures, references verified, release gate passes, no DRY-RUN text. Done when: `make paper` succeeds.

## Phase 11 - Extract the app specification
Sort every PROCESS_LOG step into: engine does it / agent does it with approval / only a person can do it. That table is the app spec.
