# Guide for implementation (ordered; each step has a gate)
0. `pip install -r requirements.txt && make test` -- gate: 8 passed.
1. `git init` (done) ; `git clone https://github.com/sednabcn/LLM-HypatiaX-REPRO external/LLM-HypatiaX-REPRO` -- supplies the real `experiment_protocol_*.py`, `run_comparative_suite_benchmark_{v2,pca}.py`.
2. `source scripts/env.sh smoke` ; `python rsc/experiments/benchmarks/experiment_protocol_hybrid.py --list-coverage` -- gate: no "fallback catalog" lines (closes A-016); compare with `config/experiments.yml`.
3. `bash scripts/run_hybrid_all.sh --dry-run` then a smoke run (`--step hybrid_exp1`, 2 cases, seed 42, needs ANTHROPIC_API_KEY) -- gate: result JSON has `seed`, and ingest reports no unmapped methods / missing r2_far (A-013).
4. Full run: `source scripts/env.sh full` ; `bash scripts/run_hybrid_all.sh` (CI: `ci_runner_hybrid.yml`).
5. `make real RESULTS=$RESULTS_DIR` -- ingest -> audit -> tables/figures -> PDF.
6. `make refs` (internet) -- gate: all cited keys verified; extend toward >100 entries by search, not recall.
7. `python .github/scripts/assert_real_build.py` -- release gate.
8. Iterate: every defect -> entry in `audit/issues/issues.yml` + patch in `audit/patches/`; re-run only affected steps.
