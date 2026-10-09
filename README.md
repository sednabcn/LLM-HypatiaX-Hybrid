# HypatiaX Hybrid Methods -- paper project
Goals: (G1) win rate of 7 hybrid variants vs baselines, (G2) best extrapolation per domain, (G3) limitations/advantages.

| Read | Why |
|---|---|
| `DESIGN.md` | paper plan, experiments, tables/figures, status |
| `AGENT_PLAYBOOK.md` | the step-by-step loop for building, experimenting, auditing, iterating |
| `GUIDE_FOR_IMPLEMENTATION.md` | ordered commands |
| `audit/issues/issues.yml` | live issue register (drives Table T12) |

Quick start: `pip install -r requirements.txt && make test && make dryrun` (synthetic, watermarked PDF at `paper/main.pdf`).
Real results: clone the real repo to `external/LLM-HypatiaX-REPRO`, `source scripts/env.sh full`, run `scripts/run_hybrid_all.sh`, then `make real RESULTS=$RESULTS_DIR`.
**Numbers from `make dryrun` are random and must never be cited.**
