# PROCESS_LOG
One line per finished step: `date | playbook step | minutes | what you did | what went wrong / surprised you`

2026-10-10 | 1.1 | - | case validator + tests added by the agent | validator cannot see generator-level input correlation (moved to 5.3)
2026-10-12 | 1.1 | - | validator rebuilt (earlier copy was not in LLM-HypatiaX-Hybrid.zip: scripts/ cases/ rsc/ were empty) + 27 planted-defect tests + P1 schema header | PyYAML reads 1.0e6 as a string (needs 1.0e+6): custom loader added; random sampling misses poles: dense scan + near-singular warning added (cannot tell a pole from steep growth, so it is a warning)
2026-10-12 | 5 (methods layer) | - | agent refactor: baselines + V1-V7 on one stack (rsc/methods), config/api_env.yml, optional RAG, row adapter; 136 new tests in rsc/experiments/tests | GUIDE step 0 gate says '8 passed' but the baseline is 27 (now 163 + 1 skip without config/variants.yml); DESIGN.md and the standalone scripts disagree on V1 gate and V3 weighting (F1, F2); V5 documented seeds make 5-seed sweeps identical (F3); scripts/ingest_results.py not in the zip so row field names are unconfirmed
