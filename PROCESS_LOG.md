# PROCESS_LOG
One line per finished step: `date | playbook step | minutes | what you did | what went wrong / surprised you`

2026-10-10 | 1.1 | - | case validator + tests added by the agent | validator cannot see generator-level input correlation (moved to 5.3)
2026-10-12 | 1.1 | - | validator rebuilt (earlier copy was not in LLM-HypatiaX-Hybrid.zip: scripts/ cases/ rsc/ were empty) + 27 planted-defect tests + P1 schema header | PyYAML reads 1.0e6 as a string (needs 1.0e+6): custom loader added; random sampling misses poles: dense scan + near-singular warning added (cannot tell a pole from steep growth, so it is a warning)
