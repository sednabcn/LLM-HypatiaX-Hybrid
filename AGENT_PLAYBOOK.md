# Agent playbook: build, experiment, audit, iterate

The loop is: **plan -> build the smallest runnable slice -> run it -> check it against an independent expectation -> record what is wrong -> patch -> re-run only what changed.** Never advance on "it ran"; advance on "it ran and a check I did not write for it passed".

## Phase 0 - Ground the inputs (before writing code)
1. List every artefact; read what is in context, list archives instead of extracting blindly.
2. Extract **facts, not intentions** from each source and tabulate disagreements (here: PDF says 5 variants, package 7; sh says methods 1-9). Disagreements become issues (A-001...), not silent choices.
3. Probe the environment (LaTeX? PySR/Julia? API key? real repo?). Whatever is missing defines what you can *not* claim.

## Phase 1 - Freeze the vocabulary
One ID table (`config/variants.yml`), one experiment table (`config/experiments.yml`) verified against the executable (`run_hybrid_all.sh`), not against prose. Rule: **config mirrors code; prose mirrors config.**

## Phase 2 - Build the pipeline backwards from the paper
Paper tables/figures -> DB schema -> ingest -> stats -> exports. Each stage has a contract test:
ingest (guards: synthetic rows refused, engine < v5.0 rejected, seed/engine gaps reported, unmapped methods reported), stats (win/tie rule, bootstrap CI contains estimate, Holm monotone), build (every `\input` target exists).

## Phase 3 - Dry-run on synthetic data (plumbing only)
`make dryrun`: random generator with *planted* defects (a Portfolio-Variance-style catastrophe, an old-engine row). Pass criterion: the audit finds the planted defects, tables/figures/PDF build, every output is watermarked DRY-RUN, `assert_real_build.py` refuses it. This catches bugs like the runner not writing seeds, and my own stale-DB mistake. It proves nothing about the methods.

## Phase 4 - Experiments (needs the real repo, Julia, API key)
Order by cost: `--list-coverage` -> smoke (2 cases, seed 42, 200 generations) -> one protocol, one seed -> full steps. After each stage:
- ingest and read the summary line first (unmapped, missing fields, rejected rows);
- compare 2-3 numbers against the repo's own published tables;
- only then spend the next stage's budget.
Seeds: scripts default to 42; the paper design needs `EFFECTIVE_SEEDS=42,43,44,45,46` (A-018).

## Phase 5 - Audit (continuous, not at the end)
Three sources of findings, each logged in `audit/issues/issues.yml` with kind, status, affected experiments, patch:
1. **Static** (`audit_checks.py`): leakage tokens in prompts, test-residual std, `use_llm` never read. Hits are *leads* with file:line; a human or you read the code.
2. **Data**: engine_version missing, zero API calls for an LLM variant (silent no-op signature), in-sample R2~1 with far R2 << -1.
3. **Design review**: read the variant source against its documentation (this found V5 = retry loop not template fallback, V3 weighting mismatch, V5's 5x budget).
A finding is *fixed* only when a patch exists and a check that previously failed now passes. Fix rate = fixed / total, per experiment (T12).

## Phase 6 - Iterate
For each open issue pick by (affects headline claims) x (cheap to fix). Patch -> add/extend a test -> re-run affected steps only -> re-ingest with `--reset` for those experiments -> rebuild -> update issue status. Re-derive tables; never hand-edit generated files.

## Phase 7 - Write only what the data supports
Sections 5 and abstract are generated from tables; claims in prose must cite a table/figure. If a comparison is undefined (Nguyen-12 has no baseline) say so in the paper. Release gate: real DB only, references verified, no synthetic marker anywhere.

## Stop conditions (ask the human)
Real repo layout differs from assumptions; a variant's documented behaviour contradicts its code; budget for API/Julia is needed; a result looks too good (R2 = 1.000) - treat as leakage until shown otherwise.
