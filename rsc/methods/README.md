# rsc/methods -- the methods layer (baselines + V1..V7) on one shared stack

Status: PROPOSED by an agent branch (`agent/variants-refactor`). Nothing here is frozen and no number it prints is a result.

## Layout (where each thing lives in this repo)

```
config/api_env.yml              the one API environment: model, temperature, key VARIABLE NAME, mode, budget
config/formula_kb.example.yml   six-card EXAMPLE knowledge base for the optional RAG (not for anchor runs)
.env.example                    how the key is supplied (normally just `export ANTHROPIC_API_KEY=...`)
rsc/methods/env.py              ApiEnv: reads api_env.yml, resolves the key, decides live vs mock
rsc/methods/llm.py              your llm.py + temperature policy, cache-key fix, token/budget tracking, FixtureLLM
rsc/methods/nn.py               your nn.py + run_seed and tuple widths (defaults reproduce the old behaviour)
rsc/methods/pysr.py             your pysr.py, UNCHANGED
rsc/methods/common.py           sandbox, r2, Task, FitResult (one copy; replaces the block pasted in every script)
rsc/methods/hypotheses.py       the shared "ask the LLM" step; RAG plugs in here
rsc/methods/search.py           one interface over real PySR / a labelled substitute
rsc/methods/retrieval.py        optional RAG: offline TF-IDF, leak filter, prompt audit (OFF by default)
rsc/methods/stack.py            build_stack(): llm + nn + search + retriever; strict=True for paper runs
rsc/methods/variants/           B_LLM B_NN B_PYSR V1 V2 V3 V4 V5 V6 V7: fit(task, stack) -> FitResult
rsc/methods/rows.py             FitResult -> result row with the comparability fields and the synthetic marker
rsc/methods/smoke.py            `make methods-smoke`: wiring check on one synthetic task (prints only, writes nothing)
rsc/experiments/tests/test_methods_*.py   136 tests, run by the existing `make test`
```
Ids are the FROZEN ones of `config/variants.yml` (B_LLM, B_NN, B_PYSR, V1..V7; `V4_seed` / `V4_fallback` are V4 modes).

```python
from rsc.methods.stack import build_stack
from rsc.methods.variants import make
from rsc.methods.rows import result_row

stack = build_stack(strict=True, pysr_populations=2, pysr_generations=500)   # refuses mock LLM / substitute search / sklearn NN
fit = make("V4", mode="hybrid").run(task, stack)
row = result_row("V4", task, fit, stack, X_far=Xf, y_far=yf, engine_version=..., case_set_hash=..., split="random", threshold=...)
```

## What is and is not verified

Verified (tests, numpy / scikit-learn only): gate logic of V1..V7; every LLM-based variant surviving bad LLM answers;
the formula sandbox; env / key / mode rules; the LLM client policies (against a fake SDK object); the cache-key fix;
the NN seed fix; RAG retrieval and leak filtering; row fields, the synthetic marker, `r2_far` recomputation; the ten
method names against the frozen alias regexes of `config/variants.yml`.

NOT verified: real PySR/Julia (`PySRSearch` is written against your pysr.py, never executed), the live Anthropic API,
the torch NN path, and `scripts/ingest_results.py` (not in the tree this was built against: confirm the row field names).
V1-V4 are re-implementations of documented logic, NOT your real classes; nothing was diffed against LLM-HypatiaX-REPRO.
The substitute search is not PySR: its output is wiring evidence only.

## Findings to log in `audit/issues/issues.yml` (IDs to be assigned by you; "F" numbers are provisional)

Disagreements between sources (stop condition of AGENT_PLAYBOOK: "a variant's documented behaviour contradicts its code"):
* F1  V1 gate. DESIGN.md s2 says "scipy refit + R2>=0.85 & margin>0.05". The standalone script this port used says fitted R2 >= 0.97, else an
      ensemble band, else NN. The code here follows the script (`fitted_threshold=0.97`); the DESIGN rule is NOT implemented because
      "margin" is undefined in the sources. Needs the real `EnhancedHybridSystemDeFi`.
* F2  V3 weighting. DESIGN.md says weight proportional to R2/sigma_resid; the script (and this code) uses 1/sigma_resid. AGENT_PLAYBOOK already
      records a "V3 weighting mismatch": the real class decides.
* F3  V5 seeds. Documented `42 + attempt`: the experiment seed never reaches V5, so a 5-seed V5 sweep is one run repeated. Default here mirrors the
      documentation (`seed_mode="fixed42"`); `seed_mode="task"` is available but changes the comparability key: decide and log.
* F4  NN seed. nn.py derives its seed from the description only, and says "as in the original ImprovedNN". If the real ImprovedNN does the same,
      B_NN (and every NN inside V1, V2, V3, V6) has no seed variance. `run_seed` fixes it in this layer; the real baseline needs checking.
* F5  Two LLM/PySR configurations in the tree: `model_config.yml` (claude-sonnet-5-5, T=0, 30 populations, 10000 generations) vs
      `config/anchor.yml` draft (claude-sonnet-5, T=0.25, 2 populations, 500 iterations). api_env.yml currently carries the model_config values.
      This is decision 0.3.
* F6  V6 reads the test features (never labels) to detect extrapolation; no other variant does. `use_test_features=False` removes it.
* F7  Cost per case: V7 = 1 LLM call + 2 searches, V5 = up to 5 searches, V6 trains about 7 NNs. Needed for decision 0.4.

Defects found in the earlier standalone variant scripts (NOT confirmed in the real repo; each has a regression test here):
SyntaxError from an LLM answer escaped the ValueError handlers (confirmed by running the script); `9**9**9` hangs the sandbox (confirmed);
NameError on an unknown variable uncaught; V2 chose a non-existent formula when LLM and NN both scored -inf; V3 blended zeros for an unusable
formula; V4 matched seed terms by substring ("x" in "exp(...)"); llm.py cache key ignored temperature; llm.py silently dropped a rejected
`temperature`.

## RAG (optional, OFF by default)
Turning it on changes the experiment: treat it as a declared ablation (`build_stack(retriever=Retriever.from_yaml(kb))`; rows record `rag: true`
in `provenance`). The main risk is leakage: `Retriever.view_for_case(case_id, gt_formula)` drops the case's own card and any card equal to or
containing the ground truth, and `audit_prompt_leak(prompt, gt_formula)` checks a prompt before it is sent. `Task` has no ground-truth field.
The example KB has six generic textbook cards; a real KB needs its own curation and leak audit (PLAYBOOK rule 6 wording applies: "reduced by
construction and measured by a probe").
Not built, each is also an experiment change: self-consistency sampling, critique-and-repair, a units check (pint is already in installation.yml).
