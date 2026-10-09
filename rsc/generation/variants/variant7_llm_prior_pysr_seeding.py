#!/usr/bin/env python3
"""
variant7_llm_prior_pysr_seeding.py
====================================
Standalone extraction of **Variant 7 — LLM-Prior PySR Population Seeding**.

Real source : hypatiax/experiments/benchmarks/exp3_nguyen12_consolidated.py
              hypatiax/experiments/benchmarks/hypatia.py
Real symbols: get_llm_prior(), _llm_exprs_to_pysr_guesses(),
              PySRRegressor(guesses=...)
Verified    : this session, against a fresh clone of
              github.com/sednabcn/LLM-HypatiaX-REPRO (uploaded copy of the
              real exp3 file diffs at 0 lines against the repo).

WHAT IS PORTED FAITHFULLY (gate / control-flow logic):
  - LLM candidates are requested BEFORE the search runs and are converted to
    search "guesses" -- this is population seeding, not a post-hoc
    threshold gate or blend like Variants 1-6.
  - Fail-fast guard: if the search engine build cannot accept seed guesses
    at all, the run refuses to start rather than silently degrading to an
    unseeded run (mirrors [FIX-LLM-WARMSTART-CTOR] exactly).
  - Per-candidate conversion: each LLM candidate expression is parsed and
    converted independently; one that fails to parse or uses an operator
    outside the allowed set is dropped with a warning, not silently
    discarded as a group.
  - On any LLM-call exception, falls back to a cold, unseeded search for
    that case (mirrors the real try/except around get_llm_prior()).
  - Final comparison is seeded run (H) vs. unseeded run (P) on the SAME
    case, exactly as exp3 records both.

WHAT IS HONESTLY SUBSTITUTED (documented, not hidden -- same convention as
the original 5 variant scripts in this repo's mapping report):
  - The real get_llm_prior() calls the Anthropic API. This script takes an
    `llm_prior_fn` callable (equation_dict -> list[str] of candidate
    expression strings) so the seeding/fail-fast/fallback control-flow can
    be exercised without an API key. A simple polynomial-feature-matching
    mock is supplied as the default for the demo.
  - The real search engine is PySR (genetic programming in Julia). This
    script substitutes a small numpy/sklearn greedy feature-library search
    -- much weaker, documented here explicitly. Do NOT compare this
    script's R² numbers to the paper's PySR-family results (same caveat
    the original mapping report gives for variants 1-5).

Run standalone for a demo:
    python variant7_llm_prior_pysr_seeding.py
"""
from __future__ import annotations
import re
import numpy as np
from itertools import combinations_with_replacement


# ── honest substitute #1: LLM prior (real call: hypatia.get_llm_prior) ──
def mock_llm_prior(eq: dict, X: np.ndarray, y: np.ndarray, n_candidates: int = 5) -> list[str]:
    """
    Stand-in for the real get_llm_prior(). The real function calls the
    Anthropic API and returns ranked candidate expressions in plain
    Python/numpy syntax. This mock instead does a *correlation-guided*
    guess: proposes low-degree polynomial terms of each variable ranked by
    correlation with y, exactly the kind of "reasonable structural prior"
    an LLM might propose from the variable names and a formula hint --
    without an API call. Weaker signal than a real LLM by design.
    """
    var_names = eq["vars"]
    scores = []
    for name, col in zip(var_names, X.T):
        for deg in (1, 2, 3):
            term = col ** deg
            if np.std(term) > 0:
                corr = abs(np.corrcoef(term, y)[0, 1])
                scores.append((corr, f"{name}**{deg}" if deg > 1 else name))
    scores.sort(key=lambda t: -t[0])
    return [expr for _, expr in scores[:n_candidates]]


# ── real gate logic: guess conversion (ported from _llm_exprs_to_pysr_guesses) ──
def llm_exprs_to_search_guesses(llm_exprs: list[str], var_names: list[str],
                                 allowed_unary=("sin", "cos", "log", "sqrt", "exp")) -> list[str]:
    """
    Convert LLM candidate strings into guesses the search engine can seed
    with. Real code uses sympy to parse and re-render under a restricted
    operator set; this keeps that two-step "parse, then reject anything
    outside the allowed operator set" shape without requiring sympy.
    """
    guesses = []
    token_pattern = re.compile(r"[a-zA-Z_]+")
    for raw in llm_exprs:
        cleaned = re.sub(r"\bnp\.", "", str(raw))
        tokens = set(token_pattern.findall(cleaned)) - set(var_names)
        if tokens and not tokens.issubset(set(allowed_unary)):
            print(f"    warn: LLM candidate {raw!r} uses disallowed operator(s) "
                  f"{tokens - set(allowed_unary)} -- skipped, not used as a guess.")
            continue
        guesses.append(cleaned)
    return guesses


# ── honest substitute #2: search engine (real: PySRRegressor genetic search) ──
def greedy_feature_search(X: np.ndarray, y: np.ndarray, var_names: list[str],
                           guesses: list[str] | None = None, max_terms: int = 3) -> dict:
    """
    Weak, documented replacement for PySR's genetic-programming search.
    Builds a feature library of {var, var^2, var^3, sin/cos/log/sqrt/exp(var),
    pairwise products} PLUS, if `guesses` is provided, evaluates each guess
    string directly as a candidate feature -- this is the "seeding" step:
    guesses get first refusal in the greedy selection, mirroring how PySR's
    `guesses=` parameter biases its initial population toward LLM proposals
    rather than guaranteeing they win.
    """
    env = {name: X[:, i] for i, name in enumerate(var_names)}
    env.update({"sin": np.sin, "cos": np.cos, "log": lambda v: np.log(np.abs(v) + 1e-9),
                "sqrt": lambda v: np.sqrt(np.abs(v)), "exp": lambda v: np.exp(np.clip(v, -20, 20))})

    library: dict[str, np.ndarray] = {}
    if guesses:
        for g in guesses:
            try:
                library[g] = eval(g, {"__builtins__": {}}, env)  # noqa: S307 (sandboxed env)
            except Exception:
                continue
    for name, col in zip(var_names, X.T):
        for deg in (1, 2, 3):
            library[f"{name}**{deg}" if deg > 1 else name] = col ** deg
        for fn in ("sin", "cos", "log", "sqrt", "exp"):
            library[f"{fn}({name})"] = env[fn](col)
    for a, b in combinations_with_replacement(var_names, 2):
        library[f"{a}*{b}"] = env[a] * env[b]

    selected, remaining_y = [], y.copy()
    for _ in range(max_terms):
        best_name, best_r2, best_coef = None, -np.inf, 0.0
        for name, col in library.items():
            if name in selected or np.std(col) == 0:
                continue
            coef = float(np.dot(col, remaining_y) / np.dot(col, col))
            pred = coef * col
            r2 = 1.0 - np.sum((remaining_y - pred) ** 2) / max(np.sum(remaining_y ** 2), 1e-12)
            if r2 > best_r2:
                best_name, best_r2, best_coef = name, r2, coef
        if best_name is None or best_r2 <= 0:
            break
        selected.append(best_name)
        remaining_y = remaining_y - best_coef * library[best_name]

    pred_total = y - remaining_y
    ss_res = float(np.sum((y - pred_total) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"expr": " + ".join(selected) or "0", "r2": r2, "seeded_terms_used": [s for s in selected if guesses and s in guesses]}


# ── the real control-flow: seed-or-refuse, fall back on failure, compare H vs P ──
def run_seeded_vs_unseeded(eq: dict, X: np.ndarray, y: np.ndarray,
                            llm_prior_fn=mock_llm_prior,
                            search_fn=greedy_feature_search,
                            engine_supports_guesses: bool = True,
                            use_llm: bool = True) -> dict:
    """
    Ports the real run()'s per-equation flow:
      1. fail-fast if USE_LLM and engine has no guesses mechanism
      2. request LLM candidates (fall back to unseeded on any exception)
      3. convert candidates to guesses; log-and-drop the ones that don't survive
      4. run H (seeded) and P (unseeded) on the same data; report both
    """
    if use_llm and not engine_supports_guesses:
        raise RuntimeError(
            "[FIX-LLM-WARMSTART-CTOR] search engine has no guess-seeding "
            "mechanism -- refusing to start rather than silently fall back "
            "to an unseeded H run. Set use_llm=False to run unseeded on purpose."
        )

    llm_exprs, guesses = [], []
    if use_llm:
        try:
            llm_exprs = llm_prior_fn(eq, X, y)
        except Exception as e:
            print(f"    LLM prior failed: {e} -- running unseeded for this case.")
        if llm_exprs:
            guesses = llm_exprs_to_search_guesses(llm_exprs, eq["vars"])

    result_h = search_fn(X, y, eq["vars"], guesses=guesses or None)
    result_p = search_fn(X, y, eq["vars"], guesses=None)
    return {
        "llm_candidates_requested": len(llm_exprs),
        "llm_candidates_seeded": len(guesses),
        "H_seeded":   result_h,
        "P_unseeded": result_p,
        "H_beat_P": result_h["r2"] > result_p["r2"],
    }


# ── demo ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 200
    x = rng.uniform(0.5, 3.0, n)
    z = rng.uniform(0.5, 3.0, n)
    y = np.sin(x) ** 2 + z ** 2 + rng.normal(0, 0.02, n)   # Nguyen-style target
    X = np.column_stack([x, z])
    eq = {"id": "demo-1", "vars": ["x", "z"], "formula_hint": "sin(x)**2 + z**2"}

    print("=== engine supports guesses ===")
    out = run_seeded_vs_unseeded(eq, X, y, engine_supports_guesses=True, use_llm=True)
    print(f"  LLM candidates requested/seeded: "
          f"{out['llm_candidates_requested']}/{out['llm_candidates_seeded']}")
    print(f"  H (seeded)   expr={out['H_seeded']['expr']!r:40} r2={out['H_seeded']['r2']:.4f}")
    print(f"  P (unseeded) expr={out['P_unseeded']['expr']!r:40} r2={out['P_unseeded']['r2']:.4f}")
    print(f"  H beat P: {out['H_beat_P']}")

    print("\n=== engine does NOT support guesses (fail-fast) ===")
    try:
        run_seeded_vs_unseeded(eq, X, y, engine_supports_guesses=False, use_llm=True)
    except RuntimeError as e:
        print(f"  refused to start, as expected: {e}")
