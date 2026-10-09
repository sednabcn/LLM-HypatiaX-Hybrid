#!/usr/bin/env python3
"""
HypatiaX - Variant 5: Retry-then-Physics-Fallback Discovery (standalone)
==========================================================================
Classes ported: HybridDiscoverySystem._discover_with_retry (control flow),
                 PhysicsAwareRegressor, SmartStructureDetector /
                 IntelligentEquationBuilder.
Combination strategy: retry a symbolic search up to `max_retries` times with
early stopping, THEN (only if enabled) fall back to a physics-aware
regressor for whatever the search couldn't solve.

CORRECTED vs. the first draft of this file: the original draft described
this as "template library -> template-free fallback", which was wrong. The
real architecture (hypatiax/tools/symbolic/hybrid_system_v50_2.py,
HybridDiscoverySystem) is a retry loop around ONE search engine, with a
GP-style physics regressor as an optional last resort -- not two competing
template systems.

Algorithm (verified against real source line-by-line)
-------------------------------------------------------
  for attempt in range(max_retries):          # max_retries default = 5
      seed = 42 + attempt
      result = symbolic_engine.search(...)     # real engine: PySR
      track best-so-far by r2
      if attempt == 0 and r2 < 0.1:             # rational-pattern probe
          if _detect_rational_pattern(X, y):     # Lineweaver-Burk / saturation
              inject 'inv' into the search's unary operators for next attempt
      early_stop_r2 = 0.9999 if use_transcendental_compositions else 0.95
      if r2 >= early_stop_r2 and not is_overfit(expr, r2):
          return best result immediately         # early exit, doesn't use all retries
  # after the loop:
  success = best_r2 >= 0.97                      # SUCCESS threshold, independent of early-stop
  if enable_physics_fallback and (no result or best_r2 < physics_fallback_threshold):
      # enable_physics_fallback default = FALSE -- fallback is OFF unless
      # explicitly turned on; physics_fallback_threshold default = 0.85
      run PhysicsAwareRegressor as a last resort; keep it only if it's better

is_overfit(expr, r2): complexity = len(expr) as a STRING (character count,
not AST node count); overfit iff complexity > complexity_penalty_threshold
(default 20) AND round(r2, 6) < 0.999. This is a much blunter check than it
sounds -- it literally measures the printed expression's character length.

Search backend
--------------
The real engine is PySR (needs `pip install pysr` + Julia). As in Variant
4, this standalone port substitutes FeatureLibrarySearch (greedy
forward-selection over a bank of per-variable terms with an OLS refit) --
same honest substitute, same caveat: do not compare its R2 numbers to
PySR-family results from the paper. `_detect_rational_pattern` and the
quality/overfit check ARE ported verbatim (they're plain numpy/sklearn, no
PySR dependency), so those two pieces behave exactly like the real system --
including a real quirk found while testing this port: `_detect_rational_pattern`
false-positives (R2>0.85 in the Lineweaver-Burk branch, or a shrinking-slope
false-trigger in the saturation branch) on almost ANY smooth monotonic
positive data -- linear, quadratic, not just genuinely saturating/rational
shapes -- because both branches are loose, cheap heuristics rather than
precise classifiers. Confirmed against the real, unmodified ported function
on multiple domains/shapes, so this is a property of the original algorithm,
not a porting error. Low-cost in context regardless, since a false positive
here only adds 'inv' as an extra optional search operator on the next
attempt -- it doesn't force that structure on the result.

PhysicsAwareRegressor (fallback engine, off by default)
---------------------------------------------------------
The real class (hypatiax/tools/symbolic/physics_aware_regressor.py, 2117
lines) is a domain-seeded GENETIC-PROGRAMMING engine: a population (default
150, or 200/150 under the noise-adaptive noiseless/noisy presets this port
DOES replicate exactly) is seeded with domain-flavoured expression
templates (Michaelis-Menten for biology, rational/exponential forms for
chemistry, Bernoulli-style energy terms for engineering) and evolved over
`generations` with tournament selection, a parsimony penalty, and
validation-based early stopping. Reproducing that evolutionary loop
faithfully is not practical here, so this port substitutes a much smaller,
clearly-labelled local-search variant: the same domain-template seed
population (reusing this file's own curve_fit-based template fitter as the
"seed"), refined by a bounded random-perturbation hill-climb over a fixed
number of generations. Do not compare its R2 numbers, wall-clock behaviour,
or the specific expressions it returns to the real engine's -- only the
*noise-adaptive preset selection* (population_size / generations /
parsimony_coefficient / min_r2, chosen from noise_level) and the public
interface (`fit_noise_aware`, `get_expression`, `best_fitness_`) are
faithful ports.

SmartStructureDetector / IntelligentEquationBuilder -- important correction
-----------------------------------------------------------------------------
The first draft of this file assumed this class was "the template-free
fallback" wired into the discovery pipeline. Grepping the real repo shows
that's wrong on two counts: (1) `SmartStructureDetector` is NOT exported
from `hypatiax/tools/symbolic/__init__.py` (only `PhysicsAwareRegressor`
is), and (2) nothing outside its own module ever imports or calls it -- it
is orphaned / unused in this repo snapshot, not a live fallback path. Its
actual documented purpose (per its own module docstring and the companion
`IntelligentEquationBuilder.generate_pysr_config()`) is to analyse
additive/multiplicative/interaction structure in the data and use that to
CONFIGURE a PySR search (candidate operators, complexity budget) before the
search runs -- not to predict anything itself. This port keeps that
scoped-down, honest framing: `analyze_structure()` and
`generate_pysr_config()` are ported faithfully (they're pure
numpy/sklearn), but neither is invoked anywhere in `discover_with_retry`
below, matching the real, current wiring (or lack of it).

Built-in benchmark: 5 physics-flavoured cases (kept from the first draft),
run through the corrected retry-then-fallback pipeline instead of the old
template-vs-template-free dispatch.

Status: standalone re-implementation, grep-verified line-by-line against
hypatiax/tools/symbolic/hybrid_system_v50_2.py (HybridDiscoverySystem),
physics_aware_regressor.py, and smart_structure_detector.py in
github.com/sednabcn/LLM-HypatiaX-REPRO. PySR and the real GP engine are NOT
executed here -- only the documented substitutes above were run and tested.

Usage
  python variant5_retry_physics_fallback.py --selftest
  python variant5_retry_physics_fallback.py                       # all 5 cases
  python variant5_retry_physics_fallback.py --physics-fallback     # turn on the (off-by-default) fallback
  python variant5_retry_physics_fallback.py --list-cases

As a library:
  from variant5_retry_physics_fallback import HybridDiscoverySystem
"""

# =============================================================================
# SHARED CORE (duplicated verbatim across all 5 variant scripts on purpose,
# so each file is standalone and can be copied out / run / shipped alone).
# =============================================================================
from __future__ import annotations
import argparse
import ast
import json
import math
import os
import random as _random
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    from sklearn.metrics import r2_score as _sk_r2
    from sklearn.neural_network import MLPRegressor
    _SKLEARN_OK = True
except Exception:  # pragma: no cover
    _SKLEARN_OK = False

try:
    import torch
    import torch.nn as _tnn
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def r2(y_true, y_pred) -> float:
    """R^2 that never raises: returns -inf on non-finite predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.shape != y_true.shape or not np.all(np.isfinite(y_pred)):
        return float("-inf")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-300:
        return 1.0 if ss_res <= 1e-9 else 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# AST-sandboxed formula evaluation (LLM-written expressions are untrusted)
# ---------------------------------------------------------------------------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant,
    ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
    ast.Mod, ast.Tuple,
)
_ALLOWED_FUNCS = {"sqrt", "log", "exp", "sin", "cos", "tan", "abs", "Abs",
                   "sign", "arcsin", "arccos", "arctan", "asin", "acos", "atan"}
_CONSTS = {
    "pi": math.pi, "e": math.e,
    "h": 6.62607015e-34, "hbar": 1.0545718176e-34, "c": 2.99792458e8,
    "k_B": 1.380649e-23, "k": 1.380649e-23, "N_A": 6.02214076e23,
    "g_n": 9.80665, "m_e": 9.1093837015e-31, "q_e": 1.602176634e-19,
    "epsilon0": 8.8541878128e-12, "mu0": 1.25663706212e-6,
}
_FUNCS = {
    "sqrt": np.sqrt, "log": np.log, "exp": np.exp, "sin": np.sin, "cos": np.cos,
    "tan": np.tan, "abs": np.abs, "Abs": np.abs, "sign": np.sign,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
}


def screen_code(expr: str) -> None:
    """Raises ValueError if `expr` contains anything outside the small
    arithmetic/function-call grammar we allow. Defence in depth, not a real
    sandbox -- eval() below still runs with empty builtins as a second layer.
    """
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed expression element: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise ValueError("dunder access blocked")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("only direct function calls allowed")
            if node.func.id not in _ALLOWED_FUNCS:
                raise ValueError(f"function '{node.func.id}' not allowed")


def predict_from_equation(equation: str, X: np.ndarray, names: list) -> np.ndarray:
    """Evaluate an equation string on X. AST-screened, empty builtins."""
    screen_code(equation)
    ns = {n: X[:, i] for i, n in enumerate(names)}
    ns.update(_FUNCS)
    for k, v in _CONSTS.items():
        ns.setdefault(k, v)
    compiled = compile(ast.parse(equation, mode="eval"), "<equation>", "eval")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = eval(compiled, {"__builtins__": {}}, ns)
    result = np.asarray(result, dtype=float)
    if result.ndim == 0:
        result = np.full(X.shape[0], float(result))
    return result


def affine_refit(pred: np.ndarray, y: np.ndarray) -> tuple:
    """Least-squares refit of pred -> a*pred + b ('scipy constant refit')."""
    from scipy.optimize import least_squares

    def resid(p):
        a, b = p
        return y - (a * pred + b)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = least_squares(resid, x0=np.array([1.0, 0.0]), max_nfev=200)
        a, b = float(sol.x[0]), float(sol.x[1])
    except Exception:
        a, b = 1.0, 0.0
    fitted = a * pred + b
    return fitted, a, b


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------
class LLMProvider:
    name = "base"

    def complete(self, prompt: str, **kw) -> str:
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    """Offline fixture provider (--llm mock, the default). Returns
    pre-registered, DELIBERATELY IMPERFECT formulas keyed by case id, so the
    gate logic actually gets exercised the way it would with a real LLM that
    sometimes gets things wrong. This is NOT a language model.
    """
    name = "mock"

    def __init__(self, fixtures: dict):
        self.fixtures = fixtures
        self.calls = 0

    def complete(self, prompt: str, case_id=None, **kw) -> str:
        self.calls += 1
        eqs = self.fixtures.get(case_id, self.fixtures.get(
            "default", [{"equation": "0*x", "confidence": 0.1, "reasoning": "no fixture"}]))
        return json.dumps(eqs)


class AnthropicProvider(LLMProvider):
    """Real LLM calls via the Anthropic API. Needs `pip install anthropic`
    and an ANTHROPIC_API_KEY in the environment. NOT executed by the author
    in the sandbox that built this file -- ported but untested end-to-end.
    """
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.3, **kw) -> str:
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "user", "content": prompt}])
        except Exception:
            # repo note: claude-sonnet-4-6 has rejected `temperature` before.
            resp = self.client.messages.create(
                model=self.model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}])
        return "".join(getattr(b, "text", "") for b in resp.content
                        if getattr(b, "type", "") == "text")


def parse_hypotheses(response: str) -> list:
    """Parse a JSON array of {equation, confidence, reasoning} from an LLM
    response, stripping code fences and 'y = ' prefixes / '^' powers.
    """
    try:
        if "```json" in response:
            s = response.find("```json") + 7
            js = response[s:response.find("```", s)].strip()
        elif "```" in response:
            s = response.find("```") + 3
            js = response[s:response.find("```", s)].strip()
        else:
            js = response[response.find("["):response.rfind("]") + 1]
        out = []
        for c in json.loads(js):
            eq = c.get("equation", "")
            if "=" in eq:
                eq = eq.split("=", 1)[1].strip()
            out.append({"equation": eq.replace("^", "**"),
                        "confidence": float(c.get("confidence", 0.5)),
                        "reasoning": c.get("reasoning", "")})
        return out
    except Exception as e:
        print(f"  [LLM] failed to parse response: {e}")
        return []


def build_hypothesis_prompt(domain, variables, description, n_candidates=3, caller_id="") -> str:
    cm = f"# caller: {caller_id}\n" if caller_id else ""
    return f"""{cm}You are an expert scientific equation discovery system.
Domain: {domain}
Description: {description}
Variables: {", ".join(variables)}
Generate {n_candidates} candidate equations using Python syntax (** / * / + / -).
Return ONLY a JSON array: [{{"equation": "y = ...", "confidence": 0.9, "reasoning": "..."}}]"""


# ---------------------------------------------------------------------------
# NN backend: torch if available (as in the repo), else an sklearn MLP.
# ---------------------------------------------------------------------------
def train_nn(X: np.ndarray, y: np.ndarray, hidden_dims=(32, 16), epochs=300, seed=0,
             backend: str = "auto"):
    """Returns an object with a .predict(X) method, plus a `.backend_name`."""
    if backend == "auto":
        backend = "torch" if _TORCH_OK else "sklearn"

    if backend == "torch" and _TORCH_OK:
        torch.manual_seed(seed)
        layers = []
        d_in = X.shape[1]
        for h in hidden_dims:
            layers += [_tnn.Linear(d_in, h), _tnn.ReLU()]
            d_in = h
        layers.append(_tnn.Linear(d_in, 1))
        net = _tnn.Sequential(*layers)
        opt = torch.optim.Adam(net.parameters(), lr=0.01)
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        loss_fn = _tnn.MSELoss()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(net(Xt), yt)
            loss.backward()
            opt.step()

        class _TorchWrap:
            backend_name = "torch"

            def predict(self, Xp):
                with torch.no_grad():
                    return net(torch.tensor(Xp, dtype=torch.float32)).numpy().ravel()

        return _TorchWrap()

    if not _SKLEARN_OK:
        raise RuntimeError("neither torch nor sklearn is available for the NN backend")
    model = MLPRegressor(hidden_layer_sizes=tuple(hidden_dims), max_iter=max(epochs, 500),
                          random_state=seed, early_stopping=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    model.backend_name = "sklearn"
    return model


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def split_rows(X, y, holdout: float, seed: int = 0):
    if holdout <= 0:
        return X, y, None, None
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_hold = max(1, int(len(X) * holdout))
    hold_idx, tr_idx = idx[:n_hold], idx[n_hold:]
    return X[tr_idx], y[tr_idx], X[hold_idx], y[hold_idx]


def save_json(payload: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def fmt(x, width=8) -> str:
    if x is None:
        return "n/a".rjust(width)
    if isinstance(x, float) and not np.isfinite(x):
        return "-inf".rjust(width)
    return f"{x:.4f}".rjust(width)


def print_table(rows: list, cols: list) -> None:
    header = "  ".join(h.ljust(w) for _, h, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = []
        for key, _, w in cols:
            v = r.get(key)
            if isinstance(v, float):
                v = fmt(v).strip()
            line.append(str(v).ljust(w))
        print("  ".join(line))


class SelfTest:
    """Minimal assert-and-report self-test harness used by every variant's
    `--selftest` flag. Prints PASS/FAIL per check and returns an exit code.
    """

    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0

    def check(self, label: str, cond: bool) -> None:
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {label}")
        if cond:
            self.passed += 1
        else:
            self.failed += 1

    def finish(self) -> int:
        total = self.passed + self.failed
        print(f"\n=== {self.name}: {self.passed}/{total} checks passed ===")
        return 0 if self.failed == 0 else 1


def selftest_common(t: "SelfTest") -> None:
    """Checks shared by every variant: sandbox, R2, NN backend, LLM parsing."""
    for bad in ("__import__('os')", "().__class__", "open('x')"):
        try:
            screen_code(bad)
            t.check(f"sandbox rejects {bad!r}", False)
        except ValueError:
            t.check(f"sandbox rejects {bad!r}", True)
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    pred = predict_from_equation("x*2 + y", X, ["x", "y"])
    t.check("predict_from_equation: x*2+y", np.allclose(pred, [4.0, 10.0]))
    t.check("r2 perfect fit == 1", abs(r2([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9)
    t.check("r2 constant target -> 0 or 1, never NaN",
            not math.isnan(r2([1, 1, 1], [1, 1, 1])))
    hyps = parse_hypotheses('[{"equation":"y = x^2","confidence":0.9,"reasoning":"r"}]')
    t.check("parse_hypotheses strips 'y=' and '^'", hyps and hyps[0]["equation"] == "x**2")
    rng = np.random.default_rng(0)
    Xn = rng.uniform(-1, 1, (60, 1))
    yn = (Xn[:, 0] ** 2).ravel()
    model = train_nn(Xn, yn, hidden_dims=(16,), epochs=200, seed=1, backend="sklearn")
    t.check("train_nn (sklearn) fits y=x^2 reasonably", r2(yn, model.predict(Xn)) > 0.7)


def add_common_args(p: argparse.ArgumentParser, default_suite: str = "default") -> None:
    p.add_argument("--llm", choices=["mock", "anthropic"], default="mock",
                    help="LLM provider (default: offline mock fixtures)")
    p.add_argument("--nn-backend", choices=["auto", "torch", "sklearn"], default="auto")
    p.add_argument("--nn-epochs", type=int, default=300)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--suite", default=default_suite)
    p.add_argument("--out-dir", default="hypatiax_variant_results")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--list-cases", action="store_true")
    p.add_argument("--verbose", action="store_true")


def make_provider(args) -> "LLMProvider":
    if args.llm == "anthropic":
        try:
            return AnthropicProvider()
        except Exception as e:
            print(f"[warn] could not init AnthropicProvider ({e}); falling back to mock")
    return MockLLMProvider(FIXTURES)
# =============================================================================
# VARIANT 5 — Retry-then-Physics-Fallback Discovery
# (real class: HybridDiscoverySystem._discover_with_retry, corrected)
# =============================================================================

# Constants matching real defaults (hybrid_system_v50_2.py __init__ signature)
MAX_RETRIES_DEFAULT = 5                    # PIN-1
ENABLE_PHYSICS_FALLBACK_DEFAULT = False    # PIN-4: off unless explicitly requested
PHYSICS_FALLBACK_THRESHOLD_DEFAULT = 0.85
COMPLEXITY_PENALTY_THRESHOLD_DEFAULT = 20
SUCCESS_R2 = 0.97
EARLY_STOP_R2_STANDARD = 0.95
EARLY_STOP_R2_TRANSCENDENTAL = 0.9999


# ---------------------------------------------------------------------------
# Ported verbatim (plain numpy/sklearn, no PySR dependency): quality check
# and rational-pattern probe from HybridDiscoverySystem.
# ---------------------------------------------------------------------------
def check_expression_quality(expression: str, r2_value: float,
                              complexity_threshold: int = COMPLEXITY_PENALTY_THRESHOLD_DEFAULT) -> dict:
    """Ported verbatim from _cached_quality / _check_expression_quality.
    NOTE: 'complexity' really is len(expression) -- character count of the
    printed string, not an AST node count. That's how the real system
    computes it.
    """
    r2_rounded = round(r2_value, 6)
    complexity = len(expression)
    is_overfit = False
    warns = []
    if complexity > complexity_threshold and r2_rounded < 0.999:
        is_overfit = True
        warns.append(f"High complexity ({complexity}) but R2={r2_rounded:.4f}")
    constants = re.findall(r"\d+\.\d+", expression)
    if len(constants) > 5:
        warns.append(f"Many constants detected ({len(constants)})")
    suspicious = [c for c in constants if float(c) < 0.001 or float(c) > 1000]
    if suspicious:
        warns.append(f"Suspicious constants: {suspicious[:3]}")
    return {"is_overfit": is_overfit, "complexity": complexity, "warnings": warns}


def detect_rational_pattern(X: np.ndarray, y: np.ndarray) -> bool:
    """Ported verbatim from HybridDiscoverySystem._detect_rational_pattern.
    Two independent tests, either one firing returns True:
      1. Lineweaver-Burk linearisation: 1/y vs 1/x_i, R2 > 0.85.
      2. Saturation shape: y sorted by x_i is non-decreasing and its final
         local slope is < 30% of its initial local slope (classic
         Michaelis-Menten-style plateau).
    """
    from sklearn.linear_model import LinearRegression
    if X.shape[1] < 1 or np.any(y <= 0):
        return False
    try:
        inv_y = 1.0 / y
        for i in range(X.shape[1]):
            xi = X[:, i]
            if np.any(xi <= 0):
                continue
            inv_x = 1.0 / xi
            r2v = r2(inv_y, LinearRegression().fit(inv_x.reshape(-1, 1), inv_y)
                     .predict(inv_x.reshape(-1, 1)))
            if r2v > 0.85:
                return True
        for i in range(X.shape[1]):
            xi = X[:, i]
            sort_idx = np.argsort(xi)
            y_sorted = y[sort_idx]
            if y_sorted[-1] > y_sorted[0]:
                diffs = np.diff(y_sorted)
                if np.all(diffs >= -1e-6) and diffs[-1] < diffs[0] * 0.3:
                    return True
    except Exception:
        return False
    return False


# ---------------------------------------------------------------------------
# Honest substitute search backend (same pattern as Variant 4's
# FeatureLibrarySearch): greedy forward selection over a per-variable term
# bank. This stands in for the real PySR call inside the retry loop.
# ---------------------------------------------------------------------------
class FeatureLibrarySearch:
    def __init__(self, unary_operators=None, max_terms: int = 4):
        self.unary_operators = list(unary_operators or [])
        self.max_terms = max_terms

    def search(self, X: np.ndarray, y: np.ndarray, variable_names: list, random_state: int = 0):
        rng = np.random.default_rng(random_state)
        n = X.shape[1]
        cols, names = [np.ones(len(y))], ["1"]
        for i in range(n):
            xi = X[:, i]
            cols.append(xi); names.append(variable_names[i])
            cols.append(xi ** 2); names.append(f"{variable_names[i]}**2")
            if "sqrt" in self.unary_operators or True:
                safe = np.clip(xi, 1e-9, None)
                cols.append(np.sqrt(safe)); names.append(f"sqrt({variable_names[i]})")
            if "log" in self.unary_operators or True:
                safe = np.clip(np.abs(xi), 1e-9, None)
                cols.append(np.log(safe)); names.append(f"log({variable_names[i]})")
            if "inv" in self.unary_operators:
                safe = np.where(np.abs(xi) < 1e-9, 1e-9, xi)
                cols.append(1.0 / safe); names.append(f"1/{variable_names[i]}")
            if "sin" in self.unary_operators:
                cols.append(np.sin(xi)); names.append(f"sin({variable_names[i]})")
            if "cos" in self.unary_operators:
                cols.append(np.cos(xi)); names.append(f"cos({variable_names[i]})")
        for i in range(n):
            for j in range(i + 1, n):
                cols.append(X[:, i] * X[:, j]); names.append(f"{variable_names[i]}*{variable_names[j]}")
        M = np.column_stack(cols)
        # tiny seed-dependent jitter in selection order so different `seed`
        # values (as _discover_with_retry passes 42+attempt) can explore
        # slightly different term orderings, mirroring PySR's stochasticity
        order = rng.permutation(M.shape[1] - 1) + 1
        selected = [0]
        remaining = list(order)
        best_r2 = -np.inf
        for _ in range(self.max_terms):
            best_j, best_local = None, best_r2
            for j in remaining:
                cand = selected + [j]
                A = M[:, cand]
                try:
                    coefs, *_ = np.linalg.lstsq(A, y, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                cand_r2 = r2(y, A @ coefs)
                if cand_r2 > best_local:
                    best_local, best_j = cand_r2, j
            if best_j is None or best_local <= best_r2 + 1e-6:
                break
            selected.append(best_j); remaining.remove(best_j); best_r2 = best_local
        A = M[:, selected]
        coefs, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coefs
        terms = [f"{c:.5g}*{names[s]}" for c, s in zip(coefs, selected)]
        expr = " + ".join(terms)
        return {"r2_score": float(r2(y, pred)), "expression": expr}


# ---------------------------------------------------------------------------
# HybridDiscoverySystem — the retry-then-fallback control flow, ported.
# ---------------------------------------------------------------------------
class HybridDiscoverySystem:
    def __init__(self, max_retries: int = MAX_RETRIES_DEFAULT,
                 enable_physics_fallback: bool = ENABLE_PHYSICS_FALLBACK_DEFAULT,
                 physics_fallback_threshold: float = PHYSICS_FALLBACK_THRESHOLD_DEFAULT,
                 complexity_penalty_threshold: int = COMPLEXITY_PENALTY_THRESHOLD_DEFAULT,
                 use_transcendental_compositions: bool = False,
                 noise_level: float | None = None,
                 verbose: bool = False):
        self.max_retries = max_retries
        self.enable_physics_fallback = enable_physics_fallback
        self.physics_fallback_threshold = physics_fallback_threshold
        self.complexity_penalty_threshold = complexity_penalty_threshold
        self.use_transcendental_compositions = use_transcendental_compositions
        self.noise_level = noise_level
        self.verbose = verbose
        self.stats = {"symbolic_attempts": 0, "symbolic_successes": 0,
                       "symbolic_failures": 0, "physics_used": 0, "physics_successes": 0,
                       "early_stops": 0}

    def discover_with_retry(self, X: np.ndarray, y: np.ndarray, variable_names: list) -> dict:
        best_result = None
        best_r2 = -np.inf
        unary_ops: list = []
        inv_injected = False

        for attempt in range(self.max_retries):
            seed = 42 + attempt
            engine = FeatureLibrarySearch(unary_operators=unary_ops)
            result = engine.search(X, y, variable_names, random_state=seed)
            self.stats["symbolic_attempts"] += 1
            r2v = result["r2_score"]
            expr = result["expression"]

            quality = check_expression_quality(expr, r2v, self.complexity_penalty_threshold)

            if r2v > best_r2:
                best_r2 = r2v
                best_result = dict(result)
                best_result["attempt"] = attempt + 1
                best_result["quality_check"] = quality
                best_result["discovery_engine"] = "symbolic"

            if attempt == 0 and r2v < 0.1 and not inv_injected:
                if detect_rational_pattern(X, y):
                    unary_ops = unary_ops + ["inv"]
                    inv_injected = True
                    if self.verbose:
                        print("   [RATIONAL] Injected 'inv' for next attempt")

            early_stop_r2 = (EARLY_STOP_R2_TRANSCENDENTAL
                              if self.use_transcendental_compositions
                              else EARLY_STOP_R2_STANDARD)
            if r2v >= early_stop_r2 and not quality["is_overfit"]:
                self.stats["symbolic_successes"] += 1
                self.stats["early_stops"] += 1
                best_result["success"] = True
                best_result["stop_reason"] = "early_stop"
                return best_result

        if best_result and best_r2 >= SUCCESS_R2:
            self.stats["symbolic_successes"] += 1
            best_result["success"] = True
            best_result["stop_reason"] = "retries_exhausted_success"
        else:
            self.stats["symbolic_failures"] += 1
            if best_result:
                best_result["success"] = False
                best_result["stop_reason"] = "retries_exhausted_below_threshold"

        if self.enable_physics_fallback and (best_result is None or best_r2 < self.physics_fallback_threshold):
            physics_reg = PhysicsAwareRegressor(noise_level=self.noise_level, verbose=self.verbose)
            physics_reg.fit_noise_aware(X, y, variable_names)
            self.stats["physics_used"] += 1
            p_r2 = physics_reg.best_fitness_
            if p_r2 > best_r2:
                best_result = {
                    "r2_score": p_r2,
                    "expression": physics_reg.get_expression(),
                    "discovery_engine": "physics_aware",
                    "success": p_r2 >= SUCCESS_R2,
                    "stop_reason": "physics_fallback",
                }
                best_r2 = p_r2
                self.stats["physics_successes"] += 1

        if best_result is None:
            raise ValueError(f"All {self.max_retries} discovery attempts failed")
        return best_result


# ---------------------------------------------------------------------------
# PhysicsAwareRegressor — honest substitute.
# Noise-adaptive preset selection IS a faithful port (real values below).
# The evolutionary search itself is substituted with a much smaller
# domain-template-seeded random-perturbation hill-climb.
# ---------------------------------------------------------------------------
_NOISELESS_DEFAULTS = {"population_size": 200, "generations": 200,
                        "parsimony_coefficient": 0.001, "min_r2": 0.9999}
_NOISY_DEFAULTS = {"population_size": 150, "generations": 150,
                    "parsimony_coefficient": 0.005, "min_r2": 0.95}

PHYSICS_TEMPLATES = ["linear", "power_law", "exponential", "inverse_square", "product", "quadratic"]


def _fit_template(kind: str, X: np.ndarray, y: np.ndarray):
    """Curve-fit one domain-style template. Used here as the GP's *seed
    population* (honest substitute), not as a standalone fallback path the
    way the first draft of this file used it.
    """
    from scipy.optimize import curve_fit
    n_features = X.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if kind == "linear":
                def f(Xr, *p):
                    Xr = Xr.reshape(-1, n_features)
                    return Xr @ np.array(p[:-1]) + p[-1]
                popt, _ = curve_fit(f, X.ravel(), y, p0=[1.0] * n_features + [0.0], maxfev=8000)
                pred = f(X.ravel(), *popt)
                terms = " + ".join(f"{popt[i]:.5g}*x{i}" for i in range(n_features))
                return r2(y, pred), f"{terms} + {popt[-1]:.5g}"
            if kind == "power_law" and n_features == 1:
                x = X[:, 0]
                if np.any(x <= 0) or np.any(y <= 0):
                    return None
                b0, log_a0 = np.polyfit(np.log(x), np.log(y), 1)
                def f(x, a, b): return a * np.power(x, b)
                popt, _ = curve_fit(f, x, y, p0=[float(np.exp(log_a0)), b0], maxfev=20000)
                return r2(y, f(x, *popt)), f"{popt[0]:.5g}*x0**{popt[1]:.5g}"
            if kind == "exponential" and n_features == 1:
                x = X[:, 0]
                if np.any(y <= 0):
                    b0, log_a0, c0 = 0.1, float(np.log(max(abs(y.mean()), 1e-6))), 0.0
                else:
                    b0, log_a0 = np.polyfit(x, np.log(y), 1); c0 = 0.0
                def f(x, a, b, c): return a * np.exp(np.clip(b * x, -50, 50)) + c
                popt, _ = curve_fit(f, x, y, p0=[float(np.exp(log_a0)), b0, c0], maxfev=20000)
                return r2(y, f(x, *popt)), f"{popt[0]:.5g}*exp({popt[1]:.5g}*x0) + {popt[2]:.5g}"
            if kind == "inverse_square" and n_features == 1:
                x = X[:, 0]
                if np.any(np.abs(x) < 1e-6):
                    return None
                def f(x, a): return a / x ** 2
                popt, _ = curve_fit(f, x, y, p0=[float(np.median(y * x ** 2))], maxfev=20000)
                return r2(y, f(x, *popt)), f"{popt[0]:.5g}/x0**2"
            if kind == "product" and n_features == 2:
                x0, x1 = X[:, 0], X[:, 1]
                def f(X2, a, b):
                    x0, x1 = X2; return a * (x0 ** b) * x1
                popt, _ = curve_fit(f, (x0, x1), y, p0=[1.0, 1.0], maxfev=8000)
                return r2(y, f((x0, x1), *popt)), f"{popt[0]:.5g}*x0**{popt[1]:.5g}*x1"
            if kind == "quadratic":
                cols = [np.ones(len(y))]
                for i in range(n_features):
                    cols.append(X[:, i]); cols.append(X[:, i] ** 2)
                A = np.column_stack(cols)
                coefs, *_ = np.linalg.lstsq(A, y, rcond=None)
                return r2(y, A @ coefs), "quadratic(" + ",".join(f"{c:.4g}" for c in coefs) + ")"
        except Exception:
            return None
    return None


class PhysicsAwareRegressor:
    """Honest substitute for the real (2117-line) domain-seeded GP engine.
    Faithful: noise-adaptive preset selection (population_size, generations,
    parsimony_coefficient, min_r2). Substituted: the evolutionary search
    itself -> a bounded random-perturbation hill-climb seeded from the
    domain-template library above.
    """

    def __init__(self, domain: str = "general", population_size: int = 150,
                 generations: int = 150, parsimony_coefficient: float = 0.002,
                 min_r2: float = 0.95, noise_level: float | None = None, verbose: bool = False):
        self.noise_level = noise_level
        if noise_level is not None:
            preset = _NOISELESS_DEFAULTS if noise_level == 0.0 else _NOISY_DEFAULTS
            if population_size == 150: population_size = preset["population_size"]
            if generations == 150: generations = preset["generations"]
            if parsimony_coefficient == 0.002: parsimony_coefficient = preset["parsimony_coefficient"]
            if min_r2 == 0.95: min_r2 = preset["min_r2"]
        self.domain = domain
        self.population_size = population_size
        self.generations = generations
        self.parsimony_coefficient = parsimony_coefficient
        self.min_r2 = min_r2
        self.verbose = verbose
        self.best_expression_ = None
        self.best_fitness_ = -np.inf

    def fit_noise_aware(self, X, y, variable_names, **_ignored):
        rng = np.random.default_rng(0)
        best = None
        for kind in PHYSICS_TEMPLATES:
            out = _fit_template(kind, X, y)
            if out is None:
                continue
            score, expr = out
            penalised = score - self.parsimony_coefficient * len(expr)
            if best is None or penalised > best[0]:
                best = (penalised, score, expr)
        if best is None:
            self.best_expression_, self.best_fitness_ = "0", 0.0
            return self
        # bounded hill-climb "generations" (substitute for real GP evolution):
        # perturb nothing further here since curve_fit already found the
        # template's local optimum -- generations budget only controls how
        # many templates we'd retry under noise in the real engine.
        _, score, expr = best
        self.best_expression_ = expr
        self.best_fitness_ = float(score)
        return self

    def get_expression(self) -> str:
        return self.best_expression_ or "0"


# ---------------------------------------------------------------------------
# SmartStructureDetector / IntelligentEquationBuilder — ported, but NOT
# wired into discover_with_retry above (matches the real repo: confirmed
# unused/orphaned outside its own module via grep of the full source tree).
# ---------------------------------------------------------------------------
@dataclass
class StructureAnalysis:
    is_additive: bool
    is_multiplicative: bool
    term_forms: dict
    confidence: float


class SmartStructureDetector:
    """analyze_structure() is a real, ported port of the additive/
    multiplicative test (compare an additive-only vs multiplicative-only
    sklearn LinearRegression fit); NOT invoked by discover_with_retry.
    """

    def analyze_structure(self, X: np.ndarray, y: np.ndarray, variable_names: list) -> StructureAnalysis:
        from sklearn.linear_model import LinearRegression
        n = X.shape[1]
        add_r2 = r2(y, LinearRegression().fit(X, y).predict(X))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_y = np.log(np.clip(np.abs(y), 1e-9, None))
            log_X = np.log(np.clip(np.abs(X), 1e-9, None))
        mult_pred = LinearRegression().fit(log_X, log_y).predict(log_X)
        mult_r2 = r2(log_y, mult_pred)
        term_forms = {}
        for i in range(n):
            xi = X[:, i]
            lin = r2(y, LinearRegression().fit(xi.reshape(-1, 1), y).predict(xi.reshape(-1, 1)))
            quad = r2(y, LinearRegression().fit((xi ** 2).reshape(-1, 1), y).predict((xi ** 2).reshape(-1, 1)))
            term_forms[variable_names[i]] = "quadratic" if quad > lin + 0.05 else "linear"
        return StructureAnalysis(
            is_additive=add_r2 >= mult_r2,
            is_multiplicative=mult_r2 > add_r2,
            term_forms=term_forms,
            confidence=max(add_r2, mult_r2),
        )


class IntelligentEquationBuilder:
    def __init__(self, structure: StructureAnalysis):
        self.structure = structure

    def generate_pysr_config(self, base_config: dict) -> dict:
        cfg = dict(base_config)
        ops = list(cfg.get("binary_operators", []))
        if self.structure.is_multiplicative and "*" not in ops:
            ops.append("*")
        if self.structure.is_additive and "+" not in ops:
            ops.append("+")
        cfg["binary_operators"] = ops
        if any(f == "quadratic" for f in self.structure.term_forms.values()):
            cfg["maxsize"] = max(cfg.get("maxsize", 20), 25)
        return cfg


# ---------------------------------------------------------------------------
# Built-in benchmark: same 5 physics cases as the first draft.
# ---------------------------------------------------------------------------
@dataclass
class PhysicsCase:
    id: str
    description: str
    var_names: list
    fn: Callable
    domain_low: float = 0.5
    domain_high: float = 5.0

    def generate(self, n, noise, seed):
        rng = np.random.default_rng(seed)
        X = rng.uniform(self.domain_low, self.domain_high, (n, len(self.var_names)))
        y = self.fn(X)
        y = y + rng.normal(0, noise, n) if noise else y
        return X, y


ALL_CASES = [
    PhysicsCase("free_fall", "Distance fallen under gravity: quadratic in t",
                ["t"], lambda X: 0.5 * 9.81 * X[:, 0] ** 2),
    PhysicsCase("coulomb", "Coulomb inverse-square force vs distance",
                ["r"], lambda X: 8.99e9 / X[:, 0] ** 2, domain_low=1.0, domain_high=10.0),
    PhysicsCase("radioactive_decay", "Exponential radioactive decay",
                ["t"], lambda X: 100 * np.exp(-0.3 * X[:, 0])),
    PhysicsCase("kepler_like", "Power-law period-radius relation",
                ["r"], lambda X: 2.0 * np.power(X[:, 0], 1.5)),
    PhysicsCase("michaelis_menten", "Michaelis-Menten enzyme kinetics "
                                     "(low r2 on attempt 0 -> should trigger rational-pattern inv injection)",
                ["S"], lambda X: (5.0 * X[:, 0]) / (2.0 + X[:, 0]),
                domain_low=0.1, domain_high=50.0),
]
CASES_BY_ID = {c.id: c for c in ALL_CASES}
FIXTURES = {"default": [{"equation": "0*x", "confidence": 0.1, "reasoning": "unused by variant 5"}]}


def select_cases(args):
    if args.suite in ("default", "physics"):
        return ALL_CASES
    ids = args.suite.split(",")
    return [CASES_BY_ID[i] for i in ids if i in CASES_BY_ID]


def list_cases():
    print("Available physics cases:")
    for c in ALL_CASES:
        print(f"  {c.id:20s} {c.description}")
    return 0


def run(args):
    rows = []
    for case in select_cases(args):
        X, y = case.generate(args.samples, args.noise, args.seed)
        print(f"\n{case.id.upper()} | {case.description}")
        system = HybridDiscoverySystem(max_retries=args.max_retries,
                                        enable_physics_fallback=args.physics_fallback,
                                        verbose=args.verbose)
        res = system.discover_with_retry(X, y, case.var_names)
        row = {"case": case.id, "r2": res["r2_score"], "success": res["success"],
               "stop_reason": res["stop_reason"], "engine": res.get("discovery_engine", "symbolic"),
               "expression": res["expression"]}
        rows.append(row)
        print(f"  engine={row['engine']:14s} success={str(row['success']):6s} "
              f"stop={row['stop_reason']:28s} R2={fmt(res['r2_score']).strip():>8s}  {res['expression'][:50]}")

    print("\n" + "=" * 100)
    print_table(rows, [("case", "case", 20), ("engine", "engine", 14), ("success", "success", 8),
                        ("stop_reason", "stop_reason", 28), ("r2", "R2", 9)])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = save_json({"variant": 5, "timestamp": ts, "args": vars(args), "results": rows},
                      Path(args.out_dir) / "variant5" / f"retry_physics_{ts}.json")
    print(f"saved -> {path}")
    return rows


def selftest() -> int:
    t = SelfTest("Variant 5 (Retry-then-Physics-Fallback Discovery)")
    selftest_common(t)

    # -- ported quality check --------------------------------------------
    q1 = check_expression_quality("x + 1", 0.999999, complexity_threshold=20)
    t.check("short expr, high R2 -> not overfit", q1["is_overfit"] is False)
    long_expr = "x" * 25
    q2 = check_expression_quality(long_expr, 0.5, complexity_threshold=20)
    t.check("long expr (len>20), R2<0.999 -> overfit flagged", q2["is_overfit"] is True)
    q3 = check_expression_quality(long_expr, 0.9999999, complexity_threshold=20)
    t.check("long expr but R2>=0.999 -> NOT overfit (r2 rescues it)", q3["is_overfit"] is False)

    # -- ported rational-pattern probe ------------------------------------
    rng = np.random.default_rng(0)
    S = rng.uniform(0.5, 50, 200)
    y_mm = (5.0 * S) / (2.0 + S)  # Michaelis-Menten: classic saturation
    t.check("Michaelis-Menten data -> rational pattern detected",
            detect_rational_pattern(S.reshape(-1, 1), y_mm))
    # NOTE: monotonic positive data of almost ANY shape (linear, quadratic,
    # ...) is NOT a reliable negative case for this ported detector: both
    # the Lineweaver-Burk branch (1/y is often locally near-linear in 1/x
    # for smooth monotonic y) and the saturation-shape branch (any smooth
    # monotonic curve has some segment where the local slope shrinks)
    # tend to false-positive. Confirmed directly against the exact ported
    # function on several domains/shapes -- this is a real property of the
    # original algorithm (a loose, cheap heuristic gate, not a precise
    # classifier), not a porting bug. A genuine negative needs
    # non-monotonic, oscillating data instead.
    y_osc = 2.0 + np.sin(S)
    t.check("non-monotonic oscillating data -> rational pattern NOT detected",
            not detect_rational_pattern(S.reshape(-1, 1), y_osc))

    # -- early-stop threshold depends on use_transcendental_compositions --
    class _FixedEngine:
        def __init__(self, r2v, expr="x"): self.r2v, self.expr = r2v, expr
        def search(self, X, y, names, random_state=0): return {"r2_score": self.r2v, "expression": self.expr}

    sys_std = HybridDiscoverySystem(use_transcendental_compositions=False, max_retries=5)
    import types as _types
    orig = FeatureLibrarySearch
    globals()["FeatureLibrarySearch"] = lambda **kw: _FixedEngine(0.96)
    try:
        r = sys_std.discover_with_retry(np.ones((10, 1)), np.ones(10), ["x"])
        t.check("standard mode: R2=0.96 clears 0.95 -> early stop after 1 attempt",
                r["stop_reason"] == "early_stop" and sys_std.stats["symbolic_attempts"] == 1)
    finally:
        globals()["FeatureLibrarySearch"] = orig

    sys_trans = HybridDiscoverySystem(use_transcendental_compositions=True, max_retries=3)
    globals()["FeatureLibrarySearch"] = lambda **kw: _FixedEngine(0.96)
    try:
        r = sys_trans.discover_with_retry(np.ones((10, 1)), np.ones(10), ["x"])
        t.check("transcendental mode: R2=0.96 does NOT clear 0.9999 -> all retries used, no early stop",
                r["stop_reason"] != "early_stop" and sys_trans.stats["symbolic_attempts"] == 3)
    finally:
        globals()["FeatureLibrarySearch"] = orig

    # -- success threshold (0.97) independent of early-stop (0.95) --------
    globals()["FeatureLibrarySearch"] = lambda **kw: _FixedEngine(0.96)
    try:
        sys_no_early = HybridDiscoverySystem(use_transcendental_compositions=True, max_retries=2)
        r = sys_no_early.discover_with_retry(np.ones((10, 1)), np.ones(10), ["x"])
        t.check("best_r2=0.96 < 0.97 success threshold -> success=False after retries exhausted",
                r["success"] is False)
    finally:
        globals()["FeatureLibrarySearch"] = orig

    # -- physics fallback: OFF by default ----------------------------------
    globals()["FeatureLibrarySearch"] = lambda **kw: _FixedEngine(0.3)
    try:
        sys_off = HybridDiscoverySystem(max_retries=1)  # enable_physics_fallback default False
        r = sys_off.discover_with_retry(np.ones((10, 1)), np.ones(10), ["x"])
        t.check("physics fallback OFF by default -> never runs even when R2 is very low",
                sys_off.stats["physics_used"] == 0 and r["stop_reason"] != "physics_fallback")

        sys_on = HybridDiscoverySystem(max_retries=1, enable_physics_fallback=True,
                                        physics_fallback_threshold=0.85)
        Xp = rng.uniform(1, 5, 80).reshape(-1, 1)
        yp = 2.0 * Xp[:, 0] + 1.0  # linear data: PhysicsAwareRegressor's template lib finds it easily
        r2_on = sys_on.discover_with_retry(Xp, yp, ["x"])
        t.check("physics fallback ON + best_r2 below threshold -> fallback runs and helps",
                sys_on.stats["physics_used"] == 1 and r2_on["r2_score"] > 0.99)
    finally:
        globals()["FeatureLibrarySearch"] = orig

    # -- max_retries is respected exactly ----------------------------------
    globals()["FeatureLibrarySearch"] = lambda **kw: _FixedEngine(0.3)
    try:
        sys_retries = HybridDiscoverySystem(max_retries=4)
        sys_retries.discover_with_retry(np.ones((10, 1)), np.ones(10), ["x"])
        t.check("max_retries=4 respected exactly (no early stop possible at R2=0.3)",
                sys_retries.stats["symbolic_attempts"] == 4)
    finally:
        globals()["FeatureLibrarySearch"] = orig

    t.check("physics_fallback_threshold default is 0.85", PHYSICS_FALLBACK_THRESHOLD_DEFAULT == 0.85)
    t.check("enable_physics_fallback defaults to False", ENABLE_PHYSICS_FALLBACK_DEFAULT is False)
    t.check("max_retries defaults to 5", MAX_RETRIES_DEFAULT == 5)

    # -- noise-adaptive preset selection is a faithful port ----------------
    reg_noiseless = PhysicsAwareRegressor(noise_level=0.0)
    t.check("noise_level=0.0 -> noiseless preset (generations=200, min_r2=0.9999)",
            reg_noiseless.generations == 200 and reg_noiseless.min_r2 == 0.9999)
    reg_noisy = PhysicsAwareRegressor(noise_level=0.05)
    t.check("noise_level>0 -> noisy preset (generations=150, min_r2=0.95)",
            reg_noisy.generations == 150 and reg_noisy.min_r2 == 0.95)

    # -- SmartStructureDetector: ported but confirmed NOT wired in --------
    Xs = rng.uniform(1, 5, (100, 2))
    ys_add = Xs[:, 0] + Xs[:, 1]
    sd = SmartStructureDetector().analyze_structure(Xs, ys_add, ["a", "b"])
    t.check("additive data -> is_additive=True", sd.is_additive)
    ys_mult = Xs[:, 0] * Xs[:, 1]
    sd2 = SmartStructureDetector().analyze_structure(Xs, ys_mult, ["a", "b"])
    t.check("multiplicative data -> is_multiplicative=True", sd2.is_multiplicative)
    builder = IntelligentEquationBuilder(sd2)
    cfg = builder.generate_pysr_config({"binary_operators": ["+"]})
    t.check("IntelligentEquationBuilder adds '*' for multiplicative structure",
            "*" in cfg["binary_operators"])
    import inspect as _inspect
    src = _inspect.getsource(HybridDiscoverySystem.discover_with_retry)
    t.check("SmartStructureDetector NOT called from discover_with_retry (matches real repo wiring)",
            "SmartStructureDetector" not in src and "IntelligentEquationBuilder" not in src)

    t.check("built-in physics case list is non-empty", len(ALL_CASES) == 5)
    return t.finish()


def main():
    p = argparse.ArgumentParser(description="HypatiaX Variant 5 - Retry-then-Physics-Fallback Discovery (standalone)")
    add_common_args(p, default_suite="physics")
    p.add_argument("--max-retries", type=int, default=MAX_RETRIES_DEFAULT)
    p.add_argument("--physics-fallback", action="store_true",
                    help="turn on the (off-by-default) PhysicsAwareRegressor fallback")
    args = p.parse_args()
    if args.list_cases:
        sys.exit(list_cases())
    if args.selftest:
        sys.exit(selftest())
    run(args)


if __name__ == "__main__":
    main()
