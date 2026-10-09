#!/usr/bin/env python3
"""
HypatiaX - Variant 5: Template Physics Fallback (standalone)
================================================================
Classes ported: PhysicsAwareRegressor / SmartStructureDetector
Combination strategy: template-library fit, with a template-free fallback.

Algorithm
---------
1. PhysicsAwareRegressor tries every template in a small physics-style
   library (linear, power law, exponential, inverse-square, a two-variable
   product form, quadratic) via scipy.optimize.curve_fit, and keeps the
   best-scoring one -- but only if its R2 clears --threshold (default 0.90).
2. If nothing in the library clears the threshold, control falls through to
   SmartStructureDetector: a template-free greedy least-squares search over
   a generic polynomial/interaction feature bank (this is the "structure
   fit" side of the documented combination strategy -- no named physics
   form is assumed, whatever a handful of low-order terms can explain is
   accepted).

Built-in benchmark: 5 physics-flavoured cases, most matching one of the
named templates exactly, one (Michaelis-Menten enzyme kinetics) that
deliberately matches none of them, to exercise the template_free fallback
path for real rather than only in theory.

Status: standalone re-implementation of the documented template-then-fallback
strategy (the original classes live in private repo modules not available to
grep directly, and the repo notes this fallback is off by default / rarely
exercised there). Runs with only numpy/scipy -- no hypatiax package import,
no LLM involved (this variant is purely a curve-fitting strategy).

Usage
  python variant5_template_physics_fallback.py --selftest
  python variant5_template_physics_fallback.py                    # all 5 cases
  python variant5_template_physics_fallback.py --threshold 0.995   # force more fallbacks
  python variant5_template_physics_fallback.py --list-cases

As a library:
  from variant5_template_physics_fallback import template_physics_fallback
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
# END SHARED CORE
# =============================================================================

# =============================================================================
# VARIANT 5 — Template Physics Fallback
# (classes: PhysicsAwareRegressor / SmartStructureDetector)
# Combination strategy: template-library fit, falling back to a
# template-free generic-structure fit when no template clears the threshold.
# =============================================================================
TEMPLATE_R2_THRESHOLD = 0.90


def _fit_template(kind: str, X: np.ndarray, y: np.ndarray):
    """Fit one physics-style template to (X, y) via scipy curve_fit.
    Returns (r2, expr_string, params) or None if the template doesn't apply
    (wrong arity, or the fit fails outright).
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
                p0 = [1.0] * n_features + [0.0]
                popt, _ = curve_fit(f, X.ravel(), y, p0=p0, maxfev=8000)
                pred = f(X.ravel(), *popt)
                terms = " + ".join(f"{popt[i]:.5g}*x{i}" for i in range(n_features))
                return r2(y, pred), f"{terms} + {popt[-1]:.5g}", list(popt)

            if kind == "power_law" and n_features == 1:
                x = X[:, 0]
                if np.any(x <= 0) or np.any(y <= 0):
                    return None
                # data-driven start: fit a line in log-log space first
                b0, log_a0 = np.polyfit(np.log(x), np.log(y), 1)
                a0 = float(np.exp(log_a0))

                def f(x, a, b):
                    return a * np.power(x, b)
                popt, _ = curve_fit(f, x, y, p0=[a0, b0], maxfev=20000)
                pred = f(x, *popt)
                return r2(y, pred), f"{popt[0]:.5g}*x0**{popt[1]:.5g}", list(popt)

            if kind == "exponential" and n_features == 1:
                x = X[:, 0]
                if np.any(y <= 0):
                    b0, log_a0, c0 = 0.1, float(np.log(max(abs(y.mean()), 1e-6))), 0.0
                else:
                    b0, log_a0 = np.polyfit(x, np.log(y), 1)
                    c0 = 0.0
                a0 = float(np.exp(log_a0))

                def f(x, a, b, c):
                    return a * np.exp(np.clip(b * x, -50, 50)) + c
                popt, _ = curve_fit(f, x, y, p0=[a0, b0, c0], maxfev=20000)
                pred = f(x, *popt)
                return r2(y, pred), f"{popt[0]:.5g}*exp({popt[1]:.5g}*x0) + {popt[2]:.5g}", list(popt)

            if kind == "inverse_square" and n_features == 1:
                x = X[:, 0]
                if np.any(np.abs(x) < 1e-6):
                    return None
                a0 = float(np.median(y * x ** 2))

                def f(x, a):
                    return a / x ** 2
                popt, _ = curve_fit(f, x, y, p0=[a0], maxfev=20000)
                pred = f(x, *popt)
                return r2(y, pred), f"{popt[0]:.5g}/x0**2", list(popt)

            if kind == "product" and n_features == 2:
                x0, x1 = X[:, 0], X[:, 1]

                def f(X2, a, b):
                    x0, x1 = X2
                    return a * (x0 ** b) * x1
                popt, _ = curve_fit(f, (x0, x1), y, p0=[1.0, 1.0], maxfev=8000)
                pred = f((x0, x1), *popt)
                return r2(y, pred), f"{popt[0]:.5g}*x0**{popt[1]:.5g}*x1", list(popt)

            if kind == "quadratic":
                cols = [np.ones(len(y))]
                for i in range(n_features):
                    cols.append(X[:, i])
                    cols.append(X[:, i] ** 2)
                A = np.column_stack(cols)
                coefs, *_ = np.linalg.lstsq(A, y, rcond=None)
                pred = A @ coefs
                return r2(y, pred), "quadratic(" + ",".join(f"{c:.4g}" for c in coefs) + ")", list(coefs)
        except Exception:
            return None
    return None


TEMPLATE_LIBRARY = ["linear", "power_law", "exponential", "inverse_square", "product", "quadratic"]


class PhysicsAwareRegressor:
    """Fits every template in TEMPLATE_LIBRARY that applies to the data's
    shape and returns the best-scoring one, provided it clears
    TEMPLATE_R2_THRESHOLD. Returns None if nothing clears the bar (the
    caller is then expected to fall through to the template-free detector).
    """

    def __init__(self, threshold: float = TEMPLATE_R2_THRESHOLD, verbose: bool = False):
        self.threshold = threshold
        self.verbose = verbose

    def fit(self, X, y):
        best = None
        for kind in TEMPLATE_LIBRARY:
            out = _fit_template(kind, X, y)
            if out is None:
                continue
            score, expr, params = out
            if self.verbose:
                print(f"    template {kind:15s} R2={score:.4f}  {expr[:50]}")
            if best is None or score > best[0]:
                best = (score, kind, expr, params)
        if best is None or best[0] < self.threshold:
            return None
        score, kind, expr, params = best
        return {"r2_score": score, "template": kind, "expression": expr, "params": params}


class SmartStructureDetector:
    """Template-free fallback: fits a small polynomial feature bank
    (degree-1..3 per variable plus pairwise products) via sparse greedy
    least squares, the same style of search used by Variant 4's backend but
    applied here as a last resort when no named physics template fits.
    """

    def __init__(self, max_terms: int = 5):
        self.max_terms = max_terms

    def fit(self, X, y):
        n_features = X.shape[1]
        cols, names = [np.ones(len(y))], ["1"]
        for i in range(n_features):
            for p in (1, 2, 3):
                cols.append(X[:, i] ** p)
                names.append(f"x{i}^{p}")
        for i in range(n_features):
            for j in range(i + 1, n_features):
                cols.append(X[:, i] * X[:, j])
                names.append(f"x{i}*x{j}")
        M = np.column_stack(cols)

        selected = [0]  # intercept always included
        remaining = list(range(1, M.shape[1]))
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
            selected.append(best_j)
            remaining.remove(best_j)
            best_r2 = best_local

        A = M[:, selected]
        coefs, *_ = np.linalg.lstsq(A, y, rcond=None)
        terms = [f"{c:.5g}*{names[s]}" for c, s in zip(coefs, selected)]
        expr = " + ".join(terms)
        return {"r2_score": float(r2(y, A @ coefs)), "template": "structure_free",
                "expression": expr, "params": list(coefs), "used_terms": [names[s] for s in selected]}


def template_physics_fallback(X, y, threshold: float = TEMPLATE_R2_THRESHOLD, verbose: bool = False):
    """The combination strategy itself: try the template library first;
    only fall through to the template-free structure detector if nothing
    in the library clears `threshold`.
    """
    reg = PhysicsAwareRegressor(threshold=threshold, verbose=verbose)
    result = reg.fit(X, y)
    if result is not None:
        result["path"] = "template_library"
        return result
    result = SmartStructureDetector().fit(X, y)
    result["path"] = "template_free"
    return result


# ---------------------------------------------------------------------------
# Built-in benchmark: physics relations, some matching a template exactly
# and some deliberately not, to exercise both paths.
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
                                     "(no template covers this exactly)",
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


def make_dataset(case, n, noise, seed):
    return case.generate(n, noise, seed)


def list_cases():
    print("Available physics cases:")
    for c in ALL_CASES:
        print(f"  {c.id:20s} {c.description}")
    return 0


def run(args):
    rows = []
    for case in select_cases(args):
        X, y = make_dataset(case, args.samples, args.noise, args.seed)
        Xtr, ytr, Xte, yte = split_rows(X, y, args.holdout, args.seed)
        print(f"\n{case.id.upper()} | {case.description}")
        res = template_physics_fallback(Xtr, ytr, threshold=args.threshold, verbose=args.verbose)
        hold_r2 = None
        if Xte is not None and res["path"] == "template_library":
            try:
                params = res["params"]
                score2, expr2, _ = _fit_template(res["template"], Xte, yte) or (None, None, None)
                hold_r2 = score2
            except Exception:
                hold_r2 = None
        row = {"case": case.id, "path": res["path"], "template": res["template"],
               "r2": res["r2_score"], "holdout_r2": hold_r2, "expression": res["expression"]}
        rows.append(row)
        print(f"  path={res['path']:16s} template={res['template']:16s} "
              f"R2={fmt(res['r2_score']).strip():>8s}  {res['expression'][:55]}")

    print("\n" + "=" * 100)
    print_table(rows, [("case", "case", 20), ("path", "path", 16), ("template", "template", 16),
                        ("r2", "R2", 9), ("holdout_r2", "holdout", 9)])
    from collections import Counter
    print("paths taken:", dict(Counter(r["path"] for r in rows)))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = save_json({"variant": 5, "timestamp": ts, "args": vars(args), "results": rows},
                      Path(args.out_dir) / "variant5" / f"template_physics_{ts}.json")
    print(f"saved -> {path}")
    return rows


def selftest() -> int:
    t = SelfTest("Variant 5 (Template Physics Fallback)")
    selftest_common(t)

    rng = np.random.default_rng(0)
    x = rng.uniform(1, 8, 100).reshape(-1, 1)
    y = 3.0 * np.exp(0.4 * x[:, 0])
    res = template_physics_fallback(x, y, threshold=0.9)
    t.check("exponential data -> template_library path used", res["path"] == "template_library")
    t.check("exponential data -> 'exponential' template wins", res["template"] == "exponential")
    t.check("exponential template fit is near-perfect", res["r2_score"] > 0.99)

    x2 = rng.uniform(0.1, 50, 150).reshape(-1, 1)
    y2 = (5.0 * x2[:, 0]) / (2.0 + x2[:, 0])   # Michaelis-Menten saturation curve:
                                                # no template in the library covers this shape
    res2 = template_physics_fallback(x2, y2, threshold=0.995)
    t.check("Michaelis-Menten data (no matching template) -> falls through to template_free",
            res2["path"] == "template_free")
    t.check("template-free fallback still gets a usable R2",
            res2["r2_score"] > 0.8)

    x3 = rng.uniform(1, 5, 80).reshape(-1, 1)
    y3 = 2.0 * x3[:, 0] + 1.0
    reg = PhysicsAwareRegressor(threshold=0.9)
    r3 = reg.fit(x3, y3)
    t.check("linear data -> PhysicsAwareRegressor finds it directly",
            r3 is not None and r3["template"] == "linear")

    t.check("PhysicsAwareRegressor returns None below threshold on pure noise",
            PhysicsAwareRegressor(threshold=0.99).fit(
                rng.uniform(0, 1, (50, 1)), rng.normal(0, 1, 50)) is None)

    t.check("built-in physics case list is non-empty", len(ALL_CASES) == 5)
    return t.finish()


def main():
    p = argparse.ArgumentParser(description="HypatiaX Variant 5 - Template Physics Fallback (standalone)")
    add_common_args(p, default_suite="physics")
    p.add_argument("--threshold", type=float, default=TEMPLATE_R2_THRESHOLD,
                    help="R2 a template must clear before it's accepted (else fall through)")
    args = p.parse_args()
    if args.list_cases:
        sys.exit(list_cases())
    if args.selftest:
        sys.exit(selftest())
    run(args)


if __name__ == "__main__":
    main()
