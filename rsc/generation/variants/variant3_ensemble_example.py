#!/usr/bin/env python3
"""
HypatiaX - Variant 3: Ensemble Example (standalone)
======================================================
Function ported: ensemble_llm_nn()
Combination strategy: inverse-residual-std weighted blend.

Algorithm
---------
1. Ask the LLM for a candidate formula, score its R2 on training data.
2. Train an NN, score its R2.
3. ensemble_llm_nn(): blend = w_llm*LLM_pred + w_nn*NN_pred where
   w_llm = (1/std(resid_llm)) / (1/std(resid_llm) + 1/std(resid_nn)) -- the
   predictor with the smaller (more consistent) training residual spread
   gets the larger weight.
4. Offline fallback: if the weighting step raises (--force-fallback
   simulates this; the documented real-world trigger is an import path that
   doesn't resolve at runtime), the ensemble falls back to whichever of
   LLM/NN has the higher training R2 -- i.e. max(LLM, NN) on R2, not a
   blend at all.

Built-in benchmark: 3 classic-physics cases (projectile range, resistive
power dissipation, small-angle pendulum period).

Status: standalone re-implementation of the documented combination
strategy and its documented fallback behaviour (the original function
lives in a private repo module not available to grep directly). Runs with
only numpy/scikit-learn -- no hypatiax package import.

Usage
  python variant3_ensemble_example.py --selftest
  python variant3_ensemble_example.py                     # all 3 cases, mock LLM
  python variant3_ensemble_example.py --force-fallback     # exercise the max(LLM,NN) path
  python variant3_ensemble_example.py --llm anthropic      # needs ANTHROPIC_API_KEY

As a library:  from variant3_ensemble_example import ensemble_llm_nn
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
# VARIANT 3 — Ensemble Example  (function: ensemble_llm_nn())
# Combination strategy: inverse-residual-std weighted blend, with an
# offline fallback of max(LLM, NN) on test R2 when the weighting can't be
# computed (mirrors the bare `except` -> max(...) fallback documented for
# this variant).
# =============================================================================
@dataclass
class EnsembleCase:
    id: str
    description: str
    var_names: list

    def generate(self, n, noise, seed):
        raise NotImplementedError


class Projectile(EnsembleCase):
    def __init__(self):
        super().__init__("projectile_range", "Horizontal range of a projectile "
                                              "launched at angle theta", ["v", "theta"])

    def generate(self, n, noise, seed):
        rng = np.random.default_rng(seed)
        v = rng.uniform(5, 50, n)
        theta = rng.uniform(0.1, 1.4, n)
        g = 9.81
        y = v ** 2 * np.sin(2 * theta) / g
        y = y + rng.normal(0, noise, n) if noise else y
        return np.column_stack([v, theta]), y


class OhmPower(EnsembleCase):
    def __init__(self):
        super().__init__("ohm_power", "Electrical power dissipated in a resistor",
                          ["I", "R"])

    def generate(self, n, noise, seed):
        rng = np.random.default_rng(seed)
        I = rng.uniform(0.1, 5, n)
        R = rng.uniform(1, 100, n)
        y = I ** 2 * R
        y = y + rng.normal(0, noise, n) if noise else y
        return np.column_stack([I, R]), y


class Pendulum(EnsembleCase):
    def __init__(self):
        super().__init__("pendulum_period", "Small-angle pendulum period", ["L"])

    def generate(self, n, noise, seed):
        rng = np.random.default_rng(seed)
        L = rng.uniform(0.1, 5, n)
        g = 9.81
        y = 2 * np.pi * np.sqrt(L / g)
        y = y + rng.normal(0, noise, n) if noise else y
        return L.reshape(-1, 1), y


ALL_CASES = [Projectile(), OhmPower(), Pendulum()]
CASES_BY_ID = {c.id: c for c in ALL_CASES}

FIXTURES = {
    "projectile_range": [{"equation": "v**2*sin(2*theta)/9.81", "confidence": 0.9, "reasoning": "exact"}],
    "ohm_power": [{"equation": "I*R", "confidence": 0.4, "reasoning": "missing square on I"}],
    "pendulum_period": [{"equation": "2*3.14159*sqrt(L/9.81)", "confidence": 0.85, "reasoning": "exact"}],
    "default": [{"equation": "0*x", "confidence": 0.1, "reasoning": "no fixture"}],
}


def select_cases(args):
    if args.suite in ("default", "ensemble"):
        return ALL_CASES
    ids = args.suite.split(",")
    return [CASES_BY_ID[i] for i in ids if i in CASES_BY_ID]


def make_dataset(case, n, noise, seed):
    return case.generate(n, noise, seed)


def list_cases():
    print("Available ensemble cases:")
    for c in ALL_CASES:
        print(f"  {c.id:20s} {c.description}")
    return 0


def ensemble_llm_nn(llm_pred_tr, nn_pred_tr, ytr, llm_r2, nn_r2, force_fallback=False):
    """Combination strategy: inverse-residual-std weighted blend.

    weight_llm = (1/std(resid_llm)) / (1/std(resid_llm) + 1/std(resid_nn))
    weight_nn  = 1 - weight_llm
    blend = weight_llm * llm_pred + weight_nn * nn_pred

    Offline fallback (bare `except`, or --force-fallback for this demo):
    falls back to whichever raw predictor has the higher training R2
    (max(LLM, NN) on R2), matching the documented behaviour that the
    ensemble degrades to a simple argmax when the weighting can't be
    computed (e.g. an import path that doesn't resolve at runtime).
    """
    try:
        if force_fallback:
            raise RuntimeError("forced fallback path (simulates the import failure)")
        resid_llm = ytr - llm_pred_tr
        resid_nn = ytr - nn_pred_tr
        std_llm = float(np.std(resid_llm)) + 1e-9
        std_nn = float(np.std(resid_nn)) + 1e-9
        w_llm = (1 / std_llm) / (1 / std_llm + 1 / std_nn)
        w_nn = 1 - w_llm
        blend_tr = w_llm * llm_pred_tr + w_nn * nn_pred_tr
        return blend_tr, {"mode": "weighted_blend", "w_llm": w_llm, "w_nn": w_nn}
    except Exception:
        if llm_r2 >= nn_r2:
            return llm_pred_tr, {"mode": "fallback_max_llm", "w_llm": 1.0, "w_nn": 0.0}
        return nn_pred_tr, {"mode": "fallback_max_nn", "w_llm": 0.0, "w_nn": 1.0}


class EnsembleSystem:
    def __init__(self, provider: LLMProvider, nn_backend: str = "auto", nn_epochs: int = 300,
                 force_fallback: bool = False, verbose: bool = False):
        self.provider = provider
        self.nn_backend = nn_backend
        self.nn_epochs = nn_epochs
        self.force_fallback = force_fallback
        self.verbose = verbose

    def fit_predict(self, Xtr, ytr, names, description, case_id, seed):
        prompt = build_hypothesis_prompt("ensemble", names, description, n_candidates=1,
                                          caller_id="ensemble_llm_nn")
        hyps = parse_hypotheses(self.provider.complete(prompt, case_id=case_id, max_tokens=500))
        llm_eq = hyps[0]["equation"] if hyps else "0*" + names[0]
        try:
            llm_pred = predict_from_equation(llm_eq, Xtr, names)
            llm_r2 = r2(ytr, llm_pred)
        except ValueError:
            llm_pred, llm_r2 = np.zeros(len(ytr)), float("-inf")

        nn_model = train_nn(Xtr, ytr, hidden_dims=(32, 16), epochs=self.nn_epochs,
                             seed=seed, backend=self.nn_backend)
        nn_pred = nn_model.predict(Xtr)
        nn_r2 = r2(ytr, nn_pred)

        blend_tr, info = ensemble_llm_nn(llm_pred, nn_pred, ytr, llm_r2, nn_r2,
                                          force_fallback=self.force_fallback)
        blend_r2 = r2(ytr, blend_tr)

        def predict(Xany):
            if info["mode"] == "fallback_max_llm":
                return predict_from_equation(llm_eq, Xany, names)
            if info["mode"] == "fallback_max_nn":
                return nn_model.predict(Xany)
            p_llm = predict_from_equation(llm_eq, Xany, names)
            p_nn = nn_model.predict(Xany)
            return info["w_llm"] * p_llm + info["w_nn"] * p_nn

        return {"case": case_id, "llm_r2": llm_r2, "nn_r2": nn_r2, "blend_r2": blend_r2,
                "mode": info["mode"], "w_llm": info["w_llm"], "w_nn": info["w_nn"],
                "llm_equation": llm_eq, "predict": predict,
                "nn_backend": getattr(nn_model, "backend_name", "?")}


def run(args):
    provider = make_provider(args)
    system = EnsembleSystem(provider, args.nn_backend, args.nn_epochs,
                             force_fallback=args.force_fallback, verbose=args.verbose)
    rows = []
    for case in select_cases(args):
        X, y = make_dataset(case, args.samples, args.noise, args.seed)
        Xtr, ytr, Xte, yte = split_rows(X, y, args.holdout, args.seed)
        print(f"\n{case.id.upper()} | {case.description[:65]}")
        res = system.fit_predict(Xtr, ytr, case.var_names, case.description, case.id, args.seed)
        hold_r2 = None
        if Xte is not None:
            try:
                hold_r2 = r2(yte, res["predict"](Xte))
            except Exception:
                hold_r2 = None
        row = {"case": case.id, "llm_r2": res["llm_r2"], "nn_r2": res["nn_r2"],
               "blend_r2": res["blend_r2"], "mode": res["mode"],
               "w_llm": round(res["w_llm"], 3), "holdout_r2": hold_r2}
        rows.append(row)
        print(f"  mode={res['mode']:18s} w_llm={res['w_llm']:.3f} "
              f"LLM R2={fmt(res['llm_r2']).strip():>8s} NN R2={fmt(res['nn_r2']).strip():>8s} "
              f"blend R2={fmt(res['blend_r2']).strip():>8s}")

    print("\n" + "=" * 100)
    print_table(rows, [("case", "case", 20), ("mode", "mode", 18), ("w_llm", "w_llm", 8),
                        ("llm_r2", "LLM R2", 9), ("nn_r2", "NN R2", 9),
                        ("blend_r2", "blend R2", 9), ("holdout_r2", "holdout R2", 10)])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = save_json({"variant": 3, "timestamp": ts, "llm": provider.name, "args": vars(args),
                       "results": rows}, Path(args.out_dir) / "variant3" / f"ensemble_{ts}.json")
    print(f"saved -> {path}")
    return rows


def selftest() -> int:
    t = SelfTest("Variant 3 (Ensemble Example)")
    selftest_common(t)

    ytr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    llm_pred = ytr + np.array([0.01, -0.01, 0.02, -0.02, 0.01])   # very close
    nn_pred = ytr + np.array([1.0, -1.0, 1.5, -1.5, 1.0])          # noisy
    blend, info = ensemble_llm_nn(llm_pred, nn_pred, ytr, r2(ytr, llm_pred), r2(ytr, nn_pred))
    t.check("weighted blend: more-accurate predictor gets more weight",
            info["w_llm"] > info["w_nn"])
    t.check("weighted blend R2 close to the better predictor",
            r2(ytr, blend) > 0.9)

    blend2, info2 = ensemble_llm_nn(llm_pred, nn_pred, ytr, r2(ytr, llm_pred), r2(ytr, nn_pred),
                                     force_fallback=True)
    t.check("forced fallback picks argmax(R2) predictor", info2["mode"] == "fallback_max_llm")
    t.check("fallback blend equals the chosen predictor exactly",
            np.allclose(blend2, llm_pred))

    nn_pred_better = ytr + np.array([-0.5, 0.5, -0.3, 0.3, -0.2])
    _, info3 = ensemble_llm_nn(nn_pred_better, llm_pred, ytr, r2(ytr, nn_pred_better),
                                r2(ytr, llm_pred), force_fallback=True)
    t.check("forced fallback correctly swaps to the other side when it's better",
            info3["mode"] == "fallback_max_nn")

    t.check("weights always sum to 1",
            abs(info["w_llm"] + info["w_nn"] - 1.0) < 1e-9)
    t.check("built-in ensemble case list is non-empty", len(ALL_CASES) == 3)
    return t.finish()


def main():
    p = argparse.ArgumentParser(description="HypatiaX Variant 3 - Ensemble Example (standalone)")
    add_common_args(p, default_suite="ensemble")
    p.add_argument("--force-fallback", action="store_true", dest="force_fallback",
                    help="simulate the offline import-failure path -> max(LLM,NN) fallback")
    args = p.parse_args()
    if args.list_cases:
        sys.exit(list_cases())
    if args.selftest:
        sys.exit(selftest())
    run(args)


if __name__ == "__main__":
    main()
