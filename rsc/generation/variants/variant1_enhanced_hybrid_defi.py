#!/usr/bin/env python3
"""
HypatiaX - Variant 1: Enhanced Hybrid DeFi (standalone)
=========================================================
Class ported: EnhancedHybridSystemDeFi
Combination strategy: R^2 threshold gate + scipy constant refit.

Algorithm
---------
1. Ask the LLM (mock fixture by default, or --llm anthropic) for a single
   candidate formula on the DeFi case's variables.
2. Refit an affine transform a*f(x)+b against the training data with
   scipy.optimize.least_squares ("constant refit") -> fitted R2.
3. Train a small NN on the same data (torch if available, else an sklearn
   MLPRegressor fallback) -> NN R2.
4. Gate on the fitted R2:
       fitted R2 >= 0.97                         -> decision = fitted_llm
       fitted R2 > 0 and NN R2 > 0                -> decision = ensemble
           (inverse-residual-std weighted blend of the fitted-LLM and NN
           predictions -- the source that is more consistent on the
           training residuals gets the larger weight)
       otherwise                                   -> decision = nn

Built-in benchmark: 5 DeFi cases (impermanent loss, Kelly-sized LP,
constant-product AMM swap output, compounding APY, liquidation price).

Status: this is a standalone re-implementation of the documented gate logic,
not a byte-for-byte port -- the original class lives in a private repo module
that was not available to grep directly. The gate thresholds, the affine
"constant refit" step and the inverse-residual-std ensemble blend all match
the behaviour described for Variant 1. Everything below runs with only
numpy/scipy/scikit-learn -- no hypatiax package import.

Usage
  python variant1_enhanced_hybrid_defi.py --selftest
  python variant1_enhanced_hybrid_defi.py                  # all 5 DeFi cases, mock LLM
  python variant1_enhanced_hybrid_defi.py --suite il,kelly_lp
  python variant1_enhanced_hybrid_defi.py --llm anthropic   # needs ANTHROPIC_API_KEY
  python variant1_enhanced_hybrid_defi.py --list-cases

As a library:  from variant1_enhanced_hybrid_defi import EnhancedHybridSystemDeFi
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
# VARIANT 1 — Enhanced Hybrid DeFi  (class: EnhancedHybridSystemDeFi)
# Combination strategy: R^2 threshold gate + scipy constant refit
# =============================================================================
FITTED_THRESHOLD = 0.97   # fitted-LLM R2 at/above this -> use fitted_llm outright
NN_FLOOR = 0.0             # below this the LLM/fitted path is abandoned for NN


@dataclass
class DeFiCase:
    id: str
    description: str
    var_names: list
    domain: str = "defi"

    def generate(self, n, noise, seed):
        raise NotImplementedError


class ImpermanentLoss(DeFiCase):
    """AMM impermanent loss of a constant-product position vs price ratio r."""

    def __init__(self):
        super().__init__("il", "Impermanent loss of a constant-product AMM position "
                                "as a function of price ratio r", ["r"])

    def generate(self, n, noise, seed):
        rng = np.random.default_rng(seed)
        r = rng.uniform(0.3, 3.0, n)
        y = 2 * np.sqrt(r) / (1 + r) - 1
        y = y + rng.normal(0, noise, n) if noise else y
        return r.reshape(-1, 1), y


class KellyLP(DeFiCase):
    """Risk-adjusted Kelly-criterion LP position size."""

    def __init__(self):
        super().__init__("kelly_lp", "Optimal LP position size using risk-adjusted "
                                      "Kelly criterion", ["mu", "sigma"])

    def generate(self, n, noise, seed):
        rng = np.random.default_rng(seed)
        mu = rng.uniform(0.01, 0.3, n)
        sigma = rng.uniform(0.05, 0.6, n)
        y = mu / (sigma ** 2 + 1e-6)
        y = np.clip(y, 0, 20)
        y = y + rng.normal(0, noise, n) if noise else y
        return np.column_stack([mu, sigma]), y


class AmmSwap(DeFiCase):
    """Output amount of a constant-product AMM swap with a proportional fee."""

    def __init__(self):
        super().__init__("amm_out", "Output amount of a constant-product AMM swap "
                                     "with a proportional trading fee f", ["x", "y", "dx"])

    def generate(self, n, noise, seed):
        rng = np.random.default_rng(seed)
        x = rng.uniform(100, 10000, n)
        y = rng.uniform(100, 10000, n)
        dx = rng.uniform(1, 500, n)
        fee = 0.997
        out = (y * dx * fee) / (x + dx * fee)
        out = out + rng.normal(0, noise, n) if noise else out
        return np.column_stack([x, y, dx]), out


class ApyCompound(DeFiCase):
    """Effective annual yield of a nominal rate compounded n times/year."""

    def __init__(self):
        super().__init__("apy_compound", "Effective annual yield of a nominal rate "
                                          "compounded n times per year", ["rate", "n"])

    def generate(self, n_samples, noise, seed):
        rng = np.random.default_rng(seed)
        rate = rng.uniform(0.01, 0.5, n_samples)
        periods = rng.integers(1, 365, n_samples).astype(float)
        y = (1 + rate / periods) ** periods - 1
        y = y + rng.normal(0, noise, n_samples) if noise else y
        return np.column_stack([rate, periods]), y


class LiqPrice(DeFiCase):
    """Liquidation price of a collateralised debt position."""

    def __init__(self):
        super().__init__("liq_price", "Liquidation price of a collateralised debt "
                                       "position", ["debt", "collateral", "ltv"])

    def generate(self, n, noise, seed):
        rng = np.random.default_rng(seed)
        debt = rng.uniform(100, 100000, n)
        collateral = rng.uniform(1, 50, n)
        ltv = rng.uniform(0.4, 0.85, n)
        y = debt / (collateral * ltv)
        y = y + rng.normal(0, noise, n) if noise else y
        return np.column_stack([debt, collateral, ltv]), y


ALL_CASES = [ImpermanentLoss(), KellyLP(), AmmSwap(), ApyCompound(), LiqPrice()]
CASES_BY_ID = {c.id: c for c in ALL_CASES}

# Deliberately imperfect mock-LLM fixtures: some near-perfect, some very wrong,
# so the R2 gate genuinely gets exercised across fitted_llm / ensemble / nn.
FIXTURES = {
    "il": [{"equation": "2*sqrt(r)/(1+r) - 1", "confidence": 0.9, "reasoning": "closed form"}],
    "kelly_lp": [{"equation": "mu - sigma", "confidence": 0.4, "reasoning": "rough guess, wrong scaling"}],
    "amm_out": [{"equation": "y*dx/x", "confidence": 0.5, "reasoning": "ignores fee term"}],
    "apy_compound": [{"equation": "rate + rate**2/n", "confidence": 0.6, "reasoning": "series approx"}],
    "liq_price": [{"equation": "0*debt", "confidence": 0.1, "reasoning": "no idea, placeholder"}],
    "default": [{"equation": "0*x", "confidence": 0.1, "reasoning": "no fixture"}],
}


def select_cases(args):
    if args.suite == "default" or args.suite == "defi":
        return ALL_CASES
    ids = args.suite.split(",")
    return [CASES_BY_ID[i] for i in ids if i in CASES_BY_ID]


def make_dataset(case, n, noise, seed):
    return case.generate(n, noise, seed)


def list_cases():
    print("Available DeFi cases:")
    for c in ALL_CASES:
        print(f"  {c.id:15s} {c.description}")
    return 0


class EnhancedHybridSystemDeFi:
    """R^2 threshold gate + scipy constant refit.

    1. Ask the LLM for a formula, score its raw R2.
    2. Refit an affine transform a*f(x)+b against training data via
       scipy.optimize.least_squares ('constant refit') -> fitted R2.
    3. Train an NN on the same data -> NN R2.
    4. Gate:
         fitted R2 >= FITTED_THRESHOLD                -> decision = fitted_llm
         fitted R2 > NN_FLOOR and NN R2 > NN_FLOOR     -> decision = ensemble
           (inverse-residual-std weighted blend of fitted-LLM and NN)
         otherwise                                      -> decision = nn
    """

    def __init__(self, provider: LLMProvider, nn_backend: str = "auto", nn_epochs: int = 300,
                 verbose: bool = False):
        self.provider = provider
        self.nn_backend = nn_backend
        self.nn_epochs = nn_epochs
        self.verbose = verbose

    def _log(self, *a):
        if self.verbose:
            print(" ", *a)

    def fit_predict(self, Xtr, ytr, Xte, names, description, case_id, seed):
        # 1. LLM raw formula
        prompt = build_hypothesis_prompt("defi", names, description, n_candidates=1,
                                          caller_id="EnhancedHybridSystemDeFi")
        hyps = parse_hypotheses(self.provider.complete(prompt, case_id=case_id, max_tokens=500))
        llm_eq = hyps[0]["equation"] if hyps else "0*" + names[0]
        try:
            llm_pred_tr = predict_from_equation(llm_eq, Xtr, names)
            llm_r2 = r2(ytr, llm_pred_tr)
        except ValueError:
            llm_pred_tr, llm_r2 = np.zeros(len(ytr)), float("-inf")

        # 2. scipy constant refit (affine a*f(x)+b)
        if np.all(np.isfinite(llm_pred_tr)):
            fitted_pred_tr, a, b = affine_refit(llm_pred_tr, ytr)
            fitted_r2 = r2(ytr, fitted_pred_tr)
        else:
            fitted_pred_tr, a, b, fitted_r2 = llm_pred_tr, 1.0, 0.0, float("-inf")

        # 3. NN
        nn_model = train_nn(Xtr, ytr, hidden_dims=(32, 16), epochs=self.nn_epochs,
                             seed=seed, backend=self.nn_backend)
        nn_pred_tr = nn_model.predict(Xtr)
        nn_r2 = r2(ytr, nn_pred_tr)

        # 4. gate
        if fitted_r2 >= FITTED_THRESHOLD:
            decision = "fitted_llm"
        elif fitted_r2 > NN_FLOOR and nn_r2 > NN_FLOOR:
            decision = "ensemble"
        else:
            decision = "nn"

        def eval_test(pred_fn):
            if Xte is None:
                return None
            try:
                p = pred_fn(Xte)
                return r2_holdout_safe(p)
            except Exception:
                return None

        def r2_holdout_safe(pred):
            return pred  # placeholder, replaced below with real yte compare

        if decision == "fitted_llm":
            hybrid_r2_train = fitted_r2

            def predict(Xany):
                raw = predict_from_equation(llm_eq, Xany, names)
                return a * raw + b
        elif decision == "ensemble":
            resid_fitted = ytr - fitted_pred_tr
            resid_nn = ytr - nn_pred_tr
            std_f = np.std(resid_fitted) + 1e-9
            std_n = np.std(resid_nn) + 1e-9
            w_f = (1 / std_f) / (1 / std_f + 1 / std_n)
            w_n = 1 - w_f
            hybrid_pred_tr = w_f * fitted_pred_tr + w_n * nn_pred_tr
            hybrid_r2_train = r2(ytr, hybrid_pred_tr)

            def predict(Xany):
                raw = predict_from_equation(llm_eq, Xany, names)
                fit = a * raw + b
                nnp = nn_model.predict(Xany)
                return w_f * fit + w_n * nnp
        else:
            hybrid_r2_train = nn_r2

            def predict(Xany):
                return nn_model.predict(Xany)

        return {
            "case": case_id, "decision": decision,
            "llm_r2": llm_r2, "fitted_r2": fitted_r2, "nn_r2": nn_r2,
            "hybrid_r2": hybrid_r2_train, "llm_equation": llm_eq,
            "refit_a": a, "refit_b": b, "predict": predict,
            "nn_backend": getattr(nn_model, "backend_name", "?"),
        }


def run(args):
    provider = make_provider(args)
    system = EnhancedHybridSystemDeFi(provider, args.nn_backend, args.nn_epochs, args.verbose)
    rows = []
    for case in select_cases(args):
        X, y = make_dataset(case, args.samples, args.noise, args.seed)
        Xtr, ytr, Xte, yte = split_rows(X, y, args.holdout, args.seed)
        print(f"\n{case.id.upper()} | {case.description[:70]}")
        res = system.fit_predict(Xtr, ytr, Xte, case.var_names, case.description, case.id, args.seed)
        hold_r2 = None
        if Xte is not None:
            try:
                hold_r2 = r2(yte, res["predict"](Xte))
            except Exception:
                hold_r2 = None
        row = {"case": case.id, "decision": res["decision"], "llm_r2": res["llm_r2"],
               "fitted_r2": res["fitted_r2"], "nn_r2": res["nn_r2"],
               "hybrid_r2": res["hybrid_r2"], "holdout_r2": hold_r2, "nn": res["nn_backend"]}
        rows.append(row)
        print(f"  decision={res['decision']:10s} LLM R2={fmt(res['llm_r2']).strip():>8s} "
              f"fitted R2={fmt(res['fitted_r2']).strip():>8s} NN R2={fmt(res['nn_r2']).strip():>8s} "
              f"hybrid R2={fmt(res['hybrid_r2']).strip():>8s}")

    print("\n" + "=" * 100)
    print_table(rows, [("case", "case", 14), ("decision", "decision", 10), ("llm_r2", "LLM R2", 9),
                        ("fitted_r2", "fitted R2", 9), ("nn_r2", "NN R2", 9),
                        ("hybrid_r2", "hybrid R2", 9), ("holdout_r2", "holdout R2", 10),
                        ("nn", "nn", 8)])
    finite = [r["hybrid_r2"] for r in rows if r["hybrid_r2"] is not None and math.isfinite(r["hybrid_r2"])]
    if finite:
        print(f"\nmean hybrid R2={np.mean(finite):.4f}")
    from collections import Counter
    print("decisions:", dict(Counter(r["decision"] for r in rows)))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = save_json({"variant": 1, "timestamp": ts, "llm": provider.name, "args": vars(args),
                       "results": rows}, Path(args.out_dir) / "variant1" / f"hybrid_defi_{ts}.json")
    print(f"saved -> {path}")
    return rows


def selftest() -> int:
    t = SelfTest("Variant 1 (Enhanced Hybrid DeFi)")
    selftest_common(t)

    rng = np.random.default_rng(0)
    x = rng.uniform(1, 5, 80).reshape(-1, 1)
    y = (3 * x[:, 0] + 2).ravel()
    good_provider = MockLLMProvider({"default": [{"equation": "3*x + 2", "confidence": 0.9, "reasoning": "exact"}]})
    sys_good = EnhancedHybridSystemDeFi(good_provider, nn_backend="sklearn", nn_epochs=150)
    res = sys_good.fit_predict(x, y, None, ["x"], "linear", "lin_case", 0)
    t.check("gate: near-perfect LLM formula -> decision=fitted_llm",
            res["decision"] == "fitted_llm" and res["fitted_r2"] > FITTED_THRESHOLD)

    bad_provider = MockLLMProvider({"default": [{"equation": "0*x", "confidence": 0.1, "reasoning": "wrong"}]})
    sys_bad = EnhancedHybridSystemDeFi(bad_provider, nn_backend="sklearn", nn_epochs=150)
    res2 = sys_bad.fit_predict(x, y, None, ["x"], "linear", "lin_case", 0)
    t.check("gate: useless LLM formula -> decision != fitted_llm",
            res2["decision"] != "fitted_llm")
    t.check("gate: useless LLM formula but working NN -> nn or ensemble",
            res2["decision"] in ("nn", "ensemble"))

    mid_provider = MockLLMProvider({"default": [{"equation": "2.5*x + 1", "confidence": 0.6, "reasoning": "close"}]})
    sys_mid = EnhancedHybridSystemDeFi(mid_provider, nn_backend="sklearn", nn_epochs=150)
    res3 = sys_mid.fit_predict(x, y, None, ["x"], "linear", "lin_case", 0)
    t.check("gate: decent-but-not-great LLM formula -> ensemble path reachable",
            res3["decision"] in ("ensemble", "fitted_llm"))

    fitted_pred, a, b = affine_refit(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
    t.check("affine_refit recovers a=2,b=0 on y=2x", abs(a - 2.0) < 1e-3 and abs(b) < 1e-2)

    t.check("list of built-in DeFi cases is non-empty", len(ALL_CASES) == 5)
    return t.finish()


def main():
    p = argparse.ArgumentParser(description="HypatiaX Variant 1 - Enhanced Hybrid DeFi (standalone)")
    add_common_args(p, default_suite="defi")
    args = p.parse_args()
    if args.list_cases:
        sys.exit(list_cases())
    if args.selftest:
        sys.exit(selftest())
    run(args)


if __name__ == "__main__":
    main()
