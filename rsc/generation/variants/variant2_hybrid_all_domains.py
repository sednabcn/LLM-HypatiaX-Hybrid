#!/usr/bin/env python3
"""
HypatiaX - Variant 2: Hybrid All-Domains (standalone)
========================================================
Class ported: HybridSystemAllDomains
Combination strategy: tie-favours-symbolic gate + force_llm override.

Algorithm
---------
1. Ask the LLM for a single candidate formula, score its R2 against the
   training data.
2. Train an NN on the same data, score its R2.
3. Gate:
     --force-llm set                      -> decision = llm  (reason: forced)
     LLM R2 >= NN R2 - tie_eps             -> decision = llm  (ties AND outright
                                              wins favour the symbolic formula)
     otherwise                             -> decision = nn
   When the NN wins and the LLM's R2 was also <= 0.90, the reason is reported
   as "NN fallback (LLM failed)" -- this mirrors a known mislabelling in the
   source (it fires purely because the NN was better, not because the LLM
   necessarily failed outright).

Built-in benchmark: 5 cases spanning physics, economics and chemistry
(kinematics, gravitation, supply/demand pricing, a chemical equilibrium
quotient, Newton's law of cooling) so the "all domains" framing is real.

Status: standalone re-implementation of the documented gate logic (the
original class lives in a private repo module not available to grep
directly). Runs with only numpy/scikit-learn -- no hypatiax package import.

Usage
  python variant2_hybrid_all_domains.py --selftest
  python variant2_hybrid_all_domains.py                       # all 5 cases, mock LLM
  python variant2_hybrid_all_domains.py --force-llm
  python variant2_hybrid_all_domains.py --llm anthropic        # needs ANTHROPIC_API_KEY
  python variant2_hybrid_all_domains.py --list-cases

As a library:  from variant2_hybrid_all_domains import HybridSystemAllDomains
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
# VARIANT 2 — Hybrid All-Domains  (class: HybridSystemAllDomains)
# Combination strategy: tie-favours-symbolic gate + force_llm override
# =============================================================================
TIE_EPS = 1e-3          # tolerance inside which a "tie" is declared
NN_FALLBACK_R2 = 0.90   # LLM R2 at/below this, when NN wins, is logged as "LLM failed"


@dataclass
class DomainCase:
    id: str
    description: str
    var_names: list
    domain: str

    def generate(self, n, noise, seed):
        raise NotImplementedError


class Kinematics(DomainCase):
    def __init__(self):
        super().__init__("kinematics_v0", "Final velocity under constant acceleration",
                          ["v0", "a", "t"], "physics")

    def generate(self, n, noise, seed):
        rng = np.random.default_rng(seed)
        v0 = rng.uniform(0, 20, n)
        a = rng.uniform(-5, 5, n)
        t = rng.uniform(0, 10, n)
        y = v0 + a * t
        y = y + rng.normal(0, noise, n) if noise else y
        return np.column_stack([v0, a, t]), y


class Gravitation(DomainCase):
    def __init__(self):
        super().__init__("gravitation", "Gravitational force between two point masses",
                          ["m1", "m2", "r"], "physics")

    def generate(self, n, noise, seed):
        rng = np.random.default_rng(seed)
        m1 = rng.uniform(1, 100, n)
        m2 = rng.uniform(1, 100, n)
        r = rng.uniform(1, 50, n)
        G = 6.674e-11
        y = G * m1 * m2 / r ** 2
        y = y + rng.normal(0, noise * abs(y).mean() if noise else 0, n)
        return np.column_stack([m1, m2, r]), y


class SupplyDemand(DomainCase):
    def __init__(self):
        super().__init__("supply_demand", "Market clearing price under linear "
                                           "supply and demand curves", ["a", "b", "c", "d"], "econ")

    def generate(self, n, noise, seed):
        rng = np.random.default_rng(seed)
        a = rng.uniform(10, 50, n)
        b = rng.uniform(0.1, 2, n)
        c = rng.uniform(1, 20, n)
        d = rng.uniform(0.1, 2, n)
        y = (a - c) / (b + d)
        y = y + rng.normal(0, noise, n) if noise else y
        return np.column_stack([a, b, c, d]), y


class ChemEquilibrium(DomainCase):
    def __init__(self):
        super().__init__("chem_eq", "Reaction quotient for a simple A+B<->C "
                                     "equilibrium", ["A", "B", "C"], "chemistry")

    def generate(self, n, noise, seed):
        rng = np.random.default_rng(seed)
        A = rng.uniform(0.1, 5, n)
        B = rng.uniform(0.1, 5, n)
        C = rng.uniform(0.1, 5, n)
        y = C / (A * B)
        y = y + rng.normal(0, noise, n) if noise else y
        return np.column_stack([A, B, C]), y


class Cooling(DomainCase):
    def __init__(self):
        super().__init__("newton_cooling", "Newton's law of cooling: temperature "
                                            "after time t", ["T0", "Tenv", "k", "t"], "physics")

    def generate(self, n, noise, seed):
        rng = np.random.default_rng(seed)
        T0 = rng.uniform(50, 100, n)
        Tenv = rng.uniform(15, 25, n)
        k = rng.uniform(0.01, 0.2, n)
        t = rng.uniform(0, 60, n)
        y = Tenv + (T0 - Tenv) * np.exp(-k * t)
        y = y + rng.normal(0, noise, n) if noise else y
        return np.column_stack([T0, Tenv, k, t]), y


ALL_CASES = [Kinematics(), Gravitation(), SupplyDemand(), ChemEquilibrium(), Cooling()]
CASES_BY_ID = {c.id: c for c in ALL_CASES}

FIXTURES = {
    "kinematics_v0": [{"equation": "v0 + a*t", "confidence": 0.95, "reasoning": "exact"}],
    "gravitation": [{"equation": "6.674e-11*m1*m2/r**2", "confidence": 0.9, "reasoning": "Newton"}],
    "supply_demand": [{"equation": "(a-c)/(b+d)", "confidence": 0.85, "reasoning": "market clearing"}],
    "chem_eq": [{"equation": "C/(A+B)", "confidence": 0.4, "reasoning": "wrong denominator"}],
    "newton_cooling": [{"equation": "Tenv + T0*exp(-k*t)", "confidence": 0.5, "reasoning": "missing (T0-Tenv) term"}],
    "default": [{"equation": "0*x", "confidence": 0.1, "reasoning": "no fixture"}],
}


def select_cases(args):
    if args.suite in ("default", "all", "all_domains"):
        return ALL_CASES
    ids = args.suite.split(",")
    return [CASES_BY_ID[i] for i in ids if i in CASES_BY_ID]


def make_dataset(case, n, noise, seed):
    return case.generate(n, noise, seed)


def list_cases():
    print("Available all-domains cases:")
    for c in ALL_CASES:
        print(f"  {c.id:16s} [{c.domain:10s}] {c.description}")
    return 0


class HybridSystemAllDomains:
    """Tie-favours-symbolic gate + force_llm override.

    Decision is always either 'llm' or 'nn':
      - if force_llm is set, always 'llm' (reason: 'forced').
      - if LLM R2 >= NN R2 - TIE_EPS, ties (and outright wins) favour the
        symbolic/LLM formula: decision = 'llm' (reason: 'symbolic_preferred').
      - otherwise decision = 'nn'. If the LLM's R2 was also <= NN_FALLBACK_R2
        the reason is reported as "NN fallback (LLM failed)" (matching a
        known mislabelling in the source: this fires even when the LLM
        formula wasn't literally broken, just beaten by the NN).
    """

    def __init__(self, provider: LLMProvider, nn_backend: str = "auto", nn_epochs: int = 300,
                 force_llm: bool = False, tie_eps: float = TIE_EPS, verbose: bool = False):
        self.provider = provider
        self.nn_backend = nn_backend
        self.nn_epochs = nn_epochs
        self.force_llm = force_llm
        self.tie_eps = tie_eps
        self.verbose = verbose

    def fit_predict(self, Xtr, ytr, names, description, case_id, seed):
        prompt = build_hypothesis_prompt("all_domains", names, description, n_candidates=1,
                                          caller_id="HybridSystemAllDomains")
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

        if self.force_llm:
            decision, reason = "llm", "forced"
        elif llm_r2 >= nn_r2 - self.tie_eps:
            decision, reason = "llm", "symbolic_preferred (tie favours symbolic)"
        else:
            reason = "NN fallback (LLM failed)" if llm_r2 <= NN_FALLBACK_R2 else "NN better"
            decision = "nn"

        if decision == "llm":
            def predict(Xany):
                return predict_from_equation(llm_eq, Xany, names)
        else:
            def predict(Xany):
                return nn_model.predict(Xany)

        return {"case": case_id, "decision": decision, "reason": reason,
                "llm_r2": llm_r2, "nn_r2": nn_r2, "llm_equation": llm_eq,
                "predict": predict, "nn_backend": getattr(nn_model, "backend_name", "?")}


def run(args):
    provider = make_provider(args)
    system = HybridSystemAllDomains(provider, args.nn_backend, args.nn_epochs,
                                     force_llm=args.force_llm, verbose=args.verbose)
    rows = []
    for case in select_cases(args):
        X, y = make_dataset(case, args.samples, args.noise, args.seed)
        Xtr, ytr, Xte, yte = split_rows(X, y, args.holdout, args.seed)
        print(f"\n{case.id.upper()} [{case.domain}] | {case.description[:60]}")
        res = system.fit_predict(Xtr, ytr, case.var_names, case.description, case.id, args.seed)
        hold_r2 = None
        if Xte is not None:
            try:
                hold_r2 = r2(yte, res["predict"](Xte))
            except Exception:
                hold_r2 = None
        row = {"case": case.id, "domain": case.domain, "decision": res["decision"],
               "reason": res["reason"], "llm_r2": res["llm_r2"], "nn_r2": res["nn_r2"],
               "holdout_r2": hold_r2}
        rows.append(row)
        print(f"  decision={res['decision']:4s} ({res['reason']}) "
              f"LLM R2={fmt(res['llm_r2']).strip():>8s} NN R2={fmt(res['nn_r2']).strip():>8s}")

    print("\n" + "=" * 100)
    print_table(rows, [("case", "case", 16), ("domain", "domain", 10), ("decision", "decision", 9),
                        ("llm_r2", "LLM R2", 9), ("nn_r2", "NN R2", 9),
                        ("holdout_r2", "holdout R2", 10), ("reason", "reason", 30)])
    from collections import Counter
    print("decisions:", dict(Counter(r["decision"] for r in rows)))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = save_json({"variant": 2, "timestamp": ts, "llm": provider.name, "args": vars(args),
                       "results": rows}, Path(args.out_dir) / "variant2" / f"all_domains_{ts}.json")
    print(f"saved -> {path}")
    return rows


def selftest() -> int:
    t = SelfTest("Variant 2 (Hybrid All-Domains)")
    selftest_common(t)

    rng = np.random.default_rng(0)
    x = rng.uniform(1, 5, 80).reshape(-1, 1)
    y = (4 * x[:, 0]).ravel()

    exact_provider = MockLLMProvider({"default": [{"equation": "4*x", "confidence": 0.95, "reasoning": "exact"}]})
    sys1 = HybridSystemAllDomains(exact_provider, nn_backend="sklearn", nn_epochs=150)
    r1 = sys1.fit_predict(x, y, ["x"], "linear", "c1", 0)
    t.check("tie/win favours symbolic: exact LLM formula -> decision=llm", r1["decision"] == "llm")

    bad_provider = MockLLMProvider({"default": [{"equation": "0*x", "confidence": 0.1, "reasoning": "useless"}]})
    sys2 = HybridSystemAllDomains(bad_provider, nn_backend="sklearn", nn_epochs=200)
    r2res = sys2.fit_predict(x, y, ["x"], "linear", "c2", 0)
    t.check("bad LLM formula, working NN -> decision=nn", r2res["decision"] == "nn")
    t.check("bad LLM formula reason mentions 'LLM failed'", "LLM failed" in r2res["reason"])

    sys3 = HybridSystemAllDomains(bad_provider, nn_backend="sklearn", nn_epochs=200, force_llm=True)
    r3 = sys3.fit_predict(x, y, ["x"], "linear", "c3", 0)
    t.check("force_llm overrides the gate even with a useless formula",
            r3["decision"] == "llm" and r3["reason"] == "forced")

    close_provider = MockLLMProvider({"default": [{"equation": "3.999*x", "confidence": 0.9, "reasoning": "near-exact"}]})
    sys4 = HybridSystemAllDomains(close_provider, nn_backend="sklearn", nn_epochs=150, tie_eps=1e-2)
    r4 = sys4.fit_predict(x, y, ["x"], "linear", "c4", 0)
    t.check("near-tie (within tie_eps) still favours symbolic", r4["decision"] == "llm")

    t.check("list of built-in all-domains cases spans >1 domain",
            len({c.domain for c in ALL_CASES}) > 1)
    return t.finish()


def main():
    p = argparse.ArgumentParser(description="HypatiaX Variant 2 - Hybrid All-Domains (standalone)")
    add_common_args(p, default_suite="all_domains")
    p.add_argument("--force-llm", action="store_true", dest="force_llm",
                    help="always take the LLM/symbolic path, overriding the gate")
    p.add_argument("--tie-eps", type=float, default=TIE_EPS, dest="tie_eps")
    args = p.parse_args()
    if args.list_cases:
        sys.exit(list_cases())
    if args.selftest:
        sys.exit(selftest())
    run(args)


if __name__ == "__main__":
    main()
