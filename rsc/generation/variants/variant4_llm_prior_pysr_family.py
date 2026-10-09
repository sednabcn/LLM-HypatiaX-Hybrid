#!/usr/bin/env python3
"""
HypatiaX - Variant 4: LLM-Prior / PySR-family (standalone)
==============================================================
Class ported: SymbolicEngineWithLLM (4 modes)
Combination strategy: LLM <-> search take turns proposing/refining.

Algorithm (documented modes)
-----------------------------
  none     pure symbolic search, no LLM.
  seed     LLM's first hypothesis -> seed term(s) that warm-start the search.
  hybrid   LLM first: R2_llm > 0.95 -> LLM only, search skipped.
           0.5 <= R2_llm <= 0.95    -> search seeded/biased by the LLM prior,
                                        keep whichever ends up better.
           R2_llm < 0.5             -> discard the prior, pure search.
  fallback search first: R2 > 0.90 -> done; else ask the LLM and keep the
           better of the two.

Search backend
--------------
The repo's real engine is PySR (needs `pip install pysr` + Julia) or a
numpy-only genetic-programming tree search. Neither is practical to embed
here in a way that's actually executed and tested, so this standalone port
uses a lighter, fully-tested substitute: FeatureLibrarySearch. It builds a
bank of candidate symbolic terms per variable (x, x^2, sqrt(x), log(x),
1/x, sin(x), cos(x), and pairwise products) and does greedy forward
selection with an ordinary-least-squares refit at each step. This is a
genuine structure search, just a much simpler one than PySR's genetic
search over arbitrary expression trees -- do not compare its R2 numbers to
PySR-family results from the paper.

Built-in benchmark: 6 Nguyen-style symbolic regression cases.

Status: standalone re-implementation of the documented mode-dispatch logic,
grep-verified against the transcript's own description of the 4 modes and
the retry/threshold behaviour. The PySR and live-Anthropic code paths are
NOT executed in the authoring environment -- only the mock-LLM +
feature-library-search paths were run here.

Usage
  python variant4_llm_prior_pysr_family.py --selftest
  python variant4_llm_prior_pysr_family.py                          # all 6 cases, all 4 modes, mock LLM
  python variant4_llm_prior_pysr_family.py --modes hybrid --suite nguyen5
  python variant4_llm_prior_pysr_family.py --llm anthropic           # needs ANTHROPIC_API_KEY

As a library:  from variant4_llm_prior_pysr_family import SymbolicEngineWithLLM
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
# VARIANT 4 — LLM-Prior / PySR Family  (class: SymbolicEngineWithLLM, 4 modes)
# Combination strategy: LLM <-> search take turns proposing/refining
#
# Search backend note: the repo's real engine uses PySR (needs Julia) or a
# numpy-only genetic-programming tree search. Neither is practical to embed
# here untested, so this standalone port uses a lighter, fully-tested
# feature-library search (build a bank of candidate terms per variable --
# x, x^2, sqrt(x), log(x), 1/x, sin(x), cos(x), pairwise products -- and pick
# the sparse linear combination via least squares + greedy pruning). This is
# a genuine symbolic/structure search, just a simpler one than PySR's GP;
# do not compare its R2 to PySR-family numbers from the paper.
# =============================================================================
@dataclass
class DiscoveryConfig:
    max_terms: int = 4          # sparse-search budget (analogous to PySR maxsize)
    candidate_pool: int = 40


class _Invalid(Exception):
    pass


def _safe_term(name: str, X: np.ndarray, idx: dict) -> np.ndarray:
    """Evaluate one named feature-library term on X. Raises _Invalid on a
    domain violation (log of <=0, etc.) so bad terms are dropped, not crashed
    on.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name.startswith("sqrt("):
            v = X[:, idx[name[5:-1]]]
            if np.any(v < 0):
                raise _Invalid
            return np.sqrt(v)
        if name.startswith("log("):
            v = X[:, idx[name[4:-1]]]
            if np.any(v <= 0):
                raise _Invalid
            return np.log(v)
        if name.startswith("inv("):
            v = X[:, idx[name[4:-1]]]
            if np.any(np.abs(v) < 1e-9):
                raise _Invalid
            return 1.0 / v
        if name.startswith("sq("):
            v = X[:, idx[name[3:-1]]]
            return v ** 2
        if name.startswith("sin("):
            return np.sin(X[:, idx[name[4:-1]]])
        if name.startswith("cos("):
            return np.cos(X[:, idx[name[4:-1]]])
        if "*" in name:
            a, b = name.split("*")
            return X[:, idx[a]] * X[:, idx[b]]
        return X[:, idx[name]]


def _term_library(names: list) -> list:
    terms = list(names)
    for n in names:
        terms += [f"sq({n})", f"sqrt({n})", f"log({n})", f"inv({n})", f"sin({n})", f"cos({n})"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            terms.append(f"{a}*{b}")
    return terms


def _term_to_equation(term: str) -> str:
    return (term.replace("sq(", "(").replace("inv(", "1/(")
            if term.startswith(("sq(", "inv(")) else term)


class FeatureLibrarySearch:
    """The 'search' half of the LLM<->search loop. Greedy forward selection
    over a small symbolic feature library, refit by ordinary least squares
    at each step (a compact, fully numpy/scipy stand-in for PySR/minigp).
    """
    name = "feature_library"

    def __init__(self, cfg: DiscoveryConfig):
        self.cfg = cfg

    def search(self, X, y, names, seed_terms=None) -> dict:
        idx = {n: i for i, n in enumerate(names)}
        pool = _term_library(names)
        cols, valid_terms = [], []
        for term in pool:
            try:
                v = _safe_term(term, X, idx)
                if np.all(np.isfinite(v)) and np.std(v) > 1e-9:
                    cols.append(v)
                    valid_terms.append(term)
            except _Invalid:
                continue
        if not cols:
            return {"expression": "0", "r2_score": 0.0, "terms": [], "coeffs": [], "intercept": 0.0}

        M = np.column_stack(cols)
        selected, coeffs, intercept = [], [], float(np.mean(y))
        residual = y - intercept
        remaining = list(range(len(valid_terms)))

        # seed terms (from the LLM prior) are tried first, forced in if useful
        if seed_terms:
            for st in seed_terms:
                for j in list(remaining):
                    if valid_terms[j] == st:
                        remaining.remove(j)
                        selected.append(j)
                        break

        best_r2 = -np.inf
        for _ in range(self.cfg.max_terms):
            best_j, best_local_r2, best_fit = None, best_r2, None
            trial_cols = selected + [None]
            for j in remaining:
                cand = selected + [j]
                A = np.column_stack([np.ones(len(y))] + [M[:, k] for k in cand])
                try:
                    coefs, *_ = np.linalg.lstsq(A, y, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                pred = A @ coefs
                cand_r2 = r2(y, pred)
                if cand_r2 > best_local_r2:
                    best_local_r2, best_j, best_fit = cand_r2, j, coefs
            if best_j is None or best_local_r2 <= best_r2 + 1e-6:
                break
            selected.append(best_j)
            remaining.remove(best_j)
            best_r2 = best_local_r2
            intercept = float(best_fit[0])
            coeffs = [float(c) for c in best_fit[1:]]

        terms = [valid_terms[j] for j in selected]
        parts = [f"{c:.6g}*{_term_to_equation(t)}" for c, t in zip(coeffs, terms)]
        expr = " + ".join(parts) + (f" + {intercept:.6g}" if abs(intercept) > 1e-9 else "")
        if not parts:
            expr = f"{intercept:.6g}"
        return {"expression": expr, "r2_score": float(best_r2 if selected else 0.0),
                "terms": terms, "coeffs": coeffs, "intercept": intercept}


# ---------------------------------------------------------------------------
# SymbolicEngineWithLLM: the 4 modes, ported from the documented spec.
# ---------------------------------------------------------------------------
class SymbolicEngineWithLLM:
    """
    Modes (report Section 3.4):
      none    - pure search, no LLM.
      seed    - LLM's first hypothesis -> seed term(s) for the search.
      hybrid  - LLM first: R2_llm > 0.95 -> LLM only, search skipped.
                0.5 <= R2_llm <= 0.95 -> search seeded by the LLM prior; keep
                the better of the two. R2_llm < 0.5 -> discard the prior,
                pure search; keep the better.
      fallback- search first: R2 > 0.90 -> done; else ask the LLM, keep the
                better of the two.
    """

    def __init__(self, backend: FeatureLibrarySearch, provider: Optional[LLMProvider],
                 domain: str = "general", llm_mode: str = "none", verbose: bool = False):
        self.backend = backend
        self.provider = provider
        self.domain = domain
        self.llm_mode = llm_mode if provider is not None else "none"
        self.verbose = verbose

    def _log(self, *a):
        if self.verbose:
            print(" ", *a)

    def _ask_llm(self, X, y, names, desc, case_id):
        if self.provider is None:
            return []
        prompt = build_hypothesis_prompt(self.domain, names, desc, n_candidates=1,
                                          caller_id="SymbolicEngineWithLLM")
        hyps = parse_hypotheses(self.provider.complete(prompt, case_id=case_id, max_tokens=500))
        for h in hyps:
            try:
                pred = predict_from_equation(h["equation"], X, names)
                h["r2_score"] = r2(y, pred)
            except ValueError:
                h["r2_score"] = float("-inf")
        return hyps

    def discover(self, X, y, names, description=None, case_id=None) -> dict:
        desc = description or "unknown"
        mode = self.llm_mode

        if mode == "none":
            r = self.backend.search(X, y, names)
            r["llm_mode"] = "none"
            return r

        hyps = self._ask_llm(X, y, names, desc, case_id)
        best_hyp = max(hyps, key=lambda h: h["r2_score"]) if hyps else None

        if mode == "seed":
            seed_terms = None
            if best_hyp is not None:
                seed_terms = [t for t in _term_library(names) if t in best_hyp["equation"]]
            r = self.backend.search(X, y, names, seed_terms=seed_terms)
            r["llm_mode"] = "seed"
            r["llm_hypothesis"] = best_hyp["equation"] if best_hyp else None
            return r

        if mode == "hybrid":
            if best_hyp is None:
                r = self.backend.search(X, y, names)
                r["llm_mode"] = "hybrid_llm_failed"
                return r
            self._log(f"LLM best: {best_hyp['equation']} R2={best_hyp['r2_score']:.4f}")
            if best_hyp["r2_score"] > 0.95:
                return {"expression": best_hyp["equation"], "r2_score": best_hyp["r2_score"],
                        "llm_mode": "hybrid_llm_only", "engine": "llm"}
            seed_terms = None
            if best_hyp["r2_score"] >= 0.5:
                seed_terms = [t for t in _term_library(names) if t in best_hyp["equation"]]
            else:
                self._log("LLM R2 < 0.5 -> discarding prior, pure search")
            r = self.backend.search(X, y, names, seed_terms=seed_terms)
            if r["r2_score"] > best_hyp["r2_score"]:
                r["llm_mode"] = "hybrid_search_better"
                return r
            return {"expression": best_hyp["equation"], "r2_score": best_hyp["r2_score"],
                    "llm_mode": "hybrid_llm_better", "engine": "llm"}

        if mode == "fallback":
            r = self.backend.search(X, y, names)
            if r["r2_score"] > 0.90:
                r["llm_mode"] = "fallback_search_only"
                return r
            hyps2 = self._ask_llm(X, y, names, desc, case_id)
            best2 = max(hyps2, key=lambda h: h["r2_score"]) if hyps2 else None
            if best2 is None:
                r["llm_mode"] = "fallback_both_failed"
                return r
            if best2["r2_score"] > r["r2_score"]:
                return {"expression": best2["equation"], "r2_score": best2["r2_score"],
                        "llm_mode": "fallback_llm_better", "engine": "llm"}
            r["llm_mode"] = "fallback_search_better"
            return r

        r = self.backend.search(X, y, names)
        r["llm_mode"] = "none"
        return r


# ---------------------------------------------------------------------------
# Built-in benchmark: Nguyen-style symbolic regression suite
# ---------------------------------------------------------------------------
@dataclass
class SymbolicCase:
    id: str
    description: str
    var_names: list
    fn: Callable

    def generate(self, n, noise, seed):
        rng = np.random.default_rng(seed)
        X = rng.uniform(0.3, 3.0, (n, len(self.var_names)))
        y = self.fn(X)
        y = y + rng.normal(0, noise, n) if noise else y
        return X, y


ALL_CASES = [
    SymbolicCase("nguyen1", "x^3 + x^2 + x", ["x"], lambda X: X[:, 0] ** 3 + X[:, 0] ** 2 + X[:, 0]),
    SymbolicCase("nguyen4", "x^4 + x^3 + x^2 + x", ["x"],
                 lambda X: X[:, 0] ** 4 + X[:, 0] ** 3 + X[:, 0] ** 2 + X[:, 0]),
    SymbolicCase("nguyen5", "sin(x^2)*cos(x) - 1", ["x"], lambda X: np.sin(X[:, 0] ** 2) * np.cos(X[:, 0]) - 1),
    SymbolicCase("nguyen9", "sin(x)+sin(y^2)", ["x", "y"], lambda X: np.sin(X[:, 0]) + np.sin(X[:, 1] ** 2)),
    SymbolicCase("nguyen10", "2*sin(x)*cos(y)", ["x", "y"], lambda X: 2 * np.sin(X[:, 0]) * np.cos(X[:, 1])),
    SymbolicCase("nguyen12", "x^4 - x^3 + 0.5*y^2 - y", ["x", "y"],
                 lambda X: X[:, 0] ** 4 - X[:, 0] ** 3 + 0.5 * X[:, 1] ** 2 - X[:, 1]),
]
CASES_BY_ID = {c.id: c for c in ALL_CASES}

FIXTURES = {
    "nguyen1": [{"equation": "x**3 + x**2", "confidence": 0.7, "reasoning": "missing linear term"}],
    "nguyen4": [{"equation": "x**4 + x**3 + x**2 + x", "confidence": 0.9, "reasoning": "exact"}],
    "nguyen5": [{"equation": "sin(x**2)", "confidence": 0.5, "reasoning": "partial"}],
    "nguyen9": [{"equation": "sin(x) + sin(y)", "confidence": 0.5, "reasoning": "missing square on y"}],
    "nguyen10": [{"equation": "sin(x)*cos(y)", "confidence": 0.6, "reasoning": "missing factor of 2"}],
    "nguyen12": [{"equation": "0*x", "confidence": 0.1, "reasoning": "no idea"}],
    "default": [{"equation": "0*x", "confidence": 0.1, "reasoning": "no fixture"}],
}


def select_cases(args):
    if args.suite in ("default", "nguyen"):
        return ALL_CASES
    ids = args.suite.split(",")
    return [CASES_BY_ID[i] for i in ids if i in CASES_BY_ID]


def make_dataset(case, n, noise, seed):
    return case.generate(n, noise, seed)


def list_cases():
    print("Available symbolic-regression cases:")
    for c in ALL_CASES:
        print(f"  {c.id:10s} {c.description}")
    return 0


def run(args):
    provider = make_provider(args) if "none" not in args.modes or len(args.modes) > 1 else None
    if provider is None:
        provider = make_provider(args)
    cfg = DiscoveryConfig(max_terms=args.max_terms)
    backend = FeatureLibrarySearch(cfg)
    rows = []
    for case in select_cases(args):
        X, y = make_dataset(case, args.samples, args.noise, args.seed)
        Xtr, ytr, Xte, yte = split_rows(X, y, args.holdout, args.seed)
        print(f"\n{case.id.upper()} | {case.description}")
        for mode in args.modes:
            t0 = time.time()
            engine = SymbolicEngineWithLLM(backend, provider if mode != "none" else None,
                                            "symbolic", mode, verbose=args.verbose)
            res = engine.discover(Xtr, ytr, case.var_names, case.description, case.id)
            hold_r2 = None
            if Xte is not None:
                try:
                    hold_r2 = r2(yte, predict_from_equation(
                        res["expression"].replace("^", "**"), Xte, case.var_names))
                except Exception:
                    hold_r2 = None
            row = {"case": case.id, "mode": mode, "r2": res["r2_score"],
                   "holdout_r2": hold_r2, "llm_mode": res.get("llm_mode"),
                   "secs": round(time.time() - t0, 2), "expression": res["expression"]}
            rows.append(row)
            print(f"  {mode:9s} R2={fmt(row['r2']).strip():>8s} [{str(row['llm_mode']):22s}] "
                  f"{row['secs']:>5.2f}s  {str(row['expression'])[:55]}")

    print("\n" + "=" * 100)
    print_table(rows, [("case", "case", 10), ("mode", "mode", 9), ("r2", "R2", 9),
                        ("holdout_r2", "holdout", 9), ("llm_mode", "outcome", 24), ("secs", "secs", 6)])
    for m in args.modes:
        vals = [r["r2"] for r in rows if r["mode"] == m and r["r2"] is not None]
        if vals:
            print(f"  mode {m:9s} mean R2 = {np.mean(vals):.4f} ({len(vals)} cases)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = save_json({"variant": 4, "timestamp": ts, "llm": provider.name, "backend": backend.name,
                       "args": vars(args), "results": rows},
                      Path(args.out_dir) / "variant4" / f"llm_prior_search_{ts}.json")
    print(f"saved -> {path}")
    return rows


def selftest() -> int:
    t = SelfTest("Variant 4 (LLM-Prior / Search family)")
    selftest_common(t)

    cfg = DiscoveryConfig(max_terms=3)
    backend = FeatureLibrarySearch(cfg)
    rng = np.random.default_rng(1)
    X = rng.uniform(0.5, 3, (150, 2))
    y = 2 * X[:, 0] * X[:, 1] + 1
    r = backend.search(X, y, ["a", "b"])
    t.check(f"feature-library search recovers y=2ab+1 (R2={r['r2_score']:.4f})", r["r2_score"] > 0.999)

    Xd = np.random.default_rng(3).uniform(1, 2, (60, 1))
    yd = Xd[:, 0] ** 2
    good_provider = MockLLMProvider({"default": [{"equation": "x**2", "confidence": 0.95, "reasoning": "exact"}]})
    e = SymbolicEngineWithLLM(backend, good_provider, "d", "hybrid")
    r = e.discover(Xd, yd, ["x"])
    t.check("hybrid: LLM R2 > 0.95 -> LLM only (search skipped)",
            r["llm_mode"] == "hybrid_llm_only")

    mid_provider = MockLLMProvider({"default": [{"equation": "0.85*x**2", "confidence": 0.7, "reasoning": "close but wrong scale"}]})
    e2 = SymbolicEngineWithLLM(backend, mid_provider, "d", "hybrid")
    r2res = e2.discover(Xd, yd, ["x"])
    t.check("hybrid: LLM R2 in [0.5,0.95] -> search is seeded/attempted",
            r2res["llm_mode"] in ("hybrid_search_better", "hybrid_llm_better"))

    bad_provider = MockLLMProvider({"default": [{"equation": "0*x", "confidence": 0.1, "reasoning": "useless"}]})
    e3 = SymbolicEngineWithLLM(backend, bad_provider, "d", "hybrid")
    r3 = e3.discover(Xd, yd, ["x"])
    t.check("hybrid: LLM R2 < 0.5 -> prior discarded, pure search wins",
            r3["llm_mode"] == "hybrid_search_better")

    e4 = SymbolicEngineWithLLM(backend, good_provider, "d", "fallback")
    r4 = e4.discover(Xd, yd, ["x"])
    t.check("fallback: search R2 > 0.90 -> LLM never needed",
            r4["llm_mode"] == "fallback_search_only")

    e5 = SymbolicEngineWithLLM(backend, None, "d", "none")
    r5 = e5.discover(Xd, yd, ["x"])
    t.check("none: llm_mode is 'none' and no provider used", r5["llm_mode"] == "none")

    t.check("built-in symbolic-regression suite is non-empty", len(ALL_CASES) >= 4)
    return t.finish()


def main():
    p = argparse.ArgumentParser(description="HypatiaX Variant 4 - LLM-Prior / Search family (standalone)")
    add_common_args(p, default_suite="nguyen")
    p.add_argument("--modes", nargs="+", default=["none", "seed", "hybrid", "fallback"],
                    choices=["none", "seed", "hybrid", "fallback"])
    p.add_argument("--max-terms", type=int, default=4, dest="max_terms")
    args = p.parse_args()
    if args.list_cases:
        sys.exit(list_cases())
    if args.selftest:
        sys.exit(selftest())
    run(args)


if __name__ == "__main__":
    main()
