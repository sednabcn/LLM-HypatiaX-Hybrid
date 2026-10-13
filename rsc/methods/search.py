"""Search engines behind one interface:  engine.run(X, y, names, seed, guesses, unary) -> SearchResult.

PySRSearch      the real engine (needs `pip install pysr` + Julia). NOT executed in the authoring sandbox.
SubstituteSearch greedy forward selection over a symbolic term library (numpy only). It is NOT PySR:
                 numbers from it must never be compared with the paper. It exists so gate logic can be tested.

`guesses` (equation strings) seed the search. The substitute adds each guess as one extra library column;
PySR receives them through its `guesses` argument and refuses to run (pysr.make_regressor) if the installed
build does not support it, rather than silently running unseeded.
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .common import predict_from_equation, equation_ok, r2

DEFAULT_UNARY = ("sq", "sqrt", "log", "inv", "sin", "cos")


@dataclass
class SearchResult:
    expression: str
    r2: float
    predict: Callable
    engine: str
    info: dict = field(default_factory=dict)


def library_terms(names: list, unary) -> list:
    t = list(names)
    for n in names:
        if "sq" in unary:   t.append(f"({n})**2")
        if "sqrt" in unary: t.append(f"sqrt({n})")
        if "log" in unary:  t.append(f"log({n})")
        if "inv" in unary:  t.append(f"1/({n})")
        if "sin" in unary:  t.append(f"sin({n})")
        if "cos" in unary:  t.append(f"cos({n})")
        if "exp" in unary:  t.append(f"exp({n})")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            t.append(f"({a})*({b})")
    return t


class SubstituteSearch:
    engine = "substitute_feature_library"
    supports_guesses = True

    def __init__(self, max_terms: int = 4):
        self.max_terms = max_terms

    def run(self, X, y, names, seed: int = 42, guesses=None, unary=DEFAULT_UNARY) -> SearchResult:
        pool = library_terms(names, unary) + [g for g in (guesses or []) if equation_ok(g, names)]
        cols, terms = [], []
        for term in dict.fromkeys(pool):                      # de-duplicate, keep order
            try:
                v = predict_from_equation(term, X, names)
            except Exception:
                continue
            if v.shape == (len(y),) and np.all(np.isfinite(v)) and np.std(v) > 1e-9:
                cols.append(v); terms.append(term)
        order = list(np.random.default_rng(seed).permutation(len(terms)))   # seed only breaks ties
        M = np.column_stack(cols) if cols else np.zeros((len(y), 0))
        selected: list = []
        best_r2, coefs = -np.inf, np.array([float(np.mean(y))])
        for _ in range(self.max_terms):
            best_j, best_local, best_fit = None, best_r2, None
            for j in order:
                if j in selected:
                    continue
                A = np.column_stack([np.ones(len(y))] + [M[:, k] for k in selected + [j]])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    c, *_ = np.linalg.lstsq(A, y, rcond=None)
                s = r2(y, A @ c)
                if s > best_local:
                    best_local, best_j, best_fit = s, j, c
            if best_j is None or best_local <= best_r2 + 1e-6:
                break
            selected.append(best_j); best_r2, coefs = best_local, best_fit
        parts = [f"{float(c):.10g}*({terms[j]})" for c, j in zip(coefs[1:], selected)]
        expr = " + ".join(parts + [f"{float(coefs[0]):.10g}"])
        pred = predict_from_equation(expr, X, names)
        return SearchResult(expr, r2(y, pred), lambda Xn, e=expr: predict_from_equation(e, Xn, names),
                            self.engine, {"terms": [terms[j] for j in selected], "n_guesses": len(guesses or [])})


class PySRSearch:
    """Real PySR. Untested here (no Julia in the sandbox): run `python -m rsc.methods.smoke --search pysr` first."""
    engine = "pysr"

    def __init__(self, niterations: int | None = None, timeout_s: int | None = None):
        self.niterations, self.timeout_s = niterations, timeout_s

    @property
    def supports_guesses(self) -> bool:
        from . import pysr as P
        return P.supports_guesses()

    def run(self, X, y, names, seed: int = 42, guesses=None, unary=DEFAULT_UNARY) -> SearchResult:
        from . import pysr as P
        kw: dict = dict(binary_operators=["+", "-", "*", "/"])
        ops = [u for u in unary if u in ("sin", "cos", "exp", "log", "sqrt")]
        if "inv" in unary:
            ops.append("inv(x) = 1/x"); kw["extra_sympy_mappings"] = {"inv": lambda x: 1 / x}
        if ops:
            kw["unary_operators"] = ops
        if self.timeout_s:
            kw["timeout_in_seconds"] = self.timeout_s
        model = P.make_regressor(seed=seed, generations=self.niterations, guesses=guesses or None, **kw)
        model.fit(X, y, variable_names=list(names))
        pred = np.asarray(model.predict(X), float)
        return SearchResult(str(model.sympy()), r2(y, pred), lambda Xn: np.asarray(model.predict(Xn), float),
                            self.engine, {"n_guesses": len(guesses or [])})
