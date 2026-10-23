"""Shared building blocks (one copy, replacing the block that was pasted into every variant script)."""
from __future__ import annotations
import ast, math, warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

try:
    from rsc.methods.metrics import r2 as _canonical_r2, selection_score as _selection_score
except ImportError:  # running from inside rsc/
    from methods.metrics import r2 as _canonical_r2, selection_score as _selection_score


# ---------------------------------------------------------------- task / result
@dataclass(frozen=True)
class Task:
    """Everything a method may see. There is deliberately NO ground-truth-formula field: a leak through
    the Task is structurally impossible. X_test_features (features only, never y) exists because the
    original V6 uses it to detect extrapolation; it is None unless the harness opts in."""
    X: np.ndarray
    y: np.ndarray
    names: list
    description: str
    domain: str = "general"
    seed: int = 42
    case_id: str = ""
    X_test_features: np.ndarray | None = None

    def __post_init__(self):
        if self.X.ndim != 2 or len(self.X) != len(self.y) or self.X.shape[1] != len(self.names):
            raise ValueError(f"inconsistent task shapes X{self.X.shape} y{self.y.shape} names{len(self.names)}")


@dataclass
class FitResult:
    variant: str
    decision: str
    predict: Callable[[np.ndarray], np.ndarray]
    train_r2: float
    equation: str | None = None
    engine: str = ""
    api_calls: int = 0
    search_calls: int = 0
    info: dict = field(default_factory=dict)


def safe_predict(fit: FitResult, X: np.ndarray) -> np.ndarray:
    """Raw predictions or NaNs: scoring recomputes R2 from these, never from a method-reported number."""
    try:
        p = np.asarray(fit.predict(X), dtype=float).ravel()
        return p if p.shape == (len(X),) else np.full(len(X), np.nan)
    except Exception:
        return np.full(len(X), np.nan)


# ---------------------------------------------------------------- metrics
def r2(y_true, y_pred):
    """Delegates to rsc.methods.metrics; undefined R^2 -> -inf (fail closed)."""
    return _selection_score(_canonical_r2(y_true, y_pred))


# ---------------------------------------------------------------- sandboxed evaluation
class EquationError(ValueError):
    """Equation is unparseable / uses something outside the allowed grammar."""


_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant, ast.Load,
                  ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Mod)
_ALLOWED_FUNCS = {"sqrt", "log", "exp", "sin", "cos", "tan", "abs", "Abs", "sign",
                  "arcsin", "arccos", "arctan", "asin", "acos", "atan"}
_FUNCS = {"sqrt": np.sqrt, "log": np.log, "exp": np.exp, "sin": np.sin, "cos": np.cos, "tan": np.tan,
          "abs": np.abs, "Abs": np.abs, "sign": np.sign, "arcsin": np.arcsin, "arccos": np.arccos,
          "arctan": np.arctan, "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan}
_CONSTS = {"pi": math.pi, "e": math.e, "h": 6.62607015e-34, "hbar": 1.0545718176e-34, "c": 2.99792458e8,
           "k_B": 1.380649e-23, "k": 1.380649e-23, "N_A": 6.02214076e23, "g_n": 9.80665,
           "m_e": 9.1093837015e-31, "q_e": 1.602176634e-19, "epsilon0": 8.8541878128e-12, "mu0": 1.25663706212e-6}
MAX_EXPR_CHARS = 2000


class _FloatConsts(ast.NodeTransformer):
    """All numeric literals become floats. Without this, `9**9**9` is an exact-integer power that never
    finishes (a one-line DoS from any LLM answer); as floats it raises OverflowError immediately."""
    def visit_Constant(self, node):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise EquationError("only numeric constants are allowed")
        return ast.copy_location(ast.Constant(float(node.value)), node)


def _parse(expr: str) -> ast.Expression:
    if not isinstance(expr, str) or not expr.strip() or len(expr) > MAX_EXPR_CHARS:
        raise EquationError("empty, non-string or over-long expression")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:                      # originals let SyntaxError escape the ValueError handlers
        raise EquationError(f"syntax error: {e.msg}") from e
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise EquationError(f"disallowed element: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise EquationError("dunder access blocked")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise EquationError("function not allowed")
    return ast.fix_missing_locations(_FloatConsts().visit(tree))


def screen_code(expr: str) -> None:
    _parse(expr)


def equation_ok(expr: str, names: list) -> bool:
    """Parses, passes the grammar screen, and uses only known variables / constants / functions."""
    try:
        tree = _parse(expr)
    except EquationError:
        return False
    known = set(names) | set(_FUNCS) | set(_CONSTS)
    return all(n.id in known for n in ast.walk(tree) if isinstance(n, ast.Name))


def predict_from_equation(equation: str, X: np.ndarray, names: list) -> np.ndarray:
    tree = _parse(equation)
    ns = {n: X[:, i] for i, n in enumerate(names)}
    ns.update(_FUNCS)
    for k, v in _CONSTS.items():
        ns.setdefault(k, v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = eval(compile(tree, "<equation>", "eval"), {"__builtins__": {}}, ns)
    out = np.asarray(out, dtype=float)
    return np.full(X.shape[0], float(out)) if out.ndim == 0 else out


def try_predict(equation: str | None, X: np.ndarray, names: list) -> np.ndarray | None:
    """None on ANY failure (syntax, unknown name, overflow, wrong shape). Variants branch on None."""
    if not equation:
        return None
    try:
        p = predict_from_equation(equation, X, names)
        return p if p.shape == (len(X),) else None
    except Exception:
        return None


def affine_refit(pred: np.ndarray, y: np.ndarray):
    """Least squares a*pred+b (closed form; the original used scipy least_squares on the same linear problem)."""
    A = np.column_stack([pred, np.ones_like(pred)])
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return a * pred + b, float(a), float(b)


def inverse_std_weights(resid_a: np.ndarray, resid_b: np.ndarray):
    sa, sb = float(np.std(resid_a)) + 1e-9, float(np.std(resid_b)) + 1e-9
    wa = (1 / sa) / (1 / sa + 1 / sb)
    return wa, 1 - wa
