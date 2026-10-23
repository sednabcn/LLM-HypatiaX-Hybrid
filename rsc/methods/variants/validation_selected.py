"""V6 -- validation-selected residual hybrid (ported from _select_v4_candidate in hypatiax_defi_benchmark_v4.py).
The only variant that selects on HELD-OUT data (the high end of the training range), not on training R2.
NN candidates now use the shared NNBaseline (torch when installed), where the standalone script used sklearn.
Open decision: it also reads X_test_features (features only) to detect extrapolation, information the other
variants never see. Set use_test_features=False to remove that advantage; the choice is recorded in the result."""
from __future__ import annotations
import numpy as np

from ..common import FitResult, Task, r2, try_predict
from ..hypotheses import propose, best_usable
from .base import Variant, nan_predict

ARCHITECTURES = ((64, 32), (128, 64, 32))
BLEND_GRID = np.linspace(0.0, 1.0, 21)
MIN_VAL, VAL_FRAC = 5, 0.2
PRIORITY = {"llm": 0, "residual": 1, "blend": 2, "linear_fallback": 3, "linear_fallback_local": 4, "nn": 5}


def extrapolates(X_train, X_test) -> bool:
    a, b = np.atleast_2d(X_train), np.atleast_2d(X_test)
    return bool(np.any(b.min(0) < a.min(0)) or np.any(b.max(0) > a.max(0)))


def split_internal_validation(X, y, split_var_idx: int = 0, val_frac: float = VAL_FRAC):
    n = len(X)
    if n < 2 * MIN_VAL:
        return X, y, None, None
    order = np.argsort(X[:, split_var_idx])
    n_val = min(max(MIN_VAL, int(round(n * val_frac))), n - MIN_VAL)
    inner, val = order[:-n_val], order[-n_val:]
    return X[inner], y[inner], X[val], y[val]


def _affine(Xtr, ytr, Xte, lam):
    A1, B1 = np.column_stack([np.ones(len(Xtr)), Xtr]), np.column_stack([np.ones(len(Xte)), Xte])
    coef = np.linalg.solve(A1.T @ A1 + lam * np.eye(A1.shape[1]), A1.T @ ytr)
    return B1 @ coef, A1 @ coef


def fit_linear_fallback(Xtr, ytr, Xte, lam: float = 1e-2, clamp: float = 3.0):
    Xtr, Xte, ytr = np.atleast_2d(Xtr).astype(float), np.atleast_2d(Xte).astype(float), np.asarray(ytr, float)
    best, best_res = None, 0.0
    p_te, p_tr = _affine(Xtr, ytr, Xte, lam)
    best, best_res = p_te, float(np.mean((p_tr - ytr) ** 2))
    if np.all(Xtr > 0) and np.all(Xte > 0) and np.all(ytr > 0):
        l_te, l_tr = _affine(np.log(Xtr), np.log(ytr), np.log(Xte), lam)
        res = float(np.mean((np.exp(l_tr) - ytr) ** 2))
        if np.all(np.isfinite(l_te)) and res < best_res:
            best = np.exp(l_te)
    if not np.all(np.isfinite(best)):
        best = np.nan_to_num(best, nan=float(np.median(ytr)), posinf=float(ytr.max()), neginf=float(ytr.min()))
    lo, hi = ytr.min(), ytr.max()
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    return np.clip(best, lo - clamp * span, hi + clamp * span)


def fit_linear_fallback_local(Xtr, ytr, Xte, k: int = 8, buffer_mult: float = 6.0):
    base = fit_linear_fallback(Xtr, ytr, Xte)
    Xtr, Xte, ytr = np.atleast_2d(Xtr).astype(float), np.atleast_2d(Xte).astype(float), np.asarray(ytr, float)
    mu, sd = Xtr.mean(0), np.where(Xtr.std(0) == 0, 1.0, Xtr.std(0))
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    out, k = base.copy(), min(k, len(Xtr))
    for i in range(len(B)):
        ly = ytr[np.argsort(np.sum((A - B[i]) ** 2, axis=1))[:k]]
        lo, hi = float(ly.min()), float(ly.max())
        span = hi - lo if hi > lo else max(abs(hi), 1.0)
        out[i] = float(np.clip(base[i], lo - buffer_mult * span, hi + buffer_mult * span))
    return out


def _rank_fallbacks(Xi, yi, Xv, yv) -> dict:
    out = {}
    for name, fn in (("linear_fallback", fit_linear_fallback), ("linear_fallback_local", fit_linear_fallback_local)):
        try:
            out[name] = r2(yv, fn(Xi, yi, Xv))
        except Exception:
            pass
    return out


def select_candidate(task: Task, stack, llm_fn, extrapolative: bool) -> dict:
    Xi, yi, Xv, yv = split_internal_validation(task.X, task.y)
    if Xv is None:
        sel = "llm" if llm_fn else ("linear_fallback" if extrapolative else "nn")
        return {"selected": sel, "hidden": None, "alpha": 1.0, "validation_r2": {}, "validation_n": 0}
    llm_i = llm_fn(Xi) if llm_fn else None
    llm_v = llm_fn(Xv) if llm_fn else None
    llm_ok = llm_i is not None and llm_v is not None and np.all(np.isfinite(llm_i)) and np.all(np.isfinite(llm_v))
    cands: dict = {}
    if llm_ok:
        cands["llm"] = {"r2": r2(yv, llm_v), "hidden": None, "alpha": 1.0}
    for hidden in ARCHITECTURES:
        try:
            nn_v = stack.new_nn(task, hidden=hidden).fit(Xi, yi).predict(Xv)
            cands[f"nn:{hidden}"] = {"r2": r2(yv, nn_v), "hidden": hidden, "alpha": 0.0}
            if llm_ok:
                res_v = stack.new_nn(task, hidden=hidden).fit(Xi, yi - llm_i).predict(Xv)
                cands[f"residual:{hidden}"] = {"r2": r2(yv, llm_v + res_v), "hidden": hidden, "alpha": 1.0}
                best_a, best_s = 0.0, -np.inf
                for a in BLEND_GRID:                       # strict '>' as in the source: first best alpha in grid order
                    s = r2(yv, a * llm_v + (1 - a) * nn_v)
                    if s > best_s:
                        best_s, best_a = s, float(a)
                cands[f"blend:{hidden}"] = {"r2": best_s, "hidden": hidden, "alpha": best_a}
        except Exception:
            continue
    if not cands:
        if extrapolative:
            fb = _rank_fallbacks(Xi, yi, Xv, yv) or {"linear_fallback": float("nan")}
            return {"selected": max(fb, key=lambda k: (fb[k], k == "linear_fallback")), "hidden": None, "alpha": 1.0,
                    "validation_r2": fb, "validation_n": len(yv)}
        return {"selected": "nn", "hidden": None, "alpha": 0.0, "validation_r2": {}, "validation_n": len(yv)}
    pool = cands
    if extrapolative:
        anchored = {k: v for k, v in cands.items() if not k.startswith("nn:")}
        if anchored:
            pool = anchored
        else:
            fb = _rank_fallbacks(Xi, yi, Xv, yv) or {"linear_fallback": float("nan")}
            for n, s in fb.items():
                cands[n] = {"r2": s, "hidden": None, "alpha": 1.0}
            pool = {n: cands[n] for n in fb}
    key = max(pool, key=lambda k: (pool[k]["r2"], -PRIORITY.get(k.split(":")[0], 9)))
    prefix = key.split(":")[0]
    return {"selected": "residual_nn" if prefix == "residual" else prefix, "hidden": cands[key]["hidden"],
            "alpha": float(cands[key]["alpha"]), "validation_r2": {k: v["r2"] for k, v in cands.items()},
            "validation_n": len(yv)}


class V6ValidationSelected(Variant):
    id, name = "V6", "Validation-selected hybrid"

    def fit(self, task: Task, stack) -> FitResult:
        use_feats = self.p("use_test_features", True)
        extrap = bool(use_feats and task.X_test_features is not None and extrapolates(task.X, task.X_test_features))
        h = best_usable(propose(stack, task, 1))
        eq = h.equation if h else None
        llm_fn = (lambda X, e=eq: try_predict(e, X, task.names)) if h else None
        sel = select_candidate(task, stack, llm_fn, extrap)
        s, hidden, alpha = sel["selected"], sel["hidden"], sel["alpha"]
        Xtr, ytr = task.X, task.y

        if s == "llm":
            predict = lambda X: llm_fn(X) if llm_fn(X) is not None else nan_predict(len(X))
        elif s == "nn":
            nn = stack.new_nn(task, hidden=hidden or 64).fit(Xtr, ytr)
            predict = nn.predict
        elif s == "residual_nn":
            res_nn = stack.new_nn(task, hidden=hidden).fit(Xtr, ytr - llm_fn(Xtr))
            predict = lambda X: llm_fn(X) + res_nn.predict(X)
        elif s == "blend":
            nn = stack.new_nn(task, hidden=hidden).fit(Xtr, ytr)
            predict = lambda X: alpha * llm_fn(X) + (1 - alpha) * nn.predict(X)
        elif s == "linear_fallback_local":
            predict = lambda X: fit_linear_fallback_local(Xtr, ytr, X)
        else:
            predict = lambda X: fit_linear_fallback(Xtr, ytr, X)
        try:
            train_r2 = r2(ytr, predict(Xtr))
        except Exception:
            train_r2 = float("-inf")
        return FitResult(self.id, s, predict, train_r2, equation=eq,
                         info={**{k: v for k, v in sel.items() if k != "selected"}, "extrapolative": extrap,
                               "used_test_features": bool(use_feats and task.X_test_features is not None),
                               "selects_on": "internal_validation_r2"})
