#!/usr/bin/env python3
"""
variant6_validation_selected_hybrid.py
=======================================
Standalone extraction of **Variant 6 — Validation-Selected Residual Hybrid**.

Real source : hypatiax/experiments/benchmarks/hypatiax_defi_benchmark_v4.py
Real symbols: _select_v4_candidate(), _v4_hybrid_predict_and_eval()
Verified    : this session, against a fresh clone of
              github.com/sednabcn/LLM-HypatiaX-REPRO (uploaded copy of the
              real file diffs at 0 lines against the repo).

WHAT IS PORTED FAITHFULLY (gate / decision logic, byte-for-byte in spirit):
  - The 5-candidate pool: llm, nn:<arch> (x2 architectures), residual_nn:<arch>
    (NN fit on LLM residuals), blend:<arch> (21-point alpha grid search over
    llm/nn mix), linear_fallback / linear_fallback_local.
  - Internal-validation split drawn from the *high end* of the training
    region along the extrapolation variable (never the real test set).
  - The extrapolation guard: when the benchmark's test domain falls outside
    the observed training range, bare `nn:<arch>` candidates are excluded
    from the pool entirely, and if literally nothing LLM-anchored survives,
    selection falls through to ranking the two linear-fallback candidates
    on the same held-out edge slice instead of guessing.
  - Deterministic tie-break priority: llm > residual > blend > linear
    fallbacks > nn.

WHAT IS HONESTLY SUBSTITUTED (documented, not hidden):
  - The real file trains a PyTorch MLP (_fit_nn_predict, torch.nn). This
    standalone script substitutes scikit-learn's MLPRegressor to avoid a
    torch dependency. Same role in the gate (a black-box NN candidate),
    weaker/faster fit quality -- do not compare its R² numbers to the real
    benchmark's reported figures.
  - The real "llm" candidate calls the Anthropic API to generate a Python
    formula. This script takes `llm_predict_fn` as a caller-supplied
    callable (X -> y) so the gate logic can be exercised and unit-tested
    without an API key; pass `llm_predict_fn=None` to see the pool degrade
    exactly as the real code does when no LLM formula is available.

Run standalone for a demo on synthetic in-domain and extrapolative cases:
    python variant6_validation_selected_hybrid.py
"""
from __future__ import annotations
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

_V4_ARCHITECTURES = ([64, 32], [128, 64, 32])
_V4_BLEND_GRID = np.linspace(0.0, 1.0, 21)
_V4_MIN_VAL = 5
_V4_VAL_FRAC = 0.2


# ── metrics ──────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if not np.all(np.isfinite(y_pred)):
        return {"r2": float("-inf"), "mae": float("inf"), "rmse": float("inf")}
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else float("-inf"))
    return {"r2": r2, "mae": float(np.mean(np.abs(y_true - y_pred))),
            "rmse": float(np.sqrt(ss_res / max(len(y_true), 1)))}


# ── data extrapolation check ────────────────────────────────────────────
def extrapolates(X_train: np.ndarray, X_test: np.ndarray) -> bool:
    """True if any test feature falls outside the observed train range.
    Inspects features only -- never y -- so it introduces no label leakage."""
    Xtr, Xte = np.atleast_2d(X_train), np.atleast_2d(X_test)
    return bool(np.any(Xte.min(axis=0) < Xtr.min(axis=0)) or
                np.any(Xte.max(axis=0) > Xtr.max(axis=0)))


# ── internal validation split (real Fix 24 logic) ──────────────────────
def split_internal_validation(X_train, y_train, split_var_idx: int = 0,
                               val_frac: float = _V4_VAL_FRAC):
    n = len(X_train)
    if n < 2 * _V4_MIN_VAL:
        return X_train, y_train, None, None
    vals = X_train[:, split_var_idx] if (X_train.ndim >= 2 and X_train.shape[1] > split_var_idx) \
        else X_train.flatten()
    order = np.argsort(vals)
    n_val = max(_V4_MIN_VAL, int(round(n * val_frac)))
    n_val = min(n_val, n - _V4_MIN_VAL)
    inner, val = order[:-n_val], order[-n_val:]
    return X_train[inner], y_train[inner], X_train[val], y_train[val]


# ── NN candidate (substituted: sklearn MLP, not the real torch model) ──
def fit_nn_predict(X_fit, y_fit, X_eval, hidden, seed: int):
    sx, sy = StandardScaler(), StandardScaler()
    Xf = sx.fit_transform(np.atleast_2d(X_fit))
    yf = sy.fit_transform(np.asarray(y_fit).reshape(-1, 1)).flatten()
    model = MLPRegressor(hidden_layer_sizes=tuple(hidden), max_iter=800,
                          random_state=seed, early_stopping=True)
    model.fit(Xf, yf)
    pred = model.predict(sx.transform(np.atleast_2d(X_eval)))
    return sy.inverse_transform(pred.reshape(-1, 1)).flatten()


# ── linear fallback candidates (ridge affine / log-affine + local clamp) ─
def _ridge_affine_fit(X1, y, lam):
    A = X1.T @ X1 + lam * np.eye(X1.shape[1])
    return np.linalg.solve(A, X1.T @ y)


def fit_linear_fallback(X_train, y_train, X_test, ridge_lambda: float = 1e-2,
                         clamp_factor: float = 3.0) -> np.ndarray:
    Xtr, Xte = np.atleast_2d(X_train).astype(float), np.atleast_2d(X_test).astype(float)
    ytr = np.asarray(y_train, dtype=float)

    def _affine(Xtr_, ytr_, Xte_):
        Xtr1 = np.column_stack([np.ones(len(Xtr_)), Xtr_])
        Xte1 = np.column_stack([np.ones(len(Xte_)), Xte_])
        coef = _ridge_affine_fit(Xtr1, ytr_, ridge_lambda)
        return Xte1 @ coef, Xtr1 @ coef

    pred_te, pred_tr = _affine(Xtr, ytr, Xte)
    best_pred, best_resid = pred_te, float(np.mean((pred_tr - ytr) ** 2))

    if np.all(Xtr > 0) and np.all(Xte > 0) and np.all(ytr > 0):
        log_te, log_tr = _affine(np.log(Xtr), np.log(ytr), np.log(Xte))
        resid = float(np.mean((np.exp(log_tr) - ytr) ** 2))
        if np.all(np.isfinite(log_te)) and resid < best_resid:
            best_pred, best_resid = np.exp(log_te), resid

    if not np.all(np.isfinite(best_pred)):
        best_pred = np.nan_to_num(best_pred, nan=float(np.median(ytr)),
                                   posinf=float(ytr.max()), neginf=float(ytr.min()))
    lo, hi = ytr.min(), ytr.max()
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    return np.clip(best_pred, lo - clamp_factor * span, hi + clamp_factor * span)


def fit_linear_fallback_local(X_train, y_train, X_test, k: int = 8,
                               buffer_mult: float = 6.0) -> np.ndarray:
    base = fit_linear_fallback(X_train, y_train, X_test)
    Xtr, Xte = np.atleast_2d(X_train).astype(float), np.atleast_2d(X_test).astype(float)
    ytr = np.asarray(y_train, dtype=float)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd
    k_eff = min(k, len(Xtr))
    out = base.copy()
    for i in range(len(Xte_s)):
        dist = np.sum((Xtr_s - Xte_s[i]) ** 2, axis=1)
        local_y = ytr[np.argsort(dist)[:k_eff]]
        lo, hi = float(local_y.min()), float(local_y.max())
        span = hi - lo if hi > lo else max(abs(hi), 1.0)
        out[i] = float(np.clip(base[i], lo - buffer_mult * span, hi + buffer_mult * span))
    return out


def rank_fallback_candidates(Xi, yi, Xv, yv) -> dict:
    out = {}
    try:
        out["linear_fallback"] = compute_metrics(yv, fit_linear_fallback(Xi, yi, Xv))["r2"]
    except Exception:
        pass
    try:
        out["linear_fallback_local"] = compute_metrics(yv, fit_linear_fallback_local(Xi, yi, Xv))["r2"]
    except Exception:
        pass
    return out


# ── the gate itself: real _select_v4_candidate logic, ported ───────────
def select_v4_candidate(X_train, y_train, llm_predict_fn, seed: int,
                         extrapolative: bool = False) -> dict:
    """
    llm_predict_fn: callable(X) -> y, or None if no trustworthy LLM formula
                    is available (mirrors the real code's llm_code == "").
    """
    Xi, yi, Xv, yv = split_internal_validation(X_train, y_train)
    if Xv is None:
        if llm_predict_fn is not None:
            selected = "llm"
        elif extrapolative:
            selected = "linear_fallback"
        else:
            selected = "nn"
        return {"selected": selected, "validation_r2": {}, "validation_n": 0}

    llm_i = llm_predict_fn(Xi) if llm_predict_fn else None
    llm_v = llm_predict_fn(Xv) if llm_predict_fn else None
    llm_ok = llm_i is not None and llm_v is not None

    candidates = {}
    if llm_ok:
        candidates["llm"] = {"r2": compute_metrics(yv, llm_v)["r2"], "hidden": None, "alpha": 1.0}

    for hidden in _V4_ARCHITECTURES:
        try:
            nn_v = fit_nn_predict(Xi, yi, Xv, hidden, seed)
            candidates[f"nn:{hidden}"] = {"r2": compute_metrics(yv, nn_v)["r2"], "hidden": hidden, "alpha": 0.0}
            if llm_ok:
                residual = yi - llm_i
                res_v = fit_nn_predict(Xi, residual, Xv, hidden, seed)
                candidates[f"residual:{hidden}"] = {
                    "r2": compute_metrics(yv, llm_v + res_v)["r2"], "hidden": hidden, "alpha": 1.0}
                best_alpha, best_r2 = 0.0, -np.inf
                for a in _V4_BLEND_GRID:
                    r2 = compute_metrics(yv, a * llm_v + (1.0 - a) * nn_v)["r2"]
                    if r2 > best_r2:
                        best_r2, best_alpha = r2, float(a)
                candidates[f"blend:{hidden}"] = {"r2": best_r2, "hidden": hidden, "alpha": best_alpha}
        except Exception:
            continue

    if not candidates:
        if extrapolative:
            fb = rank_fallback_candidates(Xi, yi, Xv, yv) or {"linear_fallback": float("nan")}
            winner = max(fb, key=lambda k: (fb[k], k == "linear_fallback"))
            return {"selected": winner, "validation_r2": fb, "validation_n": len(yv)}
        return {"selected": "nn", "validation_r2": {}, "validation_n": len(yv)}

    priority = {"llm": 0, "residual": 1, "blend": 2, "linear_fallback": 3,
                "linear_fallback_local": 4, "nn": 5}
    pool = candidates
    if extrapolative:
        anchored = {k: v for k, v in candidates.items() if not k.startswith("nn:")}
        if anchored:
            pool = anchored
        else:
            fb = rank_fallback_candidates(Xi, yi, Xv, yv) or {"linear_fallback": float("nan")}
            for name, r2v in fb.items():
                candidates[name] = {"r2": r2v, "hidden": None, "alpha": 1.0}
            pool = {name: candidates[name] for name in fb}

    winner_key = max(pool, key=lambda k: (pool[k]["r2"], -priority.get(k.split(":")[0], 9)))
    prefix = winner_key.split(":")[0]
    selected_name = "residual_nn" if prefix == "residual" else prefix
    return {
        "selected": selected_name,
        "hidden": candidates[winner_key].get("hidden"),
        "blend_alpha": float(candidates[winner_key].get("alpha", 1.0)),
        "validation_r2": {k: v["r2"] for k, v in candidates.items()},
        "validation_n": len(yv),
    }


# ── demo ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    def make_case(n=60, extrap=False):
        X = rng.uniform(1.0, 5.0, size=(n, 2))
        y = 2.0 * X[:, 0] + 0.5 * X[:, 1] ** 2 + rng.normal(0, 0.1, n)
        if extrap:
            X_test = rng.uniform(5.0, 9.0, size=(15, 2))     # outside train range
        else:
            X_test = rng.uniform(1.0, 5.0, size=(15, 2))
        y_test = 2.0 * X_test[:, 0] + 0.5 * X_test[:, 1] ** 2
        return X, y, X_test, y_test

    llm_fn = lambda X: 2.0 * X[:, 0] + 0.5 * X[:, 1] ** 2   # stand-in "correct" LLM formula

    for label, extrap in [("in-domain", False), ("extrapolative", True)]:
        X, y, X_test, y_test = make_case(extrap=extrap)
        is_extrap = extrapolates(X, X_test)
        result = select_v4_candidate(X, y, llm_fn, seed=42, extrapolative=is_extrap)
        print(f"\n[{label}]  extrapolative={is_extrap}")
        print(f"  selected     : {result['selected']}")
        print(f"  validation r2: { {k: round(v, 4) for k, v in result['validation_r2'].items()} }")

        print(f"  --- also demoing no-LLM-available case ---")
        result_nollm = select_v4_candidate(X, y, None, seed=42, extrapolative=is_extrap)
        print(f"  selected (no llm): {result_nollm['selected']}")
