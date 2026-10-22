"""Single source of truth for R^2 and recovery scoring.

Every experiment, variant and selection routine must import from here.
Do not re-implement ss_tot / ss_res anywhere else; this is enforced by
test_metrics_policy.py::test_no_other_r2_implementations.

Policy
------
* Degenerate target (variance relative to mean square <= REL_TOL):
  R^2 is undefined -> (nan, "degenerate_target"). Never 0.0, never 1.0.
* Non-finite prediction -> (-inf, "nonfinite_pred"). This is a failure.
* Otherwise R^2 = 1 - ss_res / ss_tot, status "ok".

The tolerance is RELATIVE, so physically tiny but varying targets (e.g.
Newton gravity, ~1e-11) are scored normally, while truly constant or
near-constant targets are flagged instead of being scored on float noise.

Aggregate with status counts, not bare means: nan and -inf must never be
averaged silently. Report degenerate rows separately.
"""
from __future__ import annotations

import numpy as np

REL_TOL = 1e-24
THRESH_PRIMARY = 0.99
THRESH_STRICT = 0.999999


def r2_checked(y_true, y_pred) -> tuple[float, str]:
    """Return (r2, status). Status is one of:
    ok, degenerate_target, nonfinite_target, nonfinite_pred, bad_shape.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape != yp.shape:
        return float("-inf"), "bad_shape"  # method returned the wrong shape: a failure
    if yt.size < 2:
        return float("nan"), "degenerate_target"
    if not np.isfinite(yt).all():
        return float("nan"), "nonfinite_target"
    if not np.isfinite(yp).all():
        return float("-inf"), "nonfinite_pred"

    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    scale = float(np.sum(yt ** 2))
    if ss_tot <= REL_TOL * max(scale, 1e-300):
        return float("nan"), "degenerate_target"

    ss_res = float(np.sum((yt - yp) ** 2))
    return 1.0 - ss_res / ss_tot, "ok"


def r2(y_true, y_pred) -> float:
    """Float-only convenience wrapper (nan if undefined, -inf if pred non-finite)."""
    return r2_checked(y_true, y_pred)[0]


def is_recovered(r2_value: float, threshold: float = THRESH_PRIMARY) -> bool:
    """True only for a finite R^2 at or above the threshold. nan/-inf -> False."""
    return bool(np.isfinite(r2_value) and r2_value >= threshold)


def selection_score(r2_value: float) -> float:
    """Use when ranking candidates by R^2: undefined scores must lose, not win
    or be silently dropped by NaN comparisons."""
    return float("-inf") if not np.isfinite(r2_value) else float(r2_value)
