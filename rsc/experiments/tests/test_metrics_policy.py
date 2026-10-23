"""Regression tests for the canonical R^2 evaluator.

Place at rsc/experiments/tests/test_metrics_policy.py and
the module at rsc/methods/metrics.py.
"""
import math
import pathlib
import re

import numpy as np
import pytest

try:
    from rsc.methods.metrics import (
        THRESH_PRIMARY, THRESH_STRICT, is_recovered, r2, r2_checked, selection_score,
    )
except ImportError:  # running from inside rsc/
    from methods.metrics import (
        THRESH_PRIMARY, THRESH_STRICT, is_recovered, r2, r2_checked, selection_score,
    )

RNG = np.random.default_rng(0)


def test_exact_formula_recovers_at_both_thresholds():
    y = RNG.normal(size=200)
    val, status = r2_checked(y, y.copy())
    assert status == "ok"
    assert val == pytest.approx(1.0, abs=1e-12)
    assert is_recovered(val, THRESH_PRIMARY)
    assert is_recovered(val, THRESH_STRICT)


def test_wrong_formula_fails():
    y = RNG.normal(size=200)
    assert not is_recovered(r2(y, y * 1.5), THRESH_PRIMARY)
    assert not is_recovered(r2(y, np.zeros_like(y)), THRESH_PRIMARY)


def test_constant_target_is_undefined_never_recovered():
    y = np.full(100, 5.0)
    for pred in (y.copy(), y + 100.0):  # even an exact match is not a "recovery"
        val, status = r2_checked(y, pred)
        assert math.isnan(val) and status == "degenerate_target"
        assert not is_recovered(val)


def test_near_constant_target_is_flagged_not_scored_on_noise():
    y = 5.0 + 1e-12 * RNG.normal(size=100)
    val, status = r2_checked(y, np.full_like(y, 5.0))
    assert status == "degenerate_target" and math.isnan(val)


def test_all_zero_target_is_degenerate():
    assert r2_checked(np.zeros(50), np.zeros(50))[1] == "degenerate_target"


def test_tiny_scale_but_varying_target_is_scored_normally():
    # Newton-gravity-like magnitudes: absolute variance ~1e-21, relative variance large.
    y = np.linspace(1.0, 2.0, 50) * 1e-11
    assert r2(y, y) == pytest.approx(1.0, abs=1e-12)
    assert not is_recovered(r2(y, y * 1.5), THRESH_PRIMARY)


def test_threshold_edges():
    assert is_recovered(0.99, THRESH_PRIMARY)
    assert not is_recovered(0.98999, THRESH_PRIMARY)
    assert is_recovered(0.999999, THRESH_STRICT)
    assert not is_recovered(0.9999989, THRESH_STRICT)
    assert not is_recovered(float("nan"))
    assert not is_recovered(float("-inf"))


def test_nonfinite_prediction_is_failure_not_undefined():
    y = RNG.normal(size=20)
    p = y.copy()
    p[3] = np.nan
    val, status = r2_checked(y, p)
    assert val == float("-inf") and status == "nonfinite_pred"


def test_mis_shaped_prediction_is_failure():
    val, status = r2_checked(np.ones(5), np.ones(4))
    assert val == float("-inf") and status == "bad_shape"


def test_single_sample_is_degenerate():
    assert r2_checked(np.ones(1), np.ones(1))[1] == "degenerate_target"


def test_selection_score_sends_undefined_to_bottom():
    assert selection_score(float("nan")) == float("-inf")
    assert selection_score(0.7) == 0.7
    assert max([selection_score(float("nan")), selection_score(0.1)]) == 0.1


def test_no_other_r2_implementations():
    """Only metrics.py may compute ss_tot. Fails until every copy is migrated;
    the failure message is the migration checklist."""
    rsc_root = pathlib.Path(__file__).resolve().parents[2]
    pattern = re.compile(r"\bss_tot\b")
    offenders = []
    for path in rsc_root.rglob("*.py"):
        parts = set(path.parts)
        if "__pycache__" in parts or "tests" in parts or path.name == "metrics.py":
            continue
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(rsc_root)}:{lineno}")
    assert not offenders, "R^2 re-implemented outside metrics.py:\n" + "\n".join(offenders)
