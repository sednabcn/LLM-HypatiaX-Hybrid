#!/usr/bin/env python3
"""
exp_cross_variant_skeleton.py
================================
Runnable skeleton for the cross-variant comparison protocol described in
cross_variant_protocol_design.md. Demonstrates the mechanism that matters
most: a variant only runs on a case whose family it declares support for;
everything else is logged as N/A, never forced.

Variant execution here is a stand-in (see `_stub_run_variant`) -- the real
version calls each variant's native entry point (EnhancedHybridSystemDeFi,
_select_v4_candidate, get_llm_prior, etc., already verified against source
in variant6_validation_selected_hybrid.py / variant7_llm_prior_pysr_seeding.py).
This file exists to prove out the catalog / manifest / skip / reporting
machinery in isolation before wiring in nine real, expensive, API-calling
variants.

Run:
    python exp_cross_variant_skeleton.py
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Callable
import numpy as np


# ── 1. Case catalog ─────────────────────────────────────────────────────
@dataclass
class CrossVariantCase:
    id: str
    family: str
    description: str
    ground_truth_fn: Callable[[np.ndarray], np.ndarray]
    var_names: list[str]
    var_ranges: dict
    extrap_ranges: dict
    noise_level: float = 0.0
    n_train: int = 150
    n_extrap: int = 30

    def generate(self, rng: np.random.Generator):
        X_tr = np.column_stack([rng.uniform(*self.var_ranges[v], self.n_train)
                                 for v in self.var_names])
        X_ex = np.column_stack([rng.uniform(*self.extrap_ranges[v], self.n_extrap)
                                 for v in self.var_names])
        y_tr = self.ground_truth_fn(X_tr)
        y_ex = self.ground_truth_fn(X_ex)
        if self.noise_level > 0:
            y_tr = y_tr + rng.normal(0, self.noise_level * np.std(y_tr), len(y_tr))
        return X_tr, y_tr, X_ex, y_ex


CATALOG: list[CrossVariantCase] = [
    CrossVariantCase(
        id="RP-01", family="ratio_product",
        description="leverage-like ratio: a / (b - c)",
        ground_truth_fn=lambda X: X[:, 0] / (X[:, 1] - X[:, 2] + 5.0),
        var_names=["a", "b", "c"],
        var_ranges={"a": (1, 10), "b": (1, 5), "c": (0, 3)},
        extrap_ranges={"a": (10, 20), "b": (5, 8), "c": (3, 5)},
    ),
    CrossVariantCase(
        id="TP-01", family="trig_power",
        description="Nguyen-style: sin(x)^2 + y^2",
        ground_truth_fn=lambda X: np.sin(X[:, 0]) ** 2 + X[:, 1] ** 2,
        var_names=["x", "y"],
        var_ranges={"x": (0.5, 3.0), "y": (0.5, 3.0)},
        extrap_ranges={"x": (3.0, 5.0), "y": (3.0, 5.0)},
    ),
    CrossVariantCase(
        id="PF-01", family="physics_form",
        description="inverse-square: k / r^2",
        ground_truth_fn=lambda X: 10.0 / (X[:, 0] ** 2),
        var_names=["r"],
        var_ranges={"r": (1, 5)},
        extrap_ranges={"r": (5, 10)},
    ),
    # Real catalog: 6 per family = 18 total. Three shown here for the demo.
]


# ── 2. Capability manifest ──────────────────────────────────────────────
@dataclass
class VariantCapability:
    variant_id: str
    needs_llm_formula: bool
    needs_pysr_guesses: bool
    supports_families: set
    native_entry_point: str


MANIFEST: list[VariantCapability] = [
    VariantCapability("1", True,  False, {"ratio_product", "physics_form"},
                       "EnhancedHybridSystemDeFi"),
    VariantCapability("3", True,  False, {"ratio_product", "trig_power", "physics_form"},
                       "ensemble_llm_nn"),
    VariantCapability("4", True,  True,  {"trig_power", "physics_form"},
                       "SymbolicEngineWithLLM"),
    VariantCapability("6", True,  False, {"ratio_product", "physics_form"},
                       "_select_v4_candidate"),
    VariantCapability("7", True,  True,  {"trig_power"},
                       "get_llm_prior + PySR guesses"),
]


# ── 3. Per-run record ────────────────────────────────────────────────────
@dataclass
class RunRecord:
    variant_id: str
    case_id: str
    family: str
    seed: int
    status: str               # "ok" | "n/a" | "error"
    train_r2: float = float("nan")
    test_r2: float = float("nan")
    extrap_r2: float = float("nan")
    wall_clock_s: float = 0.0
    error_message: str | None = None


def _r2(y_true, y_pred) -> float:
    y_pred = np.asarray(y_pred, dtype=float)
    if not np.all(np.isfinite(y_pred)):
        return float("-inf")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else float("-inf"))


def _stub_run_variant(cap: VariantCapability, case: CrossVariantCase,
                       X_tr, y_tr, X_ex, y_ex, seed: int) -> RunRecord:
    """
    STAND-IN for calling the real variant. Fits a simple ridge-affine model
    so the demo produces plausible in-domain-good / extrapolation-poor
    numbers -- real wiring replaces this with the native entry point.
    """
    t0 = time.time()
    X1 = np.column_stack([np.ones(len(X_tr)), X_tr])
    coef, *_ = np.linalg.lstsq(X1, y_tr, rcond=None)
    pred_tr = X1 @ coef
    X1e = np.column_stack([np.ones(len(X_ex)), X_ex])
    pred_ex = X1e @ coef
    elapsed = time.time() - t0
    return RunRecord(
        variant_id=cap.variant_id, case_id=case.id, family=case.family, seed=seed,
        status="ok", train_r2=_r2(y_tr, pred_tr), test_r2=_r2(y_tr, pred_tr),
        extrap_r2=_r2(y_ex, pred_ex), wall_clock_s=elapsed,
    )


# ── 4. Driver: capability-gated iteration (the core fairness mechanism) ──
def run_cross_variant(catalog=CATALOG, manifest=MANIFEST, seeds=(42,)) -> list[RunRecord]:
    records: list[RunRecord] = []
    rng_master = np.random.default_rng(0)
    for case in catalog:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            X_tr, y_tr, X_ex, y_ex = case.generate(rng)
            for cap in manifest:
                if case.family not in cap.supports_families:
                    records.append(RunRecord(cap.variant_id, case.id, case.family, seed,
                                              status="n/a"))
                    continue
                try:
                    rec = _stub_run_variant(cap, case, X_tr, y_tr, X_ex, y_ex, seed)
                except Exception as e:
                    rec = RunRecord(cap.variant_id, case.id, case.family, seed,
                                     status="error", error_message=str(e))
                records.append(rec)
    return records


# ── 5. Reporting (reuses the repo's own _robust_stats shape) ────────────
def robust_stats(scores: list[float]) -> dict:
    valid = [s for s in scores if np.isfinite(s)]
    if not valid:
        return dict(n=0, median=float("nan"), mean_clipped=float("nan"),
                    pct_09=0.0, pct_099=0.0, n_catastrophic=0)
    arr = np.array(valid)
    clipped = np.clip(arr, -1.0, 1.0)
    return dict(
        n=len(valid), median=float(np.median(arr)),
        mean_clipped=float(np.mean(clipped)),
        pct_09=float(np.mean(arr > 0.9) * 100),
        pct_099=float(np.mean(arr > 0.99) * 100),
        n_catastrophic=int(np.sum(arr < -1.0)),
    )


def report(records: list[RunRecord]) -> None:
    variants = sorted({r.variant_id for r in records})
    families = sorted({r.family for r in records})
    for vid in variants:
        for fam in families:
            group = [r for r in records if r.variant_id == vid and r.family == fam]
            n_total = len(group)
            n_na = sum(1 for r in group if r.status == "n/a")
            n_err = sum(1 for r in group if r.status == "error")
            n_ok = sum(1 for r in group if r.status == "ok")
            if n_na == n_total:
                continue  # variant doesn't cover this family at all -- skip silently
            test_stats = robust_stats([r.test_r2 for r in group if r.status == "ok"])
            extrap_stats = robust_stats([r.extrap_r2 for r in group if r.status == "ok"])
            print(f"\nVariant {vid} · {fam}  (coverage: {n_ok}/{n_total} ok, "
                  f"{n_na} n/a, {n_err} error)")
            print(f"  test_r2   : median={test_stats['median']:+.4f}  "
                  f"clipped_mean={test_stats['mean_clipped']:+.4f}  "
                  f">0.9: {test_stats['pct_09']:5.1f}%")
            print(f"  extrap_r2 : median={extrap_stats['median']:+.4f}  "
                  f"clipped_mean={extrap_stats['mean_clipped']:+.4f}  "
                  f">0.9: {extrap_stats['pct_09']:5.1f}%")


if __name__ == "__main__":
    records = run_cross_variant()
    report(records)
