#!/usr/bin/env python3
"""
experiment_protocol_hybrid.py
==============================
Cross-protocol, capability-gated experiment protocol for comparing all 8
catalogued HypatiaX variants (1, 2, 3, 4, 5, 6, 6-PCA, 7).

Why this file exists
---------------------
The repo already has four protocol modules, each with an *identical*
runner-facing interface:

    protocol.get_all_domains() -> list[str]
    protocol.load_test_data(domain, num_samples=None, noise_level=None,
                             seed=None) -> list[(description, X, y,
                                                  var_names, metadata)]

    experiment_protocol_defi.py         -> DeFiExperimentProtocol
    experiment_protocol_benchmark_v2.py -> BenchmarkProtocol   (Feynman/SRBench)
    experiment_protocol_nguyen12.py     -> NguYenProtocol
    experiment_protocol_all_30.py       -> ExperimentProtocolAll

This file does NOT reimplement any of them. It is a thin adapter layer
that:

  1. Wraps all four behind the SAME get_all_domains()/load_test_data()
     interface, so it drops into run_all.sh / run_comparative_hybrid_methods.py
     exactly the way each of the four already does (see
     hybrid_ci_pipeline_plan.md, Phase 1 step 4).
  2. Tags every case that comes out of every protocol with a `family`
     and a `protocol_source`, in metadata, so variants can be gated by
     what they actually support rather than what protocol happens to be
     running.
  3. Ships a VARIANT_MANIFEST (grounded in
     hypatiax_variants_source_mapping_report.md and the "what's genuinely
     good / where it breaks down" discussion in Hybrid-pipeline.txt) that
     says, per variant, which families/protocols it is honestly able to
     run on.
  4. Runs the exp_cross_variant_skeleton.py gate mechanism against the
     real (or, standalone, a small representative fallback) case set:
     a variant only executes on a case whose family+protocol it declares
     support for. Everything else is logged "n/a", never forced.
  5. Reports strictly per (variant, protocol_source) -- never pooled
     across protocols -- because the four protocols have different
     denominators, noise regimes and success thresholds (the all_30
     benchmark file is explicit that headline R²>0.99 rates use a FIXED
     denominator of 74 and are not comparable across suites; pooling here
     would repeat that exact mistake at a larger scale).

Explicitly NOT attempted here (see Hybrid-pipeline.txt's pushback on the
original ask): forcing all 8 variants onto all 4 protocols as a uniform
8x4 cross product. Variant 6/6-PCA's candidate selector is DeFi-feature
specific; Variant 7's PySR-guesses mechanism has no analog outside
Nguyen-12. Porting either is a reviewed redesign task, not a side effect
of this file. What IS delivered: every one of the 8 variants has at
least one real, honest family/protocol it can run against here, and the
gate mechanism makes the boundary explicit and auditable instead of
silent.

`_run_variant` is an explicit, documented stand-in -- same convention as
exp_cross_variant_skeleton.py -- for calling each variant's real native
entry point (EnhancedHybridSystemDeFi, ensemble_llm_nn,
SymbolicEngineWithLLM, _select_v4_candidate, get_llm_prior, ...). Wire
the real calls in when integrating into run_comparative_hybrid_methods.py;
nothing else in this file needs to change to support that swap.

Usage
-----
    python experiment_protocol_hybrid.py --describe
    python experiment_protocol_hybrid.py --list-coverage
    python experiment_protocol_hybrid.py --run
    python experiment_protocol_hybrid.py --run --protocols defi,nguyen12 \\
        --variants 6,6-pca,7 --sample-cases 3 --seeds 42,99
"""
from __future__ import annotations

import argparse
import os as _os
import pathlib as _pathlib
import sys as _sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# ── sys.path bootstrap ───────────────────────────────────────────────────
# Same convention as the other experiment_protocol_*.py files: resolves
# hypatiax.* imports whether this file is run directly or imported by
# run_all_checkpoint.py / run_comparative_hybrid_methods.py.
_PROTO_DIR = _pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _pathlib.Path(_os.environ.get("REPRO_ROOT", str(_PROTO_DIR.parent)))
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "hypatiax")]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _os, _pathlib, _sys, _p


# ═══════════════════════════════════════════════════════════════════════
# 1. Adapters over the four real protocol modules (with graceful fallback)
# ═══════════════════════════════════════════════════════════════════════

# family + noteworthy flags assigned per protocol source. This mapping is
# a heuristic first pass (documented, not silent) -- refine per-domain if
# a specific domain's formula shape doesn't match its protocol's default
# family (e.g. a DeFi "staking" compounding-growth case is arguably closer
# to "physics_broad" than "ratio_product"; left coarse-grained on purpose
# so the gate mechanism stays simple and auditable).
_PROTOCOL_DEFAULT_FAMILY = {
    "defi": "ratio_product",
    "feynman": "physics_broad",
    "nguyen12": None,        # resolved per-domain below (poly/transc/bivariate)
    "all30": "multi_domain_breadth",
}

_NGUYEN_DOMAIN_FAMILY = {
    "nguyen_polynomial": "nguyen_polynomial",
    "nguyen_transcendental": "nguyen_transcendental",
    "nguyen_bivariate": "nguyen_bivariate",
}


class _ProtocolAdapter:
    """Wraps one real protocol object behind a uniform interface and
    stamps `family` / `protocol_source` onto every case's metadata."""

    def __init__(self, source: str, protocol_obj):
        self.source = source
        self._proto = protocol_obj

    def get_all_domains(self) -> list[str]:
        return list(self._proto.get_all_domains())

    def load_test_data(self, domain: str, num_samples: Optional[int] = None,
                        seed: Optional[int] = None) -> list[tuple]:
        kwargs = {}
        if num_samples is not None:
            kwargs["num_samples"] = num_samples
        if seed is not None:
            kwargs["seed"] = seed
        raw = self._proto.load_test_data(domain, **kwargs)
        tagged = []
        for desc, X, y, var_names, meta in raw:
            meta = dict(meta)  # never mutate the source protocol's own dict
            meta.setdefault("protocol_source", self.source)
            family = _NGUYEN_DOMAIN_FAMILY.get(domain) or _PROTOCOL_DEFAULT_FAMILY[self.source]
            meta.setdefault("family", family)
            meta.setdefault("domain", domain)
            tagged.append((desc, X, y, var_names, meta))
        return tagged


def _load_real_adapters() -> dict[str, _ProtocolAdapter]:
    """Import the four real protocol modules. Any subset may be missing
    (e.g. a review checkout without the full hypatiax/ tree) -- import
    failures are per-module and non-fatal so partial coverage still runs.
    """
    adapters: dict[str, _ProtocolAdapter] = {}

    try:
        from experiment_protocol_defi import DeFiExperimentProtocol
        adapters["defi"] = _ProtocolAdapter("defi", DeFiExperimentProtocol())
    except Exception as e:
        print(f"[hybrid-protocol] defi adapter unavailable: {e}")

    try:
        from experiment_protocol_benchmark_v2 import BenchmarkProtocol
        adapters["feynman"] = _ProtocolAdapter("feynman", BenchmarkProtocol())
    except Exception as e:
        print(f"[hybrid-protocol] feynman adapter unavailable: {e}")

    try:
        from experiment_protocol_nguyen12 import NguYenProtocol
        adapters["nguyen12"] = _ProtocolAdapter("nguyen12", NguYenProtocol())
    except Exception as e:
        print(f"[hybrid-protocol] nguyen12 adapter unavailable: {e}")

    try:
        from experiment_protocol_all_30 import ExperimentProtocolAll
        adapters["all30"] = _ProtocolAdapter("all30", ExperimentProtocolAll())
    except Exception as e:
        print(f"[hybrid-protocol] all30 adapter unavailable: {e}")

    return adapters


# ═══════════════════════════════════════════════════════════════════════
# 2. Standalone fallback catalog (used only if none of the four real
#    protocol modules import -- keeps --describe / --list-coverage / a
#    smoke --run usable for review without the full repo checked out).
#    Mirrors exp_cross_variant_skeleton.py's CATALOG, extended with a
#    nguyen-shaped family so the fallback path still exercises all 8
#    variants' gates, not just the 5 the skeleton demoed.
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class _FallbackCase:
    id: str
    family: str
    protocol_source: str
    domain: str
    description: str
    ground_truth_fn: Callable[[np.ndarray], np.ndarray]
    var_names: list[str]
    var_ranges: dict
    extrap_ranges: dict
    noise_level: float = 0.0
    n_train: int = 150
    n_extrap: int = 30

    def as_test_tuple(self, num_samples: Optional[int] = None, seed: int = 42):
        rng = np.random.default_rng(seed)
        n_tr = num_samples or self.n_train
        X_tr = np.column_stack([rng.uniform(*self.var_ranges[v], n_tr)
                                 for v in self.var_names])
        X_ex = np.column_stack([rng.uniform(*self.extrap_ranges[v], self.n_extrap)
                                 for v in self.var_names])
        y_tr = self.ground_truth_fn(X_tr)
        y_ex = self.ground_truth_fn(X_ex)
        if self.noise_level > 0:
            y_tr = y_tr + rng.normal(0, self.noise_level * np.std(y_tr), len(y_tr))
        meta = {
            "id": self.id, "family": self.family,
            "protocol_source": self.protocol_source, "domain": self.domain,
            "extrapolation_test": True,
            "X_extrap": X_ex, "y_extrap": y_ex,
        }
        return self.description, X_tr, y_tr, self.var_names, meta


_FALLBACK_CATALOG: list[_FallbackCase] = [
    _FallbackCase(
        id="RP-01", family="ratio_product", protocol_source="defi", domain="amm",
        description="leverage-like ratio: a / (b - c)",
        ground_truth_fn=lambda X: X[:, 0] / (X[:, 1] - X[:, 2] + 5.0),
        var_names=["a", "b", "c"],
        var_ranges={"a": (1, 10), "b": (1, 5), "c": (0, 3)},
        extrap_ranges={"a": (10, 20), "b": (5, 8), "c": (3, 5)},
    ),
    _FallbackCase(
        id="TP-01", family="physics_broad", protocol_source="feynman", domain="feynman_mechanics",
        description="Feynman-style: sin(x)^2 + y^2",
        ground_truth_fn=lambda X: np.sin(X[:, 0]) ** 2 + X[:, 1] ** 2,
        var_names=["x", "y"],
        var_ranges={"x": (0.5, 3.0), "y": (0.5, 3.0)},
        extrap_ranges={"x": (3.0, 5.0), "y": (3.0, 5.0)},
    ),
    _FallbackCase(
        id="PF-01", family="multi_domain_breadth", protocol_source="all30", domain="mechanics",
        description="inverse-square: k / r^2",
        ground_truth_fn=lambda X: 10.0 / (X[:, 0] ** 2),
        var_names=["r"],
        var_ranges={"r": (1, 5)},
        extrap_ranges={"r": (5, 10)},
    ),
    _FallbackCase(
        id="NG-01", family="nguyen_polynomial", protocol_source="nguyen12", domain="nguyen_polynomial",
        description="Nguyen-4 shaped: x^6 + x^5 + x^4 + x^3 + x^2 + x",
        ground_truth_fn=lambda X: sum(X[:, 0] ** k for k in range(1, 7)),
        var_names=["x"],
        var_ranges={"x": (-1, 1)},
        extrap_ranges={"x": (1, 2)},
    ),
    _FallbackCase(
        id="NG-02", family="nguyen_transcendental", protocol_source="nguyen12", domain="nguyen_transcendental",
        description="Nguyen-7 shaped: log(x+1) + log(x^2+1)",
        ground_truth_fn=lambda X: np.log(X[:, 0] + 1) + np.log(X[:, 0] ** 2 + 1),
        var_names=["x"],
        var_ranges={"x": (0, 2)},
        extrap_ranges={"x": (2, 4)},
    ),
    _FallbackCase(
        id="NG-03", family="nguyen_bivariate", protocol_source="nguyen12", domain="nguyen_bivariate",
        description="Nguyen-9 shaped: sin(x) + sin(y^2)",
        ground_truth_fn=lambda X: np.sin(X[:, 0]) + np.sin(X[:, 1] ** 2),
        var_names=["x", "y"],
        var_ranges={"x": (0, 2), "y": (0, 2)},
        extrap_ranges={"x": (2, 3), "y": (2, 3)},
    ),
]


class _FallbackAdapter:
    """Same interface as _ProtocolAdapter, backed by _FALLBACK_CATALOG."""

    def __init__(self, source: str):
        self.source = source
        self._cases = [c for c in _FALLBACK_CATALOG if c.protocol_source == source]

    def get_all_domains(self) -> list[str]:
        return sorted({c.domain for c in self._cases})

    def load_test_data(self, domain: str, num_samples: Optional[int] = None,
                        seed: Optional[int] = None) -> list[tuple]:
        return [c.as_test_tuple(num_samples, seed or 42)
                for c in self._cases if c.domain == domain]


def build_adapters() -> dict[str, "_ProtocolAdapter"]:
    """Real adapters where importable, fallback stand-ins for the rest.
    Prints which mode each protocol is running in so a partial-coverage
    run is never silently mistaken for a full one.
    """
    adapters = _load_real_adapters()
    for source in ("defi", "feynman", "nguyen12", "all30"):
        if source not in adapters:
            print(f"[hybrid-protocol] {source}: using built-in fallback catalog "
                  f"(review/demo mode, not the real case count)")
            adapters[source] = _FallbackAdapter(source)
    return adapters


# ═══════════════════════════════════════════════════════════════════════
# 3. Capability manifest -- all 8 variants
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class VariantCapability:
    variant_id: str
    label: str
    native_entry_point: str
    supports_protocols: set          # subset of {"defi","feynman","nguyen12","all30"}
    supports_families: set           # families it may run on within those protocols
    needs_llm_formula: bool = False
    needs_pysr_guesses: bool = False
    note: str = ""


VARIANT_MANIFEST: list[VariantCapability] = [
    VariantCapability(
        "1", "Enhanced Hybrid DeFi (R² threshold gate)",
        "EnhancedHybridSystemDeFi",
        supports_protocols={"defi", "feynman", "all30"},
        supports_families={"ratio_product", "physics_broad", "multi_domain_breadth"},
        needs_llm_formula=True,
        note="Registry-wrapped, protocol-agnostic by interface (BaseMethod). "
             "Not run on nguyen12: that protocol's flow bypasses the registry.",
    ),
    VariantCapability(
        "2", "Hybrid All-Domains (llm_ok-first gate)",
        "HybridSystemAllDomains",
        supports_protocols={"defi", "feynman", "all30"},
        supports_families={"ratio_product", "physics_broad", "multi_domain_breadth"},
        needs_llm_formula=True,
        note="Same registry-agnostic scope as Variant 1.",
    ),
    VariantCapability(
        "3", "Ensemble LLM+NN (inverse-uncertainty x R² weighting)",
        "ensemble_llm_nn",
        supports_protocols={"defi", "feynman", "all30"},
        supports_families={"ratio_product", "physics_broad", "multi_domain_breadth"},
        needs_llm_formula=True,
        note="Standalone function, not a class; not yet in METHOD_REGISTRY "
             "(the gap this comparison protocol is meant to help close).",
    ),
    VariantCapability(
        "4", "Symbolic Engine + LLM prior (mode-dependent dispatch)",
        "SymbolicEngineWithLLM",
        supports_protocols={"defi", "feynman", "all30"},
        supports_families={"ratio_product", "physics_broad", "multi_domain_breadth"},
        needs_llm_formula=True, needs_pysr_guesses=True,
        note="PySR-seeded search exists in 'hybrid' mode, but its nguyen12 "
             "extension is an open decision (plan doc, decision #1) -- not "
             "enabled here until that's confirmed, so it is scoped like 1-3.",
    ),
    VariantCapability(
        "5", "Retry + physics fallback (HybridDiscoverySystem)",
        "HybridDiscoverySystem._discover_with_retry",
        supports_protocols={"defi", "feynman", "all30"},
        supports_families={"ratio_product", "physics_broad", "multi_domain_breadth"},
        note="Physics fallback path is off by default (enable_physics_fallback=False); "
             "SmartStructureDetector is confirmed orphaned in real source and is "
             "not exercised by this protocol either.",
    ),
    VariantCapability(
        "6", "V4 selector (DeFi candidate selection)",
        "_select_v4_candidate / _v4_hybrid_predict_and_eval",
        supports_protocols={"defi"},
        supports_families={"ratio_product"},
        note="Selector is built around DeFi-specific candidates (residual_nn, "
             "extrapolation-domain linear fallback). Porting to feynman/all30 "
             "means redeciding what 'extrapolative' means in that feature "
             "space -- treated as out of scope per Hybrid-pipeline.txt, not "
             "silently attempted.",
    ),
    VariantCapability(
        "6-pca", "V4 selector, PCA split",
        "_select_v4_candidate (PCA 40/60 split fork)",
        supports_protocols={"defi"},
        supports_families={"ratio_product"},
        note="Same selector as Variant 6; differs only in the upstream "
             "train/test split, mirroring hypatiax_defi_benchmark_v4_pca.py "
             "vs. v4.py.",
    ),
    VariantCapability(
        "7", "LLM-prior PySR seeding",
        "get_llm_prior + PySR guesses",
        supports_protocols={"nguyen12"},
        supports_families={"nguyen_polynomial", "nguyen_transcendental", "nguyen_bivariate"},
        needs_llm_formula=True, needs_pysr_guesses=True,
        note="No analog in the DeFi/Feynman pipelines (they don't run PySR at "
             "all). Nguyen-12-only, per plan doc decision #1.",
    ),
]


def variant_supports(cap: VariantCapability, meta: dict) -> bool:
    """The single fairness gate: a variant runs on a case iff it declares
    support for BOTH that case's protocol_source AND its family. Never
    forced, never inferred -- this is the whole mechanism."""
    return (meta.get("protocol_source") in cap.supports_protocols
            and meta.get("family") in cap.supports_families)


# ═══════════════════════════════════════════════════════════════════════
# 4. Per-run record + stand-in variant execution
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RunRecord:
    variant_id: str
    case_id: str
    family: str
    protocol_source: str
    seed: int
    status: str               # "ok" | "n/a" | "error"
    train_r2: float = float("nan")
    extrap_r2: float = float("nan")
    wall_clock_s: float = 0.0
    error_message: Optional[str] = None


def _r2(y_true, y_pred) -> float:
    y_pred = np.asarray(y_pred, dtype=float)
    if not np.all(np.isfinite(y_pred)):
        return float("-inf")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else float("-inf"))


def _run_variant(cap: VariantCapability, desc: str, X, y, var_names: list[str],
                  meta: dict, seed: int) -> RunRecord:
    """
    STAND-IN for calling the real variant's native entry point.

    This mirrors exp_cross_variant_skeleton.py's _stub_run_variant exactly:
    a simple ridge-affine fit, so the demo produces plausible
    in-domain-good / extrapolation-poor numbers. When wiring this file
    into run_comparative_hybrid_methods.py, replace the body of this
    function with a dispatch on cap.variant_id that calls
    cap.native_entry_point's real class/function -- nothing else in this
    module (catalog, manifest, gate, reporting) needs to change.
    """
    t0 = time.time()
    X_ex = meta.get("X_extrap")
    y_ex = meta.get("y_extrap")
    try:
        X1 = np.column_stack([np.ones(len(X)), X])
        coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
        pred_tr = X1 @ coef
        train_r2 = _r2(y, pred_tr)
        extrap_r2 = float("nan")
        if X_ex is not None and y_ex is not None:
            X1e = np.column_stack([np.ones(len(X_ex)), X_ex])
            pred_ex = X1e @ coef
            extrap_r2 = _r2(y_ex, pred_ex)
        return RunRecord(
            variant_id=cap.variant_id, case_id=meta.get("id", desc[:24]),
            family=meta["family"], protocol_source=meta["protocol_source"],
            seed=seed, status="ok", train_r2=train_r2, extrap_r2=extrap_r2,
            wall_clock_s=time.time() - t0,
        )
    except Exception as e:
        return RunRecord(
            variant_id=cap.variant_id, case_id=meta.get("id", desc[:24]),
            family=meta.get("family", "?"), protocol_source=meta.get("protocol_source", "?"),
            seed=seed, status="error", error_message=str(e),
            wall_clock_s=time.time() - t0,
        )


# ═══════════════════════════════════════════════════════════════════════
# 5. Driver
# ═══════════════════════════════════════════════════════════════════════

def run_hybrid_protocol(
    adapters: dict[str, "_ProtocolAdapter"],
    manifest: list[VariantCapability] = VARIANT_MANIFEST,
    protocols: Optional[list[str]] = None,
    variant_ids: Optional[list[str]] = None,
    sample_cases: Optional[int] = None,
    seeds: tuple[int, ...] = (42,),
) -> list[RunRecord]:
    protocols = protocols or list(adapters.keys())
    manifest = [c for c in manifest if variant_ids is None or c.variant_id in variant_ids]
    records: list[RunRecord] = []

    for source in protocols:
        adapter = adapters.get(source)
        if adapter is None:
            print(f"[hybrid-protocol] skipping unknown protocol source: {source}")
            continue
        for domain in adapter.get_all_domains():
            for seed in seeds:
                cases = adapter.load_test_data(domain, seed=seed)
                if sample_cases is not None:
                    cases = cases[:sample_cases]
                for desc, X, y, var_names, meta in cases:
                    for cap in manifest:
                        if not variant_supports(cap, meta):
                            records.append(RunRecord(
                                cap.variant_id, meta.get("id", desc[:24]),
                                meta["family"], meta["protocol_source"],
                                seed, status="n/a",
                            ))
                            continue
                        records.append(_run_variant(cap, desc, X, y, var_names, meta, seed))
    return records


# ═══════════════════════════════════════════════════════════════════════
# 6. Reporting -- always per (variant, protocol_source), never pooled
# ═══════════════════════════════════════════════════════════════════════

def _robust_stats(scores: list[float]) -> dict:
    valid = [s for s in scores if np.isfinite(s)]
    if not valid:
        return dict(n=0, median=float("nan"), mean_clipped=float("nan"), pct_09=0.0)
    arr = np.array(valid)
    clipped = np.clip(arr, -1.0, 1.0)
    return dict(n=len(valid), median=float(np.median(arr)),
                mean_clipped=float(np.mean(clipped)),
                pct_09=float(np.mean(arr > 0.9) * 100))


def report(records: list[RunRecord]) -> None:
    print("\n" + "=" * 78)
    print("RESULTS -- reported per (variant, protocol_source) ONLY.")
    print("Different protocols have different denominators, noise regimes and")
    print("success thresholds; a pooled cross-protocol number is NOT computed")
    print("here on purpose (see the all_30 'FIXED denominator of 74' warning).")
    print("=" * 78)

    variants = sorted({r.variant_id for r in records})
    sources = sorted({r.protocol_source for r in records})
    for vid in variants:
        for src in sources:
            group = [r for r in records if r.variant_id == vid and r.protocol_source == src]
            if not group:
                continue
            n_total, n_na = len(group), sum(1 for r in group if r.status == "n/a")
            if n_na == n_total:
                continue  # this variant has no declared support here -- skip silently
            n_err = sum(1 for r in group if r.status == "error")
            n_ok = sum(1 for r in group if r.status == "ok")
            tr_stats = _robust_stats([r.train_r2 for r in group if r.status == "ok"])
            ex_stats = _robust_stats([r.extrap_r2 for r in group if r.status == "ok"])
            print(f"\nVariant {vid} · {src}  ({n_ok}/{n_total} ok, {n_na} n/a, {n_err} error)")
            print(f"  train_r2  : median={tr_stats['median']:+.4f}  "
                  f"clipped_mean={tr_stats['mean_clipped']:+.4f}  >0.9: {tr_stats['pct_09']:5.1f}%")
            print(f"  extrap_r2 : median={ex_stats['median']:+.4f}  "
                  f"clipped_mean={ex_stats['mean_clipped']:+.4f}  >0.9: {ex_stats['pct_09']:5.1f}%")


def print_coverage_matrix(adapters: dict[str, "_ProtocolAdapter"]) -> None:
    """Print the variant x protocol_source support matrix without running
    anything -- the fast way to sanity-check the gate before spending API
    budget."""
    sources = list(adapters.keys())
    header = "Variant".ljust(28) + "".join(s.ljust(12) for s in sources)
    print(header)
    print("-" * len(header))
    for cap in VARIANT_MANIFEST:
        row = f"{cap.variant_id} {cap.label}"[:27].ljust(28)
        for s in sources:
            row += ("YES" if s in cap.supports_protocols else "n/a").ljust(12)
        print(row)


def describe() -> None:
    print(__doc__)
    print("\nVariant manifest:\n")
    for cap in VARIANT_MANIFEST:
        print(f"  [{cap.variant_id}] {cap.label}")
        print(f"        entry point : {cap.native_entry_point}")
        print(f"        protocols   : {sorted(cap.supports_protocols)}")
        print(f"        families    : {sorted(cap.supports_families)}")
        flags = []
        if cap.needs_llm_formula:
            flags.append("needs_llm_formula")
        if cap.needs_pysr_guesses:
            flags.append("needs_pysr_guesses")
        if flags:
            print(f"        flags       : {', '.join(flags)}")
        if cap.note:
            print(f"        note        : {cap.note}")
        print()


# ═══════════════════════════════════════════════════════════════════════
# 7. CLI
# ═══════════════════════════════════════════════════════════════════════

def _parse_csv(value: Optional[str]) -> Optional[list[str]]:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-protocol, capability-gated "
                                                   "comparison for all 8 HypatiaX variants.")
    parser.add_argument("--describe", action="store_true",
                         help="Print the protocol docstring and full variant manifest, then exit.")
    parser.add_argument("--list-coverage", action="store_true",
                         help="Print the variant x protocol support matrix, then exit.")
    parser.add_argument("--run", action="store_true", help="Run the comparison and print a report.")
    parser.add_argument("--protocols", type=str, default=None,
                         help="Comma-separated subset of defi,feynman,nguyen12,all30 (default: all).")
    parser.add_argument("--variants", type=str, default=None,
                         help="Comma-separated subset of variant ids, e.g. 1,3,6,6-pca,7 (default: all).")
    parser.add_argument("--sample-cases", type=int, default=None,
                         help="Cap cases per domain (smoke-scale runs; default: no cap).")
    parser.add_argument("--seeds", type=str, default="42",
                         help="Comma-separated seed list (default: 42).")
    args = parser.parse_args()

    if args.describe:
        describe()
        return

    adapters = build_adapters()

    if args.list_coverage:
        print_coverage_matrix(adapters)
        return

    if args.run or not (args.describe or args.list_coverage):
        seeds = tuple(int(s) for s in _parse_csv(args.seeds) or ["42"])
        records = run_hybrid_protocol(
            adapters,
            protocols=_parse_csv(args.protocols),
            variant_ids=_parse_csv(args.variants),
            sample_cases=args.sample_cases,
            seeds=seeds,
        )
        report(records)


if __name__ == "__main__":
    main()
