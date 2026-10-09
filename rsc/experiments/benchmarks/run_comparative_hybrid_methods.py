#!/usr/bin/env python3
"""
run_comparative_hybrid_methods.py
===================================
Orchestrator that runs all 9 registry entries (methods 1-6 already wrapped
in run_comparative_suite_benchmark_v2.py, plus 3 new entries for Variants
3, 6/6-PCA, and 7) against any of the 4 protocols, gated by the same
capability manifest already proven out in exp_cross_variant_skeleton.py
and shipped in experiment_protocol_hybrid.py.

This is "Tier 4" from the architecture discussion: the only new code here
is (a) three wrapper classes that delegate to each variant's real,
verified entry point, and (b) a thin CLI/loop that reuses everything else
unchanged:

    catalog + capability gate  <- experiment_protocol_hybrid.py (unchanged)
    6 already-registered methods, BaseMethod/MethodResult, _safe_r2,
    outer-timeout/_ProcBox subprocess isolation, comparison-table printer
                               <- run_comparative_suite_benchmark_v2.py
                                  (or _pca.py with --pca), imported, not
                                  copy-pasted
    3 new wrapper classes      <- delegate to variant3_ensemble_example.py,
                                  variant6_validation_selected_hybrid.py,
                                  variant7_llm_prior_pysr_seeding.py

Registry mapping (deliberately keeps 1-6 exactly as they are in
run_comparative_suite_benchmark_v2.py so existing exp1/exp1b/exp2 results
stay comparable; 7/8/9 are the new gap this file closes):

    1  PureLLMBaseline        (baseline, not a "variant")
    2  ImprovedNN             (baseline, not a "variant")
    3  HybridDeFiMethod       = Variant 1  (R^2 threshold gate)
    4  HybridAllDomainsMethod = Variant 2  (llm_ok-first gate)
    5  SymbolicEngineMethod   = Variant 4  (mode-dependent LLM+PySR dispatch)
    6  HybridSystemV50_2Method= Variant 5  (retry + physics fallback)
    7  EnsembleLLMNNMethod    = Variant 3  (inverse-residual-std blend)      [NEW]
    8  V4SelectorMethod       = Variant 6 / 6-PCA (validation-selected)     [NEW]
    9  LLMPriorSeedingMethod  = Variant 7  (PySR population seeding)        [NEW]

Capability gating (WHY methods 8 and 9 are protocol-restricted, not a bug):
    Method 8 (V4Selector) only declares support for the "defi" protocol in
    experiment_protocol_hybrid.py's VARIANT_MANIFEST -- its candidate pool
    (residual_nn, linear_fallback, blend) is built around DeFi-specific
    extrapolation guards. Method 9 (LLMPriorSeeding) only declares support
    for "nguyen12" -- it seeds a search engine's initial population, and
    neither the defi nor feynman pipelines run a seedable search at all.
    This file enforces that at the CLI level (see _filter_methods_for_protocol):
    requesting an unsupported (method, protocol) pair is a hard error, not a
    silent skip, so a misconfigured CI matrix fails loudly instead of quietly
    running fewer methods than requested.

Usage
-----
    # Defi, methods 1-9 minus the ones that don't apply (auto-filtered)
    python run_comparative_hybrid_methods.py --protocol defi --methods 1 2 3 4 5 6 7 8

    # Nguyen-12, Variant 7 only
    python run_comparative_hybrid_methods.py --protocol nguyen12 --methods 9

    # Feynman/benchmark_v2, all methods that support it (1-7)
    python run_comparative_hybrid_methods.py --protocol feynman --methods 1 2 3 4 5 6 7

    # PCA split fork of the defi backend (decision #3: pure fork assumption)
    python run_comparative_hybrid_methods.py --protocol defi --pca --methods 3 8

    # Smoke scale: 2 cases per domain, single seed
    python run_comparative_hybrid_methods.py --protocol defi --sample-cases 2 --seeds 42
"""
from __future__ import annotations

import argparse
import importlib
import json
import os as _os
import pathlib as _pathlib
import sys as _sys
import time
from datetime import datetime
from typing import Optional

import numpy as np

# ── sys.path bootstrap (same convention as every experiment_protocol_*.py
#    and run_comparative_suite_benchmark_v2.py) ─────────────────────────
_THIS_DIR  = _pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _pathlib.Path(_os.environ.get("REPRO_ROOT", str(_THIS_DIR.parent.parent.parent)))
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "hypatiax"),
           str(_THIS_DIR), str(_THIS_DIR / "variants"),
           str(_REPO_ROOT / "hypatiax" / "protocols")]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _p

# ── catalog + capability gate: reused, not reimplemented ────────────────
from experiment_protocol_hybrid import (
    build_adapters, VARIANT_MANIFEST, variant_supports,
)

# Map this file's registry indices (7, 8, 9) onto the variant_id strings
# experiment_protocol_hybrid.py's VARIANT_MANIFEST already uses. 1-6 map to
# baselines / variants that aren't gated here because they're already
# proven protocol-agnostic (registry-wrapped) per hybrid_ci_pipeline_plan.md.
#
# Method 8 is special: the manifest carries TWO rows for it -- "6" (plain
# V4 selector) and "6-pca" (its PCA-split fork). Which row applies depends
# on whether --pca was passed, so this can't be a flat dict lookup (that
# was the bug: it silently always resolved to "6", even under --pca --
# harmless today only because both rows happen to declare identical
# supports_protocols={"defi"}; see Hybrid-pipeline.txt finding #2).
# _variant_id_for() is the single place that resolves idx -> manifest row,
# so every caller (the CLI gate and the default-method filter) agrees.
_REGISTRY_TO_VARIANT_ID = {7: "3", 8: "6", 9: "7"}
_CAP_BY_VARIANT_ID = {c.variant_id: c for c in VARIANT_MANIFEST}


def _variant_id_for(idx: int, use_pca: bool) -> str:
    """Resolve a registry index to its VARIANT_MANIFEST id, accounting for
    method 8's PCA/non-PCA split. Returns "" for indices that aren't gated
    (1-6), matching the old dict's .get(idx, "") default."""
    if idx == 8 and use_pca:
        return "6-pca"
    return _REGISTRY_TO_VARIANT_ID.get(idx, "")


def _filter_methods_for_protocol(method_indices: list[int], protocol: str, use_pca: bool) -> list[int]:
    """Hard-fail (not silently drop) if a requested method has no declared
    capability for this protocol. Only indices 7-9 are gated here -- 1-6
    are the already-registered, protocol-agnostic methods."""
    bad = []
    for idx in method_indices:
        cap = _CAP_BY_VARIANT_ID.get(_variant_id_for(idx, use_pca))
        if cap is not None and protocol not in cap.supports_protocols:
            bad.append((idx, cap.label, sorted(cap.supports_protocols)))
    if bad:
        lines = "\n".join(f"  method {i} ({label}) only supports: {supported}"
                           for i, label, supported in bad)
        raise SystemExit(
            f"[run_comparative_hybrid_methods] Refusing to run -- requested "
            f"method(s) have no declared capability for protocol '{protocol}':\n{lines}\n"
            f"This is the fairness gate from experiment_protocol_hybrid.py's "
            f"VARIANT_MANIFEST, not a bug. Drop these methods from --methods, "
            f"or pick a protocol they actually support."
        )
    return method_indices


# ═══════════════════════════════════════════════════════════════════════
# 1. Dynamic backend import: run_comparative_suite_benchmark_v2.py (default)
#    or _pca.py (--pca). Both are treated as "pure forks" per decision #3
#    (confirmed structurally identical class-for-class; NOT byte-verified
#    against every internal helper -- see experiment_protocol_hybrid.py's
#    --list-experiments "YES*" footnote for the same caveat).
# ═══════════════════════════════════════════════════════════════════════

def _load_backend(use_pca: bool):
    mod_name = "run_comparative_suite_benchmark_pca" if use_pca else "run_comparative_suite_benchmark_v2"
    mod = importlib.import_module(mod_name)
    # ImprovedNNMethod is named ImprovedNN in the PCA fork -- confirmed by
    # diffing the two files' top-level class lists (see conversation for
    # the diff); handled here rather than assumed silently.
    improved_nn_cls = getattr(mod, "ImprovedNNMethod", None) or getattr(mod, "ImprovedNN")
    return mod, improved_nn_cls


# ═══════════════════════════════════════════════════════════════════════
# 2. Three new wrapper classes -- Variants 3, 6/6-PCA, 7
# ═══════════════════════════════════════════════════════════════════════

def _build_pure_pysr_class(backend_mod):
    """PATCH 0003 (audit A-009): baseline B3 = V4's engine with llm_mode='none' (pure PySR, no LLM).
    Refuses to run if the backend class cannot be put in 'none' mode -- never silently runs an LLM mode."""
    Base = backend_mod.SymbolicEngineMethod

    class PureSymbolicPySRMethod(Base):
        def __init__(self, verbose: bool = False):
            try:
                super().__init__(verbose=verbose, llm_mode="none")
            except TypeError:
                super().__init__(verbose=verbose)
                if not hasattr(self, "llm_mode"):
                    raise RuntimeError("SymbolicEngineMethod exposes no llm_mode: cannot build pure-PySR baseline (A-009)")
                self.llm_mode = "none"
            if getattr(self, "llm_mode", "none") != "none":
                raise RuntimeError(f"baseline B3 must run llm_mode='none', got {self.llm_mode!r}")
            self.name = "PureSymbolicPySR(B3)"
    return PureSymbolicPySRMethod


def _build_new_wrapper_classes(backend_mod):
    """Factory so the three new wrapper classes can subclass whichever
    BaseMethod/MethodResult the selected backend (v2 or pca) exports,
    keeping _safe_r2 / _safe_rmse / _runner_eval_formula behaviour
    identical to the six already-registered methods regardless of which
    backend is active."""

    BaseMethod   = backend_mod.BaseMethod
    MethodResult = backend_mod.MethodResult

    # ── Variant 3 : ensemble_llm_nn ─────────────────────────────────────
    from variant3_ensemble_example import (
        EnsembleSystem, MockLLMProvider, AnthropicProvider,
    )

    def _default_provider():
        if _os.environ.get("ANTHROPIC_API_KEY"):
            return AnthropicProvider()
        return MockLLMProvider(fixtures={})   # degrades exactly like the real code with llm_code == ""

    class EnsembleLLMNNMethod(BaseMethod):
        """Variant 3 -- inverse-residual-std weighted blend of LLM formula
        and NN. Delegates to variant3_ensemble_example.EnsembleSystem,
        which wraps the real ensemble_llm_nn() gate. Was not previously in
        METHOD_REGISTRY (the gap this file closes, per plan doc Phase 1)."""

        def __init__(self, verbose: bool = False, llm_provider=None,
                     nn_backend: str = "auto", nn_epochs: int = 300):
            super().__init__(name="EnsembleLLMNN(V3)", verbose=verbose)
            self.provider = llm_provider or _default_provider()
            self.nn_backend = nn_backend
            self.nn_epochs = nn_epochs

        def run(self, description, X, y, var_names, metadata, verbose=False) -> "MethodResult":
            t0 = time.time()
            try:
                system = EnsembleSystem(self.provider, self.nn_backend, self.nn_epochs,
                                         verbose=verbose)
                case_id = metadata.get("id", description[:24])
                seed = int(metadata.get("seed", 42))
                out = system.fit_predict(X, y, var_names, description, case_id, seed)
                predict_fn = out["predict"]
                X_far, y_far = metadata.get("X_extrap"), metadata.get("y_extrap")
                if X_far is not None and y_far is not None:
                    y_pred = predict_fn(X_far)
                    r2v, rmse = self._safe_r2(y_far, y_pred), self._safe_rmse(y_far, y_pred)
                else:
                    y_pred = predict_fn(X)
                    r2v, rmse = self._safe_r2(y, y_pred), self._safe_rmse(y, y_pred)
                disp, fhash, full = self._make_formula_result(
                    f"blend[{out['mode']}] w_llm={out['w_llm']:.3f}: {out['llm_equation']}")
                return MethodResult(method=self.name, success=True, r2=r2v, rmse=rmse,
                                     formula=disp, formula_hash=fhash, formula_full=full,
                                     time=time.time() - t0,
                                     metadata={"mode": out["mode"], "w_llm": out["w_llm"],
                                               "w_nn": out["w_nn"], "llm_r2": out["llm_r2"],
                                               "nn_r2": out["nn_r2"]})
            except Exception as e:
                return self._unavailable(f"EnsembleLLMNNMethod failed: {e}")

    # ── Variant 6 / 6-PCA : V4 selector ─────────────────────────────────
    from variant6_validation_selected_hybrid import (
        select_v4_candidate, extrapolates,
        fit_nn_predict, fit_linear_fallback, fit_linear_fallback_local,
    )

    class V4SelectorMethod(BaseMethod):
        """Variant 6 / 6-PCA -- validation-selected residual hybrid
        (_select_v4_candidate). DeFi-only per VARIANT_MANIFEST: its pool is
        built around DeFi-specific extrapolation guards.

        NOTE (honest disclosure, same convention as the rest of this repo):
        select_v4_candidate() was verified against the real
        hypatiax_defi_benchmark_v4.py source and tells us WHICH candidate
        wins. The prediction reconstruction below (refit the winner on the
        full training set, predict on the extrap slice) mirrors the shape
        of the real `_v4_hybrid_predict_and_eval`, but that private
        function's exact body was not available to verify line-for-line --
        treat extrap_r2 from this method as directionally correct, not
        byte-identical to the real benchmark's reported figures.
        """

        def __init__(self, verbose: bool = False, llm_predict_fn=None):
            super().__init__(name="V4Selector(V6)", verbose=verbose)
            self.llm_predict_fn = llm_predict_fn  # None -> "no LLM available" pool, same as real llm_code == ""

        def run(self, description, X, y, var_names, metadata, verbose=False) -> "MethodResult":
            t0 = time.time()
            try:
                X_far, y_far = metadata.get("X_extrap"), metadata.get("y_extrap")
                is_extrap = extrapolates(X, X_far) if X_far is not None else False
                seed = int(metadata.get("seed", 42))
                sel = select_v4_candidate(X, y, self.llm_predict_fn, seed=seed,
                                           extrapolative=is_extrap)
                winner = sel["selected"]

                # Reconstruct predictions for the winning candidate.
                if winner == "llm" and self.llm_predict_fn is not None:
                    y_pred = self.llm_predict_fn(X_far if X_far is not None else X)
                elif winner in ("nn", "residual_nn"):
                    hidden = sel.get("hidden") or [64, 32]
                    base = fit_nn_predict(X, y, X_far if X_far is not None else X, hidden, seed)
                    y_pred = base
                elif winner == "linear_fallback":
                    y_pred = fit_linear_fallback(X, y, X_far if X_far is not None else X)
                elif winner == "linear_fallback_local":
                    y_pred = fit_linear_fallback_local(X, y, X_far if X_far is not None else X)
                else:  # "blend" or unexpected key -- fall back to linear rather than guess
                    y_pred = fit_linear_fallback(X, y, X_far if X_far is not None else X)

                if X_far is not None and y_far is not None:
                    r2v, rmse = self._safe_r2(y_far, y_pred), self._safe_rmse(y_far, y_pred)
                else:
                    r2v, rmse = self._safe_r2(y, y_pred), self._safe_rmse(y, y_pred)

                disp, fhash, full = self._make_formula_result(f"v4_selected:{winner}")
                return MethodResult(method=self.name, success=True, r2=r2v, rmse=rmse,
                                     formula=disp, formula_hash=fhash, formula_full=full,
                                     time=time.time() - t0,
                                     metadata={"selected": winner, "extrapolative": is_extrap,
                                               "validation_r2": sel.get("validation_r2", {}),
                                               "validation_n": sel.get("validation_n", 0)})
            except Exception as e:
                return self._unavailable(f"V4SelectorMethod failed: {e}")

    # ── Variant 7 : LLM-prior PySR/search seeding ───────────────────────
    from variant7_llm_prior_pysr_seeding import (
        run_seeded_vs_unseeded, mock_llm_prior,
    )

    class LLMPriorSeedingMethod(BaseMethod):
        """Variant 7 -- LLM-prior population seeding (get_llm_prior +
        PySRRegressor(guesses=...)). Nguyen-12-only per VARIANT_MANIFEST:
        no analog in defi/feynman (neither runs a seedable search)."""

        def __init__(self, verbose: bool = False, llm_prior_fn=None):
            super().__init__(name="LLMPriorSeeding(V7)", verbose=verbose)
            self.llm_prior_fn = llm_prior_fn or mock_llm_prior

        def run(self, description, X, y, var_names, metadata, verbose=False) -> "MethodResult":
            t0 = time.time()
            try:
                eq = {"id": metadata.get("id", description[:24]), "vars": var_names,
                      "formula_hint": metadata.get("ground_truth", "")}
                out = run_seeded_vs_unseeded(eq, X, y, llm_prior_fn=self.llm_prior_fn,
                                              engine_supports_guesses=True, use_llm=True)
                h = out["H_seeded"]
                X_far, y_far = metadata.get("X_extrap"), metadata.get("y_extrap")
                extrap_r2 = None
                if X_far is not None and y_far is not None:
                    y_pred = self._runner_eval_formula(h["expr"], X_far, var_names)
                    extrap_r2 = self._safe_r2(y_far, y_pred) if y_pred is not None else float("-inf")
                r2v = extrap_r2 if extrap_r2 is not None else h["r2"]
                disp, fhash, full = self._make_formula_result(h["expr"])
                return MethodResult(method=self.name, success=True, r2=r2v,
                                     rmse=self._safe_rmse(y, self._runner_eval_formula(
                                         h["expr"], X, var_names) if h["expr"] != "0" else np.zeros_like(y)),
                                     formula=disp, formula_hash=fhash, formula_full=full,
                                     time=time.time() - t0,
                                     metadata={"train_r2": h["r2"], "H_beat_P": out["H_beat_P"],
                                               "llm_candidates_requested": out["llm_candidates_requested"],
                                               "llm_candidates_seeded": out["llm_candidates_seeded"]})
            except Exception as e:
                return self._unavailable(f"LLMPriorSeedingMethod failed: {e}")

    return EnsembleLLMNNMethod, V4SelectorMethod, LLMPriorSeedingMethod


# ═══════════════════════════════════════════════════════════════════════
# 3. Extended suite -- registry 1-9, reusing the real backend's run_test
#    machinery for 1-6 and the three new classes for 7-9.
# ═══════════════════════════════════════════════════════════════════════

class HybridComparativeSuite:
    """Thin extension of ProtocolBenchmarkSuite's registry pattern.
    Deliberately does NOT subclass ProtocolBenchmarkSuite (that class's
    __init__ has several method-class-specific branches tuned for the
    original 6; re-deriving from it risks silently inheriting assumptions
    that don't hold for 7-9). Instead this composes: it borrows the
    backend's method classes + MethodResult/BaseMethod, and reimplements
    only the instantiation + per-test loop, which is a handful of lines.
    """

    def __init__(self, backend_mod, method_indices: Optional[list[int]] = None,
                 verbose: bool = False, no_llm_cache: bool = False, nn_seeds: int = 1):
        self.verbose = verbose
        _, ImprovedNNCls = backend_mod, None
        improved_nn_cls = getattr(backend_mod, "ImprovedNNMethod", None) or getattr(backend_mod, "ImprovedNN")
        EnsembleLLMNNMethod, V4SelectorMethod, LLMPriorSeedingMethod = _build_new_wrapper_classes(backend_mod)
        PureSymbolicPySRMethod = _build_pure_pysr_class(backend_mod)

        self.REGISTRY = [
            (1, backend_mod.PureLLMBaselineMethod),
            (2, improved_nn_cls),
            (3, backend_mod.HybridDeFiMethod),          # Variant 1
            (4, backend_mod.HybridAllDomainsMethod),     # Variant 2
            (5, backend_mod.SymbolicEngineMethod),       # Variant 4
            (6, backend_mod.HybridSystemV50_2Method),    # Variant 5
            (7, EnsembleLLMNNMethod),                    # Variant 3   [NEW]
            (8, V4SelectorMethod),                       # Variant 6/6-PCA [NEW]
            (9, LLMPriorSeedingMethod),                  # Variant 7   [NEW]
            (10, PureSymbolicPySRMethod),                # Baseline B3 [PATCH 0003]
        ]

        active = set(method_indices) if method_indices else {i for i, _ in self.REGISTRY}
        self.methods = []
        for idx, cls in self.REGISTRY:
            if idx not in active:
                continue
            if cls is backend_mod.PureLLMBaselineMethod:
                m = cls(verbose=verbose, no_cache=no_llm_cache)
            elif cls is improved_nn_cls:
                m = cls(verbose=verbose, nn_seeds=nn_seeds)
            elif cls in (backend_mod.HybridAllDomainsMethod, backend_mod.HybridDeFiMethod):
                m = cls(verbose=verbose, no_cache=no_llm_cache)
            else:
                m = cls(verbose=verbose)
            self.methods.append(m)

        print(f"\n{'=' * 80}\nHYBRID COMPARATIVE SUITE -- {len(self.methods)} active method(s)\n{'=' * 80}")
        for idx, cls in self.REGISTRY:
            flag = "\u2713" if idx in active else "-"
            print(f"  [{flag}] {idx}. {cls.__name__}")
        print(f"{'=' * 80}\n")

    def run_test(self, description, X, y, var_names, metadata, domain, verbose=True) -> dict:
        if verbose:
            print(f"\n{'-' * 80}\n  {description[:74]}\n"
                  f"  domain={domain}  family={metadata.get('family')}  "
                  f"protocol={metadata.get('protocol_source')}\n{'-' * 80}")
        row = {"description": description, "domain": domain,
               "family": metadata.get("family"), "protocol_source": metadata.get("protocol_source"),
               "seed": metadata.get("seed"),   # PATCH 0002: seed was never written (audit A-002)
               "timestamp": datetime.now().isoformat(), "results": {}}
        for m in self.methods:
            try:
                res = m.run(description, X, y, var_names, metadata, verbose=verbose)
            except Exception as e:
                res = None
                row["results"][m.name] = {"method": m.name, "success": False, "error": str(e)}
                continue
            row["results"][m.name] = res.to_dict()
            if verbose:
                print(f"    {m.name:26s} r2={res.r2:+.4f}  rmse={res.rmse:.4g}  "
                      f"{'OK' if res.success else 'FAIL: ' + str(res.error)}")
        return row


# ═══════════════════════════════════════════════════════════════════════
# 4. CLI
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-protocol comparative runner for all 9 registry entries "
                    "(6 already-registered + 3 new: Variants 3, 6/6-PCA, 7).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--protocol", choices=["defi", "feynman", "nguyen12", "all30"],
                         required=True, help="Which protocol's cases to run against.")
    parser.add_argument("--domain", type=str, default="all",
                         help="Domain key from the protocol's get_all_domains(), or 'all' (default).")
    parser.add_argument("--methods", type=int, nargs="+", default=None, metavar="N",
                         help="Registry indices to run (1-10; 10 = pure-PySR baseline B3, opt-in; default: all valid for --protocol).")
    parser.add_argument("--pca", action="store_true",
                         help="Use the PCA-split backend (run_comparative_suite_benchmark_pca.py) "
                              "instead of run_comparative_suite_benchmark_v2.py.")
    parser.add_argument("--samples", type=int, default=None, help="Data points per case (protocol default if omitted).")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seed list (default: 42).")
    parser.add_argument("--sample-cases", type=int, default=None,
                         help="Cap cases per domain (smoke-scale runs).")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="Where to write hybrid_comparison_results*.json (default: RESULTS_DIR env or cwd).")
    parser.add_argument("--no-llm-cache", action="store_true", dest="no_llm_cache")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    backend_mod, _ = _load_backend(args.pca)

    default_methods = {i for i, _ in [(1, None), (2, None), (3, None), (4, None), (5, None),
                                       (6, None), (7, None), (8, None), (9, None)]}
    requested = args.methods if args.methods else sorted(default_methods)
    # Drop (don't silently keep) any of 7/8/9 default methods that don't
    # support this protocol UNLESS the user explicitly asked for them --
    # explicit request + unsupported protocol is a hard error (see
    # _filter_methods_for_protocol); an unfiltered *default* set should
    # just quietly narrow to what's valid, matching how experiment_protocol
    # _hybrid.py's coverage matrix already documents this per protocol.
    if args.methods is None:
        requested = [i for i in requested
                     if i not in _REGISTRY_TO_VARIANT_ID
                     or args.protocol in _CAP_BY_VARIANT_ID[_variant_id_for(i, args.pca)].supports_protocols]
    else:
        _filter_methods_for_protocol(requested, args.protocol, args.pca)

    suite = HybridComparativeSuite(backend_mod, method_indices=requested, verbose=args.verbose,
                                    no_llm_cache=args.no_llm_cache)

    adapters = build_adapters()
    adapter = adapters[args.protocol]
    domains = adapter.get_all_domains() if args.domain == "all" else [args.domain]

    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    out_dir = _pathlib.Path(args.output_dir or _os.environ.get("RESULTS_DIR", ".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for domain in domains:
        for seed in seeds:
            kwargs = {"seed": seed}
            if args.samples is not None:
                kwargs["num_samples"] = args.samples
            cases = adapter.load_test_data(domain, **kwargs)
            if args.sample_cases is not None:
                cases = cases[:args.sample_cases]
            for desc, X, y, var_names, meta in cases:
                meta = dict(meta)
                meta.setdefault("seed", seed)
                row = suite.run_test(desc, X, y, var_names, meta, domain, verbose=args.verbose)
                all_rows.append(row)

    shard_tag = f"_{_os.environ.get('HYPATIAX_SHARD_ID', '').strip()}" if _os.environ.get("HYPATIAX_SHARD_ID") else ""
    pca_tag = "_pca" if args.pca else ""
    out_path = out_dir / f"hybrid_comparison_results_{args.protocol}{pca_tag}{shard_tag}.json"
    with open(out_path, "w") as f:
        json.dump({"protocol": args.protocol, "pca": args.pca, "methods": requested,
                   "seeds": list(seeds), "git_sha": _os.environ.get("GIT_SHA"),   # PATCH 0002
                   "timestamp": datetime.now().isoformat(), "n_cases": len(all_rows),
                   "rows": all_rows}, f, indent=2, default=str)
    print(f"\nWrote {len(all_rows)} case result(s) to {out_path}")


if __name__ == "__main__":
    main()
