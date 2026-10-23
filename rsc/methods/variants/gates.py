"""V1, V2, V3: gates that choose between / blend an LLM formula and an NN using TRAINING-set R2.
Provenance: re-implementations of the documented logic (not diffed against the real classes)."""
from __future__ import annotations
import numpy as np

from ..common import FitResult, Task, r2, try_predict, affine_refit, inverse_std_weights
from ..hypotheses import propose, best_usable
from .base import Variant, nan_predict


class V1EnhancedHybridDeFi(Variant):
    """EnhancedHybridSystemDeFi: affine-refit the LLM formula; gate on the refit's training R2.
         fitted_r2 >= fitted_threshold                  -> fitted LLM
         fitted_r2 > nn_floor and nn_r2 > nn_floor      -> inverse-residual-std ensemble of fitted LLM and NN
         otherwise                                      -> NN"""
    id, name = "V1", "Enhanced Hybrid DeFi"

    def fit(self, task: Task, stack) -> FitResult:
        thr, floor = self.p("fitted_threshold", 0.97), self.p("nn_floor", 0.0)
        h = best_usable(propose(stack, task, n=1))
        nn = stack.new_nn(task).fit(task.X, task.y)
        nn_tr = nn.predict(task.X); nn_r2 = r2(task.y, nn_tr)
        if h is not None:
            fitted_tr, a, b = affine_refit(h.pred, task.y)
            fitted_r2 = r2(task.y, fitted_tr)
        else:
            fitted_tr, a, b, fitted_r2 = None, 1.0, 0.0, float("-inf")
        eq = h.equation if h else None

        if fitted_r2 >= thr:
            decision, w_f = "fitted_llm", 1.0
        elif fitted_r2 > floor and nn_r2 > floor:
            decision = "ensemble"
            w_f, _ = inverse_std_weights(task.y - fitted_tr, task.y - nn_tr)
        else:
            decision, w_f = "nn", 0.0

        def predict(X):
            if decision == "nn":
                return nn.predict(X)
            fit = a * try_predict(eq, X, task.names) + b
            return fit if decision == "fitted_llm" else w_f * fit + (1 - w_f) * nn.predict(X)

        train_r2 = {"fitted_llm": fitted_r2, "nn": nn_r2}.get(decision)
        if train_r2 is None:
            train_r2 = r2(task.y, w_f * fitted_tr + (1 - w_f) * nn_tr)
        return FitResult(self.id, decision, predict, train_r2, equation=eq,
                         info={"llm_r2": h.r2 if h else float("-inf"), "fitted_r2": fitted_r2, "nn_r2": nn_r2,
                               "refit_a": a, "refit_b": b, "w_fitted": w_f, "selects_on": "training_r2"})


class V2HybridAllDomains(Variant):
    """HybridSystemAllDomains: LLM unless the NN beats it by more than tie_eps on TRAINING R2 (ties favour the formula)."""
    id, name = "V2", "Hybrid All-Domains"

    def fit(self, task: Task, stack) -> FitResult:
        eps, nn_fail_r2 = self.p("tie_eps", 1e-3), self.p("nn_fallback_r2", 0.90)
        force = self.p("force_llm", False)
        h = best_usable(propose(stack, task, n=1))
        nn = stack.new_nn(task).fit(task.X, task.y)
        nn_r2 = r2(task.y, nn.predict(task.X))
        llm_r2 = h.r2 if h else float("-inf")
        eq = h.equation if h else None
        # `h is not None` matters: -inf >= -inf - eps is True, which would pick a formula that does not exist.
        if force:
            decision, reason = "llm", "forced"
        elif h is not None and llm_r2 >= nn_r2 - eps:
            decision, reason = "llm", "symbolic_preferred (tie favours symbolic)"
        else:
            decision = "nn"
            reason = "NN fallback (LLM failed)" if llm_r2 <= nn_fail_r2 else "NN better"

        def predict(X):
            if decision == "nn":
                return nn.predict(X)
            p = try_predict(eq, X, task.names)
            return p if p is not None else nan_predict(len(X))

        return FitResult(self.id, decision, predict, llm_r2 if decision == "llm" else nn_r2, equation=eq,
                         info={"reason": reason, "llm_r2": llm_r2, "nn_r2": nn_r2,
                               # the source's label fires whenever NN wins and LLM<=0.90; this is the true flag
                               "llm_failed": h is None, "selects_on": "training_r2"})


class V3Ensemble(Variant):
    """ensemble_llm_nn: inverse-residual-std blend of the RAW LLM formula and the NN (no gate, no refit).
    Fallback to max(LLM, NN) by training R2 if the weighting raises (kept for fidelity; practically unreachable)."""
    id, name = "V3", "Ensemble LLM+NN"

    def fit(self, task: Task, stack) -> FitResult:
        h = best_usable(propose(stack, task, n=1))
        nn = stack.new_nn(task).fit(task.X, task.y)
        nn_tr = nn.predict(task.X); nn_r2 = r2(task.y, nn_tr)
        if h is None:                                   # no formula to blend: the original would have blended zeros
            return FitResult(self.id, "nn_llm_unavailable", nn.predict, nn_r2,
                             info={"llm_failed": True, "nn_r2": nn_r2, "selects_on": "training_r2"})
        eq = h.equation
        try:
            if self.p("force_fallback", False):
                raise RuntimeError("forced fallback")
            w_l, w_n = inverse_std_weights(task.y - h.pred, task.y - nn_tr)
            mode = "weighted_blend"
        except Exception:
            mode = "fallback_max_llm" if h.r2 >= nn_r2 else "fallback_max_nn"
            w_l, w_n = (1.0, 0.0) if mode == "fallback_max_llm" else (0.0, 1.0)

        def predict(X):
            p = try_predict(eq, X, task.names)
            if w_l > 0 and p is None:
                return nan_predict(len(X))
            return (w_l * p if w_l > 0 else 0.0) + (w_n * nn.predict(X) if w_n > 0 else 0.0)

        return FitResult(self.id, mode, predict, r2(task.y, w_l * h.pred + w_n * nn_tr), equation=eq,
                         info={"w_llm": w_l, "w_nn": w_n, "llm_r2": h.r2, "nn_r2": nn_r2, "selects_on": "training_r2"})
