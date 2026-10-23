"""V4 (LLM prior + search, 4 modes), V5 (retry loop around one search engine), V7 (LLM-seeded search vs unseeded).
All three run through stack.search: real PySR when installed, the clearly-labelled substitute otherwise."""
from __future__ import annotations
import re
import numpy as np

from ..common import FitResult, Task, r2, try_predict, equation_ok
from ..hypotheses import propose, best_usable
from .base import Variant, nan_predict


def _llm_predict(eq, names):
    def f(X):
        p = try_predict(eq, X, names)
        return p if p is not None else nan_predict(len(X))
    return f


def _require_guesses(stack, guesses):
    if guesses and not stack.search.supports_guesses:
        raise RuntimeError("search engine cannot take `guesses`: refusing to run a 'seeded' method unseeded (audit A-020)")


# ------------------------------------------------------------------ V4
class V4SymbolicWithLLM(Variant):
    """SymbolicEngineWithLLM. mode: none | seed | hybrid | fallback.
       hybrid : LLM first. R2_llm > hybrid_llm_only -> LLM only (search skipped);
                R2_llm >= hybrid_use_prior -> search seeded with the formula, keep the better; else plain search, keep the better.
       fallback: search first; R2 > fallback_search_ok -> done; else ask the LLM and keep the better.
    Deviation from the standalone script: seeds are the LLM equation itself (a guess), not the library terms that
    happen to occur as substrings of it (the substring test matched 'x' inside 'exp')."""
    id, name = "V4", "LLM-prior / search family"

    def fit(self, task: Task, stack) -> FitResult:
        mode = self.p("mode", "hybrid")
        t_only, t_prior, t_ok = self.p("hybrid_llm_only", 0.95), self.p("hybrid_use_prior", 0.5), self.p("fallback_search_ok", 0.90)
        unary = self.p("unary", ("sq", "sqrt", "log", "inv", "sin", "cos"))
        calls = 0

        def search(guesses=None):
            nonlocal calls
            _require_guesses(stack, guesses)
            calls += 1
            return stack.search.run(task.X, task.y, task.names, seed=task.seed, guesses=guesses, unary=unary)

        def from_search(r, tag):
            return FitResult(self.id, tag, r.predict, r.r2, equation=r.expression, engine=r.engine,
                             info={"mode": mode, "selects_on": "training_r2", **r.info})

        def from_llm(h, tag):
            return FitResult(self.id, tag, _llm_predict(h.equation, task.names), h.r2, equation=h.equation,
                             info={"mode": mode, "selects_on": "training_r2"})

        if mode == "none":
            res = from_search(search(), "none")
        elif mode == "seed":
            h = best_usable(propose(stack, task, 1))
            res = from_search(search([h.equation] if h else None), "seed" if h else "seed_llm_failed")
        elif mode == "hybrid":
            h = best_usable(propose(stack, task, 1))
            if h is None:
                res = from_search(search(), "hybrid_llm_failed")
            elif h.r2 > t_only:
                res = from_llm(h, "hybrid_llm_only")
            else:
                r = search([h.equation] if h.r2 >= t_prior else None)
                res = from_search(r, "hybrid_search_better") if r.r2 > h.r2 else from_llm(h, "hybrid_llm_better")
        elif mode == "fallback":
            r = search()
            if r.r2 > t_ok:
                res = from_search(r, "fallback_search_only")
            else:
                h = best_usable(propose(stack, task, 1))
                if h is None:
                    res = from_search(r, "fallback_both_failed")
                elif h.r2 > r.r2:
                    res = from_llm(h, "fallback_llm_better")
                else:
                    res = from_search(r, "fallback_search_better")
        else:
            raise ValueError(f"unknown V4 mode {mode!r}")
        res.search_calls = calls
        return res


# ------------------------------------------------------------------ V5
def check_expression_quality(expression: str, r2_value: float, complexity_threshold: int = 20) -> dict:
    """As in the real system: 'complexity' is len(expression) -- the printed string, not a node count."""
    complexity = len(expression)
    return {"is_overfit": complexity > complexity_threshold and round(r2_value, 6) < 0.999, "complexity": complexity}


def detect_rational_pattern(X: np.ndarray, y: np.ndarray) -> bool:
    """Lineweaver-Burk linearisation (1/y vs 1/x, R2>0.85) or a saturating shape. Loose heuristics: it fires on
    most smooth monotonic positive data (confirmed in the standalone port)."""
    if X.shape[1] < 1 or np.any(y <= 0):
        return False
    try:
        inv_y = 1.0 / y
        for i in range(X.shape[1]):
            xi = X[:, i]
            if np.any(xi <= 0):
                continue
            inv_x = 1.0 / xi
            if r2(inv_y, np.polyval(np.polyfit(inv_x, inv_y, 1), inv_x)) > 0.85:
                return True
        for i in range(X.shape[1]):
            ys = y[np.argsort(X[:, i])]
            if ys[-1] > ys[0]:
                d = np.diff(ys)
                if np.all(d >= -1e-6) and d[-1] < d[0] * 0.3:
                    return True
    except Exception:
        return False
    return False


def physics_fallback(X, y, names):
    """SUBSTITUTE for the real 2117-line GP PhysicsAwareRegressor: best of linear / quadratic / log-log power law / log-linear exponential."""
    cands = []
    A = np.column_stack([np.ones(len(y)), X]); c, *_ = np.linalg.lstsq(A, y, rcond=None)
    cands.append(("linear", lambda Z, c=c: np.column_stack([np.ones(len(Z)), Z]) @ c))
    Q = lambda Z: np.column_stack([np.ones(len(Z)), Z, Z ** 2])
    c2, *_ = np.linalg.lstsq(Q(X), y, rcond=None)
    cands.append(("quadratic", lambda Z, c=c2: Q(Z) @ c))
    if np.all(X > 0) and np.all(y > 0):
        L = lambda Z: np.column_stack([np.ones(len(Z)), np.log(Z)])
        c3, *_ = np.linalg.lstsq(L(X), np.log(y), rcond=None)
        cands.append(("power_law", lambda Z, c=c3: np.exp(L(Z) @ c)))
    if np.all(y > 0):
        c4, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
        cands.append(("exponential", lambda Z, c=c4: np.exp(np.column_stack([np.ones(len(Z)), Z]) @ c)))
    scored = [(r2(y, f(X)), n, f) for n, f in cands]
    s, n, f = max(scored, key=lambda t: t[0])
    return s, n, f


class V5RetryDiscovery(Variant):
    """HybridDiscoverySystem (hybrid_system_v50_2.py): retry ONE search engine up to max_retries times.
    No LLM call in this extraction (confirm against the real class). Search budget is up to max_retries x V4's.
    seed_mode='fixed42' (DEFAULT, mirrors the documented code and DESIGN.md s2): attempt seed = 42 + attempt, so every
        experiment seed explores the same 5 seeds and a 5-seed sweep of V5 is NOT 5 independent runs.
    seed_mode='task': attempt seed = task.seed + attempt. A deviation from the documented code: only use it as a
        declared, logged decision (it changes the comparability key)."""
    id, name = "V5", "Retry loop + optional physics fallback"

    def fit(self, task: Task, stack) -> FitResult:
        max_retries = self.p("max_retries", 5)
        early = self.p("early_stop_r2", 0.9999 if self.p("transcendental", False) else 0.95)
        success_r2, cplx = self.p("success_r2", 0.97), self.p("complexity_penalty_threshold", 20)
        fb_on, fb_thr = self.p("enable_physics_fallback", False), self.p("physics_fallback_threshold", 0.85)
        base = task.seed if self.p("seed_mode", "fixed42") == "task" else 42
        unary, inv_injected = ["sq", "sqrt", "log"], False
        best, best_r2, attempts, stop = None, -np.inf, 0, "retries_exhausted"

        for attempt in range(max_retries):
            r = stack.search.run(task.X, task.y, task.names, seed=base + attempt, unary=tuple(unary))
            attempts += 1
            q = check_expression_quality(r.expression, r.r2, cplx)
            if r.r2 > best_r2:
                best, best_r2 = r, r.r2
            if attempt == 0 and r.r2 < 0.1 and not inv_injected and detect_rational_pattern(task.X, task.y):
                unary.append("inv"); inv_injected = True
            if r.r2 >= early and not q["is_overfit"]:
                stop = "early_stop"
                break
        engine = "symbolic"
        predict, expr, final_r2 = best.predict, best.expression, best_r2
        if stop != "early_stop" and fb_on and best_r2 < fb_thr:
            s, kind, f = physics_fallback(task.X, task.y, task.names)
            if s > best_r2:
                predict, expr, final_r2, engine, stop = f, f"physics_substitute:{kind}", s, "physics_aware", "physics_fallback"
        return FitResult(self.id, stop, predict, final_r2, equation=expr, engine=best.engine, search_calls=attempts,
                         info={"attempts": attempts, "success": final_r2 >= success_r2, "discovery_engine": engine,
                               "inv_injected": inv_injected, "seed_mode": self.p("seed_mode", "fixed42"),
                               "selects_on": "training_r2"})


# ------------------------------------------------------------------ V7
class V7LLMSeededSearch(Variant):
    """get_llm_prior -> search `guesses`; the unseeded run P on the same case is recorded for the H-vs-P comparison.
    Fails fast if the engine cannot seed. Costs two searches + one LLM call per case."""
    id, name = "V7", "LLM-prior search seeding"

    def fit(self, task: Task, stack) -> FitResult:
        if not stack.search.supports_guesses:
            raise RuntimeError("[FIX-LLM-WARMSTART-CTOR] engine has no guess-seeding: refusing to run H unseeded")
        unary = self.p("unary", ("sq", "sqrt", "log", "inv", "sin", "cos"))
        guesses, requested, llm_error = [], 0, None
        try:
            hyps = propose(stack, task, self.p("n_candidates", 5))
            requested = len(hyps)
            guesses = [h.equation for h in hyps if equation_ok(h.equation, task.names)]
        except Exception as e:                       # real code: any LLM failure -> cold run for this case
            llm_error = f"{type(e).__name__}: {e}"
        h = stack.search.run(task.X, task.y, task.names, seed=task.seed, guesses=guesses or None, unary=unary)
        p = stack.search.run(task.X, task.y, task.names, seed=task.seed, guesses=None, unary=unary)
        return FitResult(self.id, "seeded" if guesses else "seeded_but_no_valid_guess", h.predict, h.r2,
                         equation=h.expression, engine=h.engine, search_calls=2,
                         info={"candidates_requested": requested, "candidates_seeded": len(guesses),
                               "unseeded_r2": p.r2, "unseeded_expression": p.expression,
                               "H_beat_P": h.r2 > p.r2, "llm_error": llm_error, "selects_on": "none (H always returned)"})
