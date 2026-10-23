import numpy as np
import pytest

from rsc.methods.common import r2, safe_predict
from rsc.methods.variants import REGISTRY, make
from rsc.methods.variants import search_variants as sv
from rsc.methods.variants import validation_selected as v6
from rsc.methods.search import SearchResult, SubstituteSearch
from methods_helpers import make_task, stack_with


# ---------------------------------------------------------------- bad LLM answers never crash any LLM-based variant
BAD = ["x +", "foo*2", "9**9**9", "__import__('os')", "", "0*x"]


@pytest.mark.parametrize("vid", ["B_LLM", "V1", "V2", "V3", "V4", "V4_seed", "V4_fallback", "V6", "V7"])
@pytest.mark.parametrize("bad", BAD)
def test_bad_llm_answer_is_survivable(vid, bad):
    task = make_task()
    res = make(vid).run(task, stack_with([bad]))
    assert res.decision and np.isfinite(res.train_r2) or res.train_r2 == float("-inf")
    safe_predict(res, task.X)                       # must not raise


@pytest.mark.parametrize("vid", ["B_LLM", "V1", "V2", "V3", "V4", "V6", "V7"])
def test_no_llm_answer_at_all(vid):
    res = make(vid).run(make_task(), stack_with([]))
    assert res.decision


@pytest.mark.parametrize("vid", list(REGISTRY))
def test_every_variant_runs_and_reports_calls(vid):
    task = make_task(fn=lambda X: 2 * X[:, 0] + X[:, 1] ** 2, names=("a", "b"), Xfar=np.full((5, 2), 8.0))
    res = make(vid).run(task, stack_with(["2*a + b**2"]))
    assert res.variant == (vid.split("_")[0] if vid.startswith("V4_") else vid)
    assert res.api_calls >= 0 and "seconds" in res.info
    assert np.isfinite(safe_predict(res, task.X)).any()


# ---------------------------------------------------------------- V1
def test_v1_near_perfect_formula_wins():
    assert make("V1").run(make_task(fn=lambda X: 3 * X[:, 0] + 2), stack_with(["3*x + 2"])).decision == "fitted_llm"


def test_v1_useless_formula_goes_to_nn_or_ensemble():
    r = make("V1").run(make_task(), stack_with(["0*x"]))
    assert r.decision in ("nn", "ensemble")


def test_v1_affine_refit_rescues_wrong_scale():
    r = make("V1").run(make_task(fn=lambda X: 3 * X[:, 0] + 2), stack_with(["x"]))   # right shape, wrong a,b
    assert r.decision == "fitted_llm" and r.info["refit_a"] == pytest.approx(3, abs=1e-6)


# ---------------------------------------------------------------- V2
def test_v2_ties_favour_symbolic_and_force():
    t = make_task()
    assert make("V2").run(t, stack_with(["4*x"])).decision == "llm"
    assert make("V2").run(t, stack_with(["0*x"])).decision == "nn"
    r = make("V2", force_llm=True).run(t, stack_with(["0*x"]))
    assert r.decision == "llm" and r.info["reason"] == "forced"


def test_v2_no_formula_never_selects_llm():
    # regression: -inf >= -inf - eps is True, so the original would pick a formula that does not exist
    r = make("V2").run(make_task(), stack_with(["x +"]))
    assert r.decision == "nn" and r.info["llm_failed"] is True


def test_v2_reason_label_vs_true_failure_flag():
    r = make("V2").run(make_task(fn=lambda X: X[:, 0] ** 2), stack_with(["x"]))   # valid but beaten formula
    assert r.decision == "nn" and r.info["llm_failed"] is False


# ---------------------------------------------------------------- V3
def test_v3_weights_sum_to_one_and_favour_better():
    r = make("V3").run(make_task(), stack_with(["4*x"]))
    assert r.info["w_llm"] + r.info["w_nn"] == pytest.approx(1.0) and r.info["w_llm"] > 0.5


def test_v3_forced_fallback_picks_argmax():
    r = make("V3", force_fallback=True).run(make_task(), stack_with(["4*x"]))
    assert r.decision == "fallback_max_llm"


def test_v3_unusable_llm_degrades_to_nn_instead_of_blending_zeros():
    assert make("V3").run(make_task(), stack_with(["x +"])).decision == "nn_llm_unavailable"


# ---------------------------------------------------------------- V4
def test_v4_modes():
    t = make_task(fn=lambda X: X[:, 0] ** 2)
    assert make("V4", mode="none").run(t, stack_with([])).api_calls == 0
    assert make("V4", mode="hybrid").run(t, stack_with(["x**2"])).decision == "hybrid_llm_only"       # R2 > 0.95: search skipped
    r = make("V4", mode="hybrid").run(t, stack_with(["0*x"]))
    assert r.decision == "hybrid_search_better" and r.search_calls == 1
    assert make("V4", mode="fallback").run(t, stack_with(["0*x"])).decision == "fallback_search_only"   # search already > 0.90, LLM not asked
    assert make("V4", mode="fallback").run(t, stack_with(["0*x"])).api_calls == 0
    with pytest.raises(ValueError):
        make("V4", mode="bogus").run(t, stack_with([]))


def test_v4_seed_without_guess_support_refuses_instead_of_running_unseeded():
    stack = stack_with(["x**2"])
    stack.search.supports_guesses = False
    with pytest.raises(RuntimeError, match="guesses"):
        make("V4", mode="seed").run(make_task(), stack)


# ---------------------------------------------------------------- V5
class _Fixed:
    engine, supports_guesses = "fixed", True
    def __init__(self, r2v, expr="x"):
        self.r2v, self.expr, self.seeds, self.unary_seen = r2v, expr, [], []
    def run(self, X, y, names, seed=42, guesses=None, unary=()):
        self.seeds.append(seed); self.unary_seen.append(tuple(unary))
        return SearchResult(self.expr, self.r2v, lambda Z: Z[:, 0], "fixed")


def _v5(stack_engine, **kw):
    stack = stack_with([]); stack.search = stack_engine
    return make("V5", **kw).run(make_task(seed=100), stack), stack_engine


def test_v5_early_stop_and_threshold_semantics():
    r, e = _v5(_Fixed(0.96))
    assert r.decision == "early_stop" and r.search_calls == 1 and r.info["success"] is False   # 0.96 stops early but < 0.97 success
    r, e = _v5(_Fixed(0.96), transcendental=True, max_retries=3)
    assert r.decision == "retries_exhausted" and r.search_calls == 3
    r, e = _v5(_Fixed(0.98))
    assert r.info["success"] is True


def test_v5_overfit_check_blocks_early_stop():
    r, e = _v5(_Fixed(0.96, expr="x" * 25), max_retries=4)      # len>20 and r2<0.999 -> 'overfit', keep retrying
    assert r.search_calls == 4


def test_v5_default_mirrors_documented_seeds_and_task_mode_is_opt_in():
    _, e = _v5(_Fixed(0.3), max_retries=3)
    assert e.seeds == [42, 43, 44]                                # documented behaviour: experiment seed does NOT reach V5
    _, e = _v5(_Fixed(0.3), max_retries=3, seed_mode="task")
    assert e.seeds == [100, 101, 102]                             # declared deviation: experiment seed + attempt


def test_v5_physics_fallback_off_by_default_then_on():
    r, _ = _v5(_Fixed(0.3), max_retries=1)
    assert r.decision == "retries_exhausted"
    stack = stack_with([]); stack.search = _Fixed(0.3)
    r = make("V5", max_retries=1, enable_physics_fallback=True).run(make_task(), stack)
    assert r.decision == "physics_fallback" and r.train_r2 > 0.99


def test_v5_makes_no_llm_calls():
    assert make("V5").run(make_task(), stack_with(["4*x"])).api_calls == 0


def test_v5_rational_probe_injects_inv_only_after_bad_first_attempt():
    rng = np.random.default_rng(0)
    S = rng.uniform(0.1, 50, (150, 1)); y = 5 * S[:, 0] / (2 + S[:, 0])
    assert sv.detect_rational_pattern(S, y)
    assert not sv.detect_rational_pattern(S, 2 + np.sin(S[:, 0]))
    e = _Fixed(0.05)
    stack = stack_with([]); stack.search = e
    from rsc.methods.common import Task
    make("V5", max_retries=2).run(Task(S, y, ["S"], "mm"), stack)
    assert "inv" not in e.unary_seen[0] and "inv" in e.unary_seen[1]


# ---------------------------------------------------------------- V6
def test_v6_validation_split_uses_high_end_of_training_range():
    X = np.arange(50, dtype=float).reshape(-1, 1); y = X[:, 0]
    Xi, yi, Xv, yv = v6.split_internal_validation(X, y)
    assert Xv.min() > Xi.max() and len(Xv) == 10


def test_v6_extrapolation_guard_drops_bare_nn():
    fn = lambda X: 2 * X[:, 0] + X[:, 1] ** 2
    near = make_task(fn=fn, names=("a", "b"), n=60, Xfar=np.full((4, 2), 3.0))       # inside the training range
    far = make_task(fn=fn, names=("a", "b"), n=60, Xfar=np.full((4, 2), 9.0))        # outside it
    r_in = make("V6").run(near, stack_with(["x +"]))                                  # no usable LLM formula
    r_out = make("V6").run(far, stack_with(["x +"]))
    assert not r_in.info["extrapolative"] and r_in.decision == "nn"
    assert r_out.info["extrapolative"] and r_out.decision in ("linear_fallback", "linear_fallback_local")
    # with a good formula the pool is LLM-anchored either way
    r_llm = make("V6").run(far, stack_with(["2*a + b**2"]))
    assert r_llm.decision in ("llm", "residual_nn", "blend")


def test_v6_use_test_features_can_be_switched_off():
    t = make_task(names=("a",), n=60, Xfar=np.full((4, 1), 9.0))
    r = make("V6", use_test_features=False).run(t, stack_with(["4*a"]))
    assert r.info["used_test_features"] is False and r.info["extrapolative"] is False


def test_v6_tiny_training_set_does_not_crash():
    t = make_task(n=8)
    assert make("V6").run(t, stack_with(["4*x"])).decision == "llm"


# ---------------------------------------------------------------- V7
def test_v7_fails_fast_without_guess_support():
    stack = stack_with(["x"]); stack.search.supports_guesses = False
    with pytest.raises(RuntimeError, match="refusing"):
        make("V7").run(make_task(), stack)


def test_v7_records_seeded_and_unseeded_and_survives_llm_error():
    t = make_task(fn=lambda X: X[:, 0] ** 3)
    r = make("V7").run(t, stack_with(["x**3", "foo", "x +"]))
    assert r.info["candidates_requested"] == 3 and r.info["candidates_seeded"] == 1 and r.search_calls == 2
    stack = stack_with([]); stack.llm.complete = lambda p: (_ for _ in ()).throw(RuntimeError("api down"))
    r2_ = make("V7").run(t, stack)
    assert "api down" in r2_.info["llm_error"] and r2_.decision == "seeded_but_no_valid_guess"


def test_substitute_guess_is_actually_used():
    rng = np.random.default_rng(1)
    X = rng.uniform(0.5, 3, (100, 1)); y = np.sin(X[:, 0]) ** 2 + 0.3 * np.exp(X[:, 0])
    eng = SubstituteSearch(max_terms=1)
    cold = eng.run(X, y, ["x"]); warm = eng.run(X, y, ["x"], guesses=["sin(x)**2 + 0.3*exp(x)"])
    assert warm.r2 > 0.9999 > cold.r2
