import json, time
import numpy as np
import pytest

from rsc.methods import common
from rsc.methods.common import Task, EquationError, predict_from_equation, try_predict, equation_ok, r2
from rsc.methods.env import ApiEnv, EnvError
from rsc.methods.llm import LLMClient, MockLLM, BudgetExceeded, TemperatureRejected
from rsc.methods.nn import NNBaseline
from methods_helpers import make_task


# ---------------------------------------------------------------- sandbox
@pytest.mark.parametrize("bad", ["__import__('os')", "().__class__", "open('x')", "x.real", "[1,2]", "'a'", "lambda: 1"])
def test_sandbox_rejects(bad):
    with pytest.raises(EquationError):
        predict_from_equation(bad, np.ones((2, 1)), ["x"])


def test_syntax_error_is_equation_error_not_syntaxerror():
    # the standalone scripts only caught ValueError; SyntaxError escaped and crashed the whole run
    with pytest.raises(EquationError):
        predict_from_equation("x +", np.ones((2, 1)), ["x"])
    assert try_predict("x +", np.ones((2, 1)), ["x"]) is None


def test_unknown_name_is_none_not_crash():
    assert try_predict("foo*2", np.ones((2, 1)), ["x"]) is None
    assert not equation_ok("foo*2", ["x"])


def test_huge_integer_power_does_not_hang():
    t0 = time.time()
    assert try_predict("9**9**9", np.ones((2, 1)), ["x"]) is None
    assert time.time() - t0 < 2.0


def test_r2_edge_cases():
    assert r2([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert r2([1, 2, 3], [1, np.nan, 3]) == float("-inf")
    assert r2([1, 1, 1], [1, 1, 1]) == 1.0 and not np.isnan(r2([1, 1, 1], [2, 2, 2]))


def test_task_has_no_ground_truth_field():
    names = set(Task.__dataclass_fields__)
    assert not any(k in names for k in ("formula", "gt", "ground_truth", "gt_formula", "y_test"))


def test_task_shape_validation():
    with pytest.raises(ValueError):
        Task(np.ones((3, 2)), np.ones(3), ["a"], "d")


# ---------------------------------------------------------------- env
def _write_cfg(tmp_path, extra=""):
    (tmp_path / "config").mkdir()
    p = tmp_path / "config" / "api_env.yml"
    p.write_text("provider: anthropic\napi_key_env: MY_KEY\ndotenv: .env\n" + extra)
    return p


def test_env_key_from_environ_then_dotenv(tmp_path, monkeypatch):
    p = _write_cfg(tmp_path)
    monkeypatch.delenv("MY_KEY", raising=False)
    env = ApiEnv.load(p)
    assert env.api_key({}) is None
    (tmp_path / ".env").write_text('# c\nMY_KEY="from-dotenv"\n')
    assert env.api_key({}) == "from-dotenv"
    assert env.api_key({"MY_KEY": "from-env"}) == "from-env"


def test_env_modes(tmp_path):
    p = _write_cfg(tmp_path, "mode: live\n")
    env = ApiEnv.load(p)
    with pytest.raises(EnvError):
        env.resolved_mode({})
    assert env.resolved_mode({"MY_KEY": "k"}) == "live"
    env.mode = "auto"
    assert env.resolved_mode({}) == "mock" and env.resolved_mode({"MY_KEY": "k"}) == "live"
    env.mode = "mock"
    assert env.resolved_mode({"MY_KEY": "k"}) == "mock"


def test_budget_without_prices_refused(tmp_path):
    with pytest.raises(EnvError):
        ApiEnv.load(_write_cfg(tmp_path, "budget_usd: 5\n"))


def test_public_dict_never_contains_key(tmp_path):
    env = ApiEnv.load(_write_cfg(tmp_path))
    assert "k-secret" not in json.dumps(env.public_dict()) and "k-secret" not in repr(env)


def test_default_config_loads():
    env = ApiEnv.load()
    assert env.model and env.mode in ("auto", "live", "mock")


# ---------------------------------------------------------------- llm client (fake SDK client)
class _Block:
    type = "text"
    def __init__(self, t): self.text = t


class _Resp:
    def __init__(self, text="[]", i=100, o=50):
        self.content = [_Block(text)]
        self.usage = type("U", (), {"input_tokens": i, "output_tokens": o})()


class FakeSDK:
    def __init__(self, reject_temperature=False):
        self.reject, self.calls = reject_temperature, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if self.reject and "temperature" in kw:
            raise RuntimeError("`temperature` is deprecated for this model")
        return _Resp()


def _env(**kw):
    e = ApiEnv(); [setattr(e, k, v) for k, v in kw.items()]; return e


def test_temperature_rejected_strict_raises():
    c = LLMClient(_env(strict_temperature=True), client=FakeSDK(True))
    with pytest.raises(TemperatureRejected):
        c.complete("p")


def test_temperature_rejected_lenient_records_drop():
    sdk = FakeSDK(True)
    c = LLMClient(_env(strict_temperature=False), client=sdk)
    c.complete("p")
    m = c.meta()
    assert m["temperature_used"] is None and m["temperature_dropped"] and m["temperature_requested"] == 0.0
    assert "temperature" not in sdk.calls[-1]


def test_unrelated_error_is_not_swallowed():
    class Boom(FakeSDK):
        def create(self, **kw): raise RuntimeError("rate limit")
    with pytest.raises(RuntimeError, match="rate limit"):
        LLMClient(_env(), client=Boom()).complete("p")


def test_budget_stops_the_run():
    c = LLMClient(_env(budget_usd=0.0001, price_in=10.0, price_out=30.0), client=FakeSDK())
    c.complete("a")                       # 100*10/1e6 + 50*30/1e6 = 0.0025 > budget
    with pytest.raises(BudgetExceeded):
        c.complete("b")


def test_cache_key_includes_temperature(tmp_path):
    path = str(tmp_path / "c.json")
    class E(ApiEnv):
        def freeze_cache_path(self, environ=None): return path
    def mk(temp):
        e = E(); e.temperature = temp
        return LLMClient(e, client=FakeSDK())
    a = mk(0.0); a.complete("same prompt"); a.complete("same prompt")
    assert a.api_calls == 1 and a.cache_hits == 1
    b = mk(0.5); b.complete("same prompt")            # different temperature must NOT hit the stale entry
    assert b.api_calls == 1 and b.cache_hits == 0


def test_mock_is_deterministic_and_parses():
    from rsc.methods.hypotheses import parse_hypotheses, build_prompt
    t = make_task(names=("a", "b"), fn=lambda X: X[:, 0] * X[:, 1])
    m = MockLLM()
    p = build_prompt(t)
    assert m.complete(p) == m.complete(p) and m.api_calls == 2
    assert parse_hypotheses(m.complete(p))[0].equation.endswith("*a*b")


# ---------------------------------------------------------------- nn seed
def test_nn_run_seed_changes_seed_and_default_is_backward_compatible():
    from rsc.methods.nn import seed_from
    assert NNBaseline("d").seed == seed_from("d")                 # old behaviour unchanged
    assert NNBaseline("d", run_seed=99).seed != NNBaseline("d", run_seed=42).seed
    assert NNBaseline("d", hidden=64).widths == (64, 64)
    assert NNBaseline("d", hidden=(8, 4)).widths == (8, 4)


def test_nn_fits_and_seeds_differ():
    t = make_task(fn=lambda X: X[:, 0] ** 2, n=60)
    a = NNBaseline("d", epochs=60, run_seed=1).fit(t.X, t.y).predict(t.X)
    b = NNBaseline("d", epochs=60, run_seed=2).fit(t.X, t.y).predict(t.X)
    assert r2(t.y, a) > 0.5 and not np.allclose(a, b)
