import pathlib, sys, numpy as np, pytest
ROOT = pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "rsc" / "core"))
import llm, nn, pysr

def test_mock_llm_counts_and_is_deterministic():
    m = llm.MockLLM(); a, b = m.complete("p"), m.complete("p"); assert a == b and m.api_calls == 2 and "def formula" in a

def test_nn_seed_stable_and_fits():
    assert nn.seed_from("abc") == nn.seed_from("abc")
    X = np.random.default_rng(0).uniform(1, 2, (80, 2)); y = X[:, 0] * X[:, 1]
    m = nn.NNBaseline("t", epochs=60).fit(X, y); assert np.isfinite(m.predict(X)).all()

def test_pysr_guesses_refused_when_unavailable():
    if pysr.available(): pytest.skip("pysr installed")
    with pytest.raises(RuntimeError): pysr.make_regressor(guesses=["x0"])
