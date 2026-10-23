import json, pathlib, re
import numpy as np
import pytest

from methods_helpers import make_task, stack_with, ROOT
from rsc.methods.common import FitResult
from rsc.methods.rows import result_row, METHOD_NAME, config_hash
from rsc.methods.variants import REGISTRY, make

KW = dict(engine_version="v5.0", case_set_hash="abc123", split="random", threshold=0.99)


def _row(vid="V2", eqs=("4*x",), **over):
    task = make_task()
    stack = stack_with(list(eqs))
    fit = make(vid).run(task, stack)
    Xf = np.linspace(6, 9, 20).reshape(-1, 1)
    return result_row(vid, task, fit, stack, X_far=Xf, y_far=4 * Xf[:, 0], **{**KW, **over}), fit, stack


def test_row_has_the_documented_fields():
    row, _, _ = _row()
    for k in ("seed", "engine_version", "api_calls", "wall_s", "decision", "r2_far", "llm_model", "temperature_requested",
              "case_set_hash", "split", "threshold", "config_hash", "comparability_key", "method", "synthetic"):
        assert k in row, k
    assert row["seed"] == 42 and row["api_calls"] == 1 and row["engine_version"] == "v5.0"
    json.dumps(row, default=str)                         # serialisable


def test_r2_far_is_recomputed_from_predictions_not_the_reported_number():
    task = make_task(); stack = stack_with([])
    lying = FitResult("V2", "llm", lambda X: np.zeros(len(X)), train_r2=0.9999)      # claims 0.9999, predicts zeros
    Xf = np.linspace(6, 9, 20).reshape(-1, 1)
    row = result_row("V2", task, lying, stack, X_far=Xf, y_far=4 * Xf[:, 0], **KW)
    assert row["train_r2_reported"] == 0.9999 and row["r2_far"] < 0


def test_missing_comparability_inputs_cannot_be_defaulted():
    task = make_task(); stack = stack_with(["4*x"]); fit = make("V2").run(task, stack)
    with pytest.raises(TypeError):
        result_row("V2", task, fit, stack, X_far=task.X, y_far=task.y)             # no engine_version / hash / split / threshold


def test_fixture_substitute_sklearn_rows_are_marked_synthetic():
    row, _, _ = _row()
    assert row["synthetic"] is True
    assert {"llm_backend=fixture", "search_engine=substitute_feature_library"} <= set(row["synthetic_reasons"])


def test_config_hash_changes_with_variant_params():
    task = make_task(); stack = stack_with(["4*x"])
    a = make("V2").run(task, stack); b = make("V2", tie_eps=0.05).run(task, stack)
    assert config_hash(stack, a) != config_hash(stack, b)
    assert config_hash(stack, a) == config_hash(stack, make("V2").run(task, stack))


def test_api_key_never_in_a_row(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-SECRET-123")
    row, _, _ = _row()
    assert "SECRET-123" not in json.dumps(row, default=str)


def test_every_registry_id_has_a_method_name():
    for vid in REGISTRY:
        base = vid.split("_")[0] if vid.startswith("V4_") else vid
        assert base in METHOD_NAME


def test_method_names_match_frozen_aliases_when_that_file_exists():
    p = ROOT / "config" / "variants.yml"
    if not p.is_file():
        pytest.skip("config/variants.yml (frozen table) is not in this tree yet")
    import yaml
    table = yaml.safe_load(p.read_text())["variants"]
    for vid, name in METHOD_NAME.items():
        assert any(re.search(a, name, re.I) for a in table[vid]["aliases"]), (vid, name)
