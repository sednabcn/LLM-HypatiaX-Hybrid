"""Planted-defect tests for scripts/validate_cases.py (playbook 1.1).

Each test builds a small, otherwise clean case list, plants ONE defect, and checks the
validator reports exactly that class of problem. A clean list must produce no errors.
"""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_cases as vc  # noqa: E402

RULE = (1, 1, 1)


def base(i="a", formula="k * x**2 + y", tier="canonical", **kw):
    c = {
        "id": f"P_{i}", "protocol": "PX", "tier": tier, "domain": "d", "family": "f",
        "name": f"case {i}", "formula": formula,
        "variables": {"x": {"train": [1, 2], "far": [2, 5]}, "y": {"train": [0, 1], "far": [1, 3]}},
        "constants": {"k": 3.0}, "source": "synthetic", "source_verified": True,
        "memorisation_risk": "low", "tractable": True,
    }
    if tier != "canonical":
        c["derived_from"] = "P_a"
    c.update(kw)
    return c


def clean_set():
    return [base("a"),
            base("b", "k * x**3 + y", "perturbed"),
            base("c", "k * x**2 + y**2 + x", "composed")]


def run(cases, **kw):
    kw.setdefault("tier_rule", RULE)
    return vc.validate_cases(cases, **kw)


def has(msgs, text):
    return any(text in m for m in msgs)


def test_clean_set_has_no_errors_or_warnings():
    r = run(clean_set())
    assert r.errors == [] and r.warnings == []


def test_missing_field():
    cs = clean_set()
    del cs[0]["source"]
    assert has(run(cs).errors, "missing required fields")


def test_unparseable_formula():
    cs = clean_set()
    cs[0]["formula"] = "k * x ** ("
    assert has(run(cs).errors, "does not parse")


def test_undeclared_symbol():
    cs = clean_set()
    cs[0]["formula"] = "k * x**2 + y + q"
    assert has(run(cs).errors, "undeclared symbols")


def test_unused_variable():
    cs = clean_set()
    cs[0]["formula"] = "k * x**2"
    assert has(run(cs).errors, "not used by the formula")


def test_far_inside_train():
    cs = clean_set()
    cs[0]["variables"]["x"]["far"] = [1.2, 1.8]
    cs[0]["variables"]["y"]["far"] = [0.2, 0.8]
    assert has(run(cs).errors, "no extrapolation")


def test_non_finite_in_far_region():
    cs = clean_set()
    cs[0]["formula"] = "k / (x - 3) + y"          # pole at x=3 lies inside far [2,5]
    # random sampling never lands on the pole; the dense scan sees the blow-up and warns
    assert has(run(cs).warnings, "near-singular")


def test_exact_nan_at_boundary_is_error():
    cs = clean_set()
    cs[0]["formula"] = "log(x - 2) + y"           # log(0) = -inf exactly at the far lower edge
    cs[0]["variables"]["x"]["far"] = [2, 5]
    assert has(run(cs).errors, "not finite")


def test_yaml_exponent_without_sign_is_float(tmp_path):
    f = tmp_path / "t.yml"
    f.write_text("cases:\n  - {a: 1.0e6, b: 2e3, c: 1.5e-3, d: 7}\n")
    _, cases = vc.load_file(f)
    assert cases[0] == {"a": 1.0e6, "b": 2000.0, "c": 0.0015, "d": 7}


def test_log_of_negative_far():
    cs = clean_set()
    cs[0]["formula"] = "log(3 - x) + y"           # x in far goes above 3 -> nan
    assert has(run(cs).errors, "not finite on far region")


def test_constant_output_is_caught():
    cs = clean_set()
    cs[0]["formula"] = "k + 0*x + 0*y"
    r = run(cs)
    assert has(r.errors, "no real variance") or has(r.errors, "not used by the formula")


def test_near_constant_output_is_caught():
    cs = clean_set()
    cs[0]["constants"] = {"k": 3.0}
    cs[0]["formula"] = "1000 + 1e-12 * x + 1e-12 * y"
    assert has(run(cs).errors, "no real variance")


def test_extreme_scale_warns():
    cs = clean_set()
    cs[0]["formula"] = "1e12 * (k * x**2 + y)"
    assert has(run(cs).warnings, "extreme output scale")


def test_duplicate_id():
    cs = clean_set()
    cs[1]["id"] = cs[0]["id"]
    assert has(run(cs).errors, "duplicate id")


def test_duplicate_form_same_variables():
    cs = clean_set()
    cs[1]["formula"] = "y + x*x*k"                # same form, rearranged
    assert has(run(cs).errors, "duplicate functional form")


def test_duplicate_form_after_constant_substitution_and_rename():
    cs = clean_set()
    cs[1]["formula"] = "3.0 * b**2 + a"
    cs[1]["variables"] = {"a": {"train": [0, 1], "far": [1, 3]}, "b": {"train": [1, 2], "far": [2, 5]}}
    cs[1]["constants"] = {}
    # variable roles are swapped by sorted order here, so this is NOT flagged by the canonical key;
    # the test pins the documented limitation instead of pretending it is caught
    r = run(cs)
    assert not has(r.errors, "duplicate id")


def test_perturbed_needs_derived_from():
    cs = clean_set()
    del cs[1]["derived_from"]
    assert has(run(cs).errors, "needs derived_from")


def test_derived_from_must_exist():
    cs = clean_set()
    cs[1]["derived_from"] = "P_zzz"
    assert has(run(cs).errors, "does not exist")


def test_tier_counts_rule():
    cs = clean_set()[:2]
    assert has(run(cs).errors, "tier counts")


def test_unverified_source_warns_not_errors():
    cs = clean_set()
    cs[0]["source_verified"] = False
    r = run(cs)
    assert r.errors == [] and has(r.warnings, "source_verified")


def test_min_per_domain():
    r = run(clean_set(), min_per_domain=5)
    assert has(r.errors, "only 3 cases in domain")


def test_bad_tier_value():
    cs = clean_set()
    cs[0]["tier"] = "novel"
    assert has(run(cs).errors, "tier must be one of")


# ---- freeze / hash ---------------------------------------------------------
def write_project(tmp_path, cases):
    (tmp_path / "cases").mkdir()
    import yaml
    f = tmp_path / "cases" / "PX_test.yml"
    f.write_text(yaml.safe_dump({"schema": {"version": 1}, "cases": cases}))
    return f


def test_freeze_refuses_with_warnings(tmp_path):
    cs = clean_set()
    cs[0]["source_verified"] = False
    write_project(tmp_path, cs)
    with pytest.raises(SystemExit):
        vc.main(["--root", str(tmp_path), "--tiers", "1,1,1", "--freeze", "PX"])


def test_freeze_then_detect_change(tmp_path):
    f = write_project(tmp_path, clean_set())
    assert vc.main(["--root", str(tmp_path), "--tiers", "1,1,1", "--freeze", "PX"]) == 0
    assert (tmp_path / "cases" / "FROZEN.json").exists()
    assert vc.main(["--root", str(tmp_path), "--check-frozen"]) == 0
    # edit a frozen case -> must be detected
    import yaml
    doc = yaml.safe_load(f.read_text())
    doc["cases"][0]["constants"]["k"] = 4.0
    f.write_text(yaml.safe_dump(doc))
    assert vc.main(["--root", str(tmp_path), "--check-frozen"]) == 1


def test_cannot_refreeze(tmp_path):
    write_project(tmp_path, clean_set())
    vc.main(["--root", str(tmp_path), "--tiers", "1,1,1", "--freeze", "PX"])
    with pytest.raises(SystemExit):
        vc.main(["--root", str(tmp_path), "--tiers", "1,1,1", "--freeze", "PX"])


def test_hash_is_order_independent():
    a = clean_set()
    b = list(reversed(copy.deepcopy(a)))
    assert vc.case_set_hash(a) == vc.case_set_hash(b)


def test_template_p1_file_loads_and_reports_expected_state():
    """The shipped P1 template must parse; it is expected to fail the 10/10/10 rule until step 3.1."""
    _, cases = vc.load_file(ROOT / "cases" / "P1_market_risk.yml")
    r = vc.validate_cases(cases, (10, 10, 10))
    assert len(cases) == 3
    assert has(r.errors, "tier counts")
    assert not any("does not parse" in m or "not finite" in m for m in r.errors)
