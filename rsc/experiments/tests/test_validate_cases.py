import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_cases as V

FIX = pathlib.Path(__file__).with_name("fixtures")

def run(name):
    cases = V.load([FIX / name]); return cases, V.validate(cases)

def codes(findings, cid): return {c for c, _ in findings[cid]}

def test_good_cases_are_clean():
    _, f = run("cases_good.yml")
    assert all(not any(c.startswith("E_") for c, _ in fs) for fs in f.values()), f

def test_each_planted_defect_is_reported():
    _, f = run("cases_bad.yml")
    assert "E_DUP" in codes(f, "B-002")          # same structure as B-001
    assert "E_CONST" in codes(f, "B-003")        # target ~ 1 everywhere. NOTE: the original Reserve-Ratio bug was two inputs generated from identical
    # linspaces; that is a DATA-GENERATOR defect (inputs must be independent) and is checked at playbook step 5.
    assert "E_NONFINITE" in codes(f, "B-004")
    assert "E_NOEXTRAP" in codes(f, "B-005")
    assert "E_LEAK" in codes(f, "B-006")
    assert "E_PARENT" in codes(f, "B-007")
    assert "W_SCALE" in codes(f, "B-008")
    assert "E_MISSING" in codes(f, "B-009")
    assert not any(c.startswith("E_") for c in codes(f, "B-001"))   # the base case itself is fine

def test_views_never_contain_formula(tmp_path):
    cases, _ = run("cases_good.yml")
    assert V.export_views(cases, tmp_path) == []
    assert not any("k * m" in p.read_text() for p in tmp_path.glob("*.json"))

def test_freeze_needs_30_cases_10_per_tier():
    cases, f = run("cases_good.yml")
    probs, h = V.freeze("T", cases, f)
    assert h is None and any("exactly 30" in p for p in probs)
