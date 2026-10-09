import json, pathlib, subprocess, sys, sqlite3, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "rsc" / "experiments" / "analytics"))
import stats as S

def M(d): return pd.DataFrame(d, index=[f"c{i}" for i in range(len(next(iter(d.values()))))])

def test_single_winner_and_tie_rule():
    m = M({"A": [0.9, 0.5, 0.5], "B": [0.5, 0.8, 0.505], "C": [0.1, 0.1, 0.1]})
    w = S.win_scores(m)
    assert list(w["A"]) == [1.0, 0.0, 0.5] and list(w["B"]) == [0.0, 1.0, 0.5] and w["C"].sum() == 0

def test_failed_run_loses():
    m = M({"A": [-np.inf, 0.2], "B": [0.1, -np.inf]})
    assert list(S.win_scores(m)["A"]) == [0.0, 1.0]

def test_winrate_ci_contains_estimate():
    rng = np.random.default_rng(1); m = M({"A": rng.normal(1, 1, 60), "B": rng.normal(0, 1, 60), "C": rng.normal(0, 1, 60)})
    wr = S.winrate_table(m, n_boot=500)
    for r in wr.itertuples(): assert r.ci_lo <= r.winrate <= r.ci_hi

def test_pairwise_antisymmetry_without_ties():
    m = M({"A": [0.1, 0.9, 0.5], "B": [0.5, 0.2, 0.9]}); pw = S.pairwise_winrate(m)
    assert abs(pw.loc["A", "B"] + pw.loc["B", "A"] - 1) < 1e-9

def test_friedman_detects_ordering_and_holm_monotone():
    rng = np.random.default_rng(0); base = rng.normal(size=40)
    m = M({"A": base + 2, "B": base + 1, "C": base, "D": base - 1})
    fn = S.friedman_nemenyi(m); assert fn["p"] < 1e-6 and fn["avg_rank"]["A"] < fn["avg_rank"]["D"]
    wh = S.wilcoxon_holm(m); assert (np.diff(wh.p_holm.values) >= -1e-12).all()

def test_ingest_guards(tmp_path):
    import os
    env = dict(os.environ, HYPATIAX_DB_DIR=str(tmp_path))          # never touch the real DB
    out = tmp_path / "r"; subprocess.run([sys.executable, str(ROOT / "rsc/tools/synth_results.py"), str(out), "--cases", "2"], check=True, capture_output=True)
    run = lambda *a: subprocess.run([sys.executable, str(ROOT / "scripts/ingest_results.py"), str(out), *a], capture_output=True, text=True, cwd=ROOT, env=env)
    p = run("--reset")                       # real mode must refuse synthetic files
    assert "skipped_synthetic_files" in p.stdout
    assert sqlite3.connect(tmp_path / "results.sqlite").execute("select count(*) from run").fetchone()[0] == 0
    p = run("--dryrun", "--reset")           # dry-run accepts them but rejects engine_version < 5.0
    assert "rejected_pre_v5" in p.stdout
    assert sqlite3.connect(tmp_path / "results_dryrun.sqlite").execute("select count(*) from run where is_synthetic=0").fetchone()[0] == 0

def test_sweep_ingest_guards(tmp_path):
    import os, csv
    env = dict(os.environ, HYPATIAX_DB_DIR=str(tmp_path))
    f = tmp_path / "s.csv"; hdr = ["kind", "variant", "case", "seed", "mode", "level", "r2_far", "r2_test_in", "engine_version", "synthetic"]
    def write(rows):
        with open(f, "w", newline="") as h: w = csv.writer(h); w.writerow(hdr); w.writerows(rows)
    run = lambda *a: subprocess.run([sys.executable, str(ROOT / "scripts/ingest_sweeps.py"), str(f), *a], capture_output=True, text=True, cwd=ROOT, env=env)
    write([["ablation", "V7", "c", 42, "seeded", "", 0.9, 0.9, "v4.2.1", 0], ["ablation", "V7", "c", 42, "unseeded", "", 0.8, 0.9, "v5.4", 0], ["noise", "V1", "c", 42, "", 0.1, 0.5, 0.6, "", 1]])
    p = run("--dryrun"); assert "rejected_pre_v5" in p.stdout and "'rows': 2" in p.stdout   # old engine rejected; synthetic kept only in dryrun
    p = run(); assert "skipped_synthetic" in p.stdout                                         # real mode refuses synthetic
    write([["ablation", "V7", "c", 42, "bogus", "", 0.9, 0.9, "v5.4", 0]])
    assert run("--dryrun").returncode != 0                                                     # unknown mode is a hard error
