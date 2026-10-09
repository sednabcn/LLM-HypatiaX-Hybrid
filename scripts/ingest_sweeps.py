#!/usr/bin/env python3
"""Ingest A1 (seeded-vs-unseeded / V4 mode ablation) and A2 (noise, sample-size sweeps) from a NORMALISED CSV.

Contract (one row per run):  kind,variant,case,seed,mode,level,r2_far,r2_test_in,engine_version,synthetic
  kind    : noise | sample | ablation
  variant : id from config/variants.yml (V1..V7, B_NN, B_LLM, B_PYSR)
  mode    : ablation only: V7 -> seeded|unseeded ; V4 -> none|seed|hybrid|fallback
  level   : noise -> noise sigma ; sample -> n_train ; ablation -> blank
The real producers (exp1_ablation.py, run_noise_sweep_benchmark.py, run_sample_complexity_benchmark.py) write their own JSON;
write one small adapter per producer that emits this CSV and inspect 3 rows by hand against the producer's own report
BEFORE ingesting (audit A-010/A-011). Guards: unknown variant/kind/mode -> hard error; engine<5.0 for V4/V5/V7 rejected;
synthetic rows only with --dryrun.
"""
from __future__ import annotations
import argparse, csv, hashlib, math, pathlib, re, sys, collections
import yaml
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "rsc" / "db"))
from db import connect  # noqa: E402
KINDS = {"noise", "sample", "ablation"}
MODES = {"V7": {"seeded", "unseeded"}, "V4": {"none", "seed", "hybrid", "fallback"}}

def fnum(x):
    try:
        v = float(x); return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path"); ap.add_argument("--dryrun", action="store_true"); ap.add_argument("--reset-sweeps", action="store_true")
    a = ap.parse_args()
    variants = set(yaml.safe_load((ROOT / "config" / "variants.yml").read_text())["variants"])
    con = connect(dryrun=a.dryrun)
    con.executescript("CREATE TABLE IF NOT EXISTS sweep (kind TEXT, variant_id TEXT, case_id TEXT, seed INTEGER, mode TEXT, level REAL, r2_far REAL, r2_test_in REAL, engine_version TEXT, is_synthetic INTEGER DEFAULT 0, source TEXT);")
    if a.reset_sweeps: con.execute("DELETE FROM sweep")
    st = collections.Counter()
    for i, r in enumerate(csv.DictReader(open(a.csv_path)), 2):
        if r["kind"] not in KINDS: sys.exit(f"line {i}: unknown kind {r['kind']!r}")
        if r["variant"] not in variants: sys.exit(f"line {i}: unknown variant {r['variant']!r}")
        if r["kind"] == "ablation" and r["mode"] not in MODES.get(r["variant"], set()): sys.exit(f"line {i}: bad mode {r['mode']!r} for {r['variant']}")
        syn = str(r.get("synthetic", "0")).strip() in ("1", "true", "True")
        if syn and not a.dryrun: st["skipped_synthetic"] += 1; continue
        m = re.search(r"(\d+)\.(\d+)", r.get("engine_version") or "")
        if m and (int(m[1]), int(m[2])) < (5, 0) and r["variant"] in ("V4", "V5", "V7"): st["rejected_pre_v5"] += 1; continue
        if not m and r["variant"] in ("V4", "V5", "V7"): st["no_engine_version"] += 1
        cid = hashlib.sha1(f"{r['kind']}|{r['case']}".encode()).hexdigest()[:12]
        con.execute("INSERT INTO sweep VALUES(?,?,?,?,?,?,?,?,?,?,?)", (r["kind"], r["variant"], cid, int(r["seed"]), r["mode"] or None,
                    fnum(r["level"]), fnum(r["r2_far"]), fnum(r["r2_test_in"]), r.get("engine_version") or None, int(syn), pathlib.Path(a.csv_path).name))
        st["rows"] += 1
    con.commit(); print("sweep ingest:", dict(st))

if __name__ == "__main__":
    main()
