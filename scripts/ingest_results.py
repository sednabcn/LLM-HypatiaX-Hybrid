#!/usr/bin/env python3
"""Ingest hybrid_comparison_results_*.json (runner output) into SQLite.

Layout expected:  <results_root>/hybrid_comparison/<exp1|exp1_pca|...>/hybrid_comparison_results_*.json
The experiment id is taken from the parent directory name (hybrid_<dir>), because the runner's JSON
does not record it (audit A-002).  Row-level keys come from MethodResult.to_dict(); because that class
lives in the real repo, field names are matched through ALIASES below -- anything unmatched is counted
and reported, never silently defaulted.

Guards (audit checks I6/A4):
  * rows whose engine_version < 5.0 are rejected (dead LLM path, pre-v5.0)
  * rows without engine_version are accepted but flagged  (engine_version_in_results check)
  * synthetic rows (top-level "synthetic": true) only go to the dry-run DB
"""
from __future__ import annotations
import argparse, hashlib, json, math, pathlib, re, sys, collections
import yaml
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rsc" / "db"))
from db import connect  # noqa: E402

ALIASES = {
    "r2_far":   ["r2_far", "extrap_r2_far", "far_r2", "r2_extrap"],
    "r2_test":  ["r2_test_in", "r2_test", "r2_in", "r2_id", "r2"],
    "r2_train": ["r2_train", "train_r2"],
    "rmse": ["rmse"], "mae": ["mae"],
    "tier": ["extrap_tier", "tier"],
    "decision": ["decision", "method_chosen", "gate_decision"],
    "expr": ["expr", "formula", "expression", "python_code", "best_expression"],
    "wall": ["wall_s", "elapsed", "elapsed_s", "time_s", "runtime"],
    "api": ["api_calls", "n_api_calls"],
    "engine": ["engine_version", "version", "VERSION"],
}

def pick(d, key):
    for k in ALIASES[key]:
        if k in d and d[k] is not None:
            return d[k]
    return None

def fnum(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def version_tuple(v):
    m = re.search(r"(\d+)\.(\d+)", str(v or ""))
    return (int(m.group(1)), int(m.group(2))) if m else None

def load_variants():
    cfg = yaml.safe_load((ROOT / "config" / "variants.yml").read_text())["variants"]
    return [(vid, [re.compile(a, re.I) for a in v["aliases"]]) for vid, v in cfg.items()], cfg

def resolve_variant(name, table):
    for vid, pats in table:
        if any(p.search(name) for p in pats):
            return vid
    return None

def case_id(protocol, domain, desc):
    return hashlib.sha1(f"{protocol}|{domain}|{desc}".encode()).hexdigest()[:12]

def ingest_file(con, path, exp_id, split, table, stats, allow_synth):
    doc = json.loads(path.read_text())
    synthetic = bool(doc.get("synthetic", False))
    if synthetic and not allow_synth:
        stats["skipped_synthetic_files"] += 1
        return
    protocol = doc["protocol"]
    con.execute("INSERT OR IGNORE INTO protocol(protocol_id,domain,n_cases) VALUES(?,?,NULL)", (protocol, protocol))
    for row in doc["rows"]:
        cid = case_id(protocol, row.get("domain", ""), row["description"])
        con.execute("INSERT OR IGNORE INTO test_case(case_id,protocol_id,domain,subdomain,ground_truth,description) VALUES(?,?,?,?,?,?)",
                    (cid, protocol, row.get("domain"), row.get("family"), row.get("ground_truth"), row["description"]))
        seed = row.get("seed")
        if seed is None:
            stats["rows_without_seed"] += 1
            seed = -1                       # unknown seed: flagged, kept in a separate stratum
        for mname, res in row["results"].items():
            vid = resolve_variant(res.get("method", mname) or mname, table) or resolve_variant(mname, table)
            if vid is None:
                stats["unmapped_methods"][mname] += 1
                continue
            eng = pick(res, "engine")
            vt = version_tuple(eng)
            if vt is None:
                stats["no_engine_version"] += 1
            elif vt < (5, 0) and vid in ("V4", "V5", "V7"):
                stats["rejected_pre_v5"] += 1
                continue
            ok = bool(res.get("success", True))
            r2f = fnum(pick(res, "r2_far"))
            if r2f is None and ok and res.get("r2") is not None:
                stats["r2_far_missing"] += 1
            r2t = fnum(pick(res, "r2_test"))
            r2 = fnum(pick(res, "r2_train"))
            cur = con.execute(
                "INSERT OR REPLACE INTO run(exp_id,variant_id,case_id,seed,engine_version,git_sha,api_calls,wall_s,status,decision,split,is_synthetic)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (exp_id, vid, cid, seed, str(eng) if eng else None, doc.get("git_sha"),
                 pick(res, "api"), fnum(pick(res, "wall")), "ok" if ok else "fail",
                 pick(res, "decision"), split, int(synthetic)))
            rid = cur.lastrowid
            np_ = int(r2f is not None and r2f > 0.99)
            cat = int(r2f is not None and r2f < -1.0)
            con.execute("INSERT OR REPLACE INTO result(run_id,r2_train,r2_test_in,r2_far,extrap_tier,rmse,mae,near_perfect,catastrophic,expr)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (rid, r2, r2t, r2f, pick(res, "tier"), fnum(pick(res, "rmse")), fnum(pick(res, "mae")), np_, cat,
                         str(pick(res, "expr"))[:2000] if pick(res, "expr") else None))
            stats["runs"] += 1

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_root", help="dir containing hybrid_comparison/<exp>/")
    ap.add_argument("--dryrun", action="store_true", help="write to results_dryrun.sqlite and accept synthetic files")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    exps = yaml.safe_load((ROOT / "config" / "experiments.yml").read_text())["experiments"]
    by_dir = {v["dir"]: (k, v) for k, v in exps.items()}
    table, vcfg = load_variants()
    con = connect(dryrun=a.dryrun, reset=a.reset)
    for vid, v in vcfg.items():
        con.execute("INSERT OR REPLACE INTO variant(variant_id,name,entry_symbol,source_file,kind) VALUES(?,?,?,?,?)",
                    (vid, v["name"], None, None, v["kind"]))
    stats = collections.Counter(); stats["unmapped_methods"] = collections.Counter()
    root = pathlib.Path(a.results_root) / "hybrid_comparison"
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name not in by_dir:
            print(f"[warn] unknown experiment dir {d.name}, skipped"); continue
        exp_id, ecfg = by_dir[d.name]
        con.execute("INSERT OR IGNORE INTO protocol(protocol_id,domain,n_cases) VALUES(?,?,NULL)", (ecfg["protocol"], ecfg["protocol"]))
        con.execute("INSERT OR REPLACE INTO experiment(exp_id,step,protocol_id,split,paper_section) VALUES(?,?,?,?,?)",
                    (exp_id, exp_id, ecfg["protocol"], ecfg["split"], ecfg["section"]))
        for f in sorted(d.glob("hybrid_comparison_results*.json")):
            ingest_file(con, f, exp_id, ecfg["split"], table, stats, a.dryrun)
    con.commit()
    print("ingest summary:", {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in stats.items()})
    if stats["rows_without_seed"]:
        print("[WARN] rows lack a seed (runner bug A-002; apply audit/patches/0002) -> seed=-1; multi-seed medians are NOT valid")
    if stats["unmapped_methods"]:
        print("[WARN] unmapped method names -> extend aliases in config/variants.yml")

if __name__ == "__main__":
    main()
