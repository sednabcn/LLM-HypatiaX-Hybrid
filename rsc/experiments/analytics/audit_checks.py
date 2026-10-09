#!/usr/bin/env python3
"""Automated audit checks from config/audit.yml (DESIGN I5, I6, A3, A4).

Static checks scan rsc/generation/variants/*.py ; data checks query the results DB.
Writes audit/mapping_issues-patches-fix-rate/checks.json and, from audit/issues/issues.yml,
the issue_patch table, fix-rate CSV and paper/<gen>/t12_audit.tex.
Static hits are LEADS for manual review, not verdicts: each carries file:line.
"""
from __future__ import annotations
import argparse, csv, json, pathlib, re, sys, collections
import yaml
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "rsc" / "db")); from db import connect  # noqa: E402

VAR_DIR = ROOT / "rsc" / "generation" / "variants"
LEAK_TOKENS = re.compile(r"(ground_truth|expected_form|true_formula|true_expr|gt_formula)", re.I)
PROMPT_CTX = re.compile(r"(prompt|messages|content\s*[:=]|f\"\"\"|f'''|user_msg|system_msg)", re.I)
RESID_STD = re.compile(r"std\(\s*\w*(pred|hat)\w*\s*-\s*(y_test|y_true|y_val|y_far|y)\s*\)", re.I)
USE_LLM = re.compile(r"\buse_llm\b")

def scan(pattern, require_ctx=None):
    hits = []
    for f in sorted(VAR_DIR.glob("*.py")):
        for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if line.lstrip().startswith("#"): continue
            if pattern.search(line) and (require_ctx is None or require_ctx.search(line)):
                hits.append(f"{f.name}:{i}: {line.strip()[:110]}")
    return hits

def check_use_llm_read():
    """Dead-path heuristic: a file that mentions use_llm but never branches on it."""
    out = []
    for f in sorted(VAR_DIR.glob("*.py")):
        t = f.read_text(errors="ignore")
        if USE_LLM.search(t) and not re.search(r"(if|while|and|or|not)\s+(self\.)?use_llm|use_llm\s*(and|or|==|\))|\w+\s*=\s*.*use_llm", t):
            out.append(f.name)
    return out

def data_checks(con):
    res = {}
    q = lambda s, *a: con.execute(s, a).fetchall()
    res["engine_version_in_results"] = {v: n for v, n in q(
        "SELECT variant_id, SUM(engine_version IS NULL) FROM run WHERE variant_id IN ('V4','V5','V7') GROUP BY 1")}
    res["llm_variant_zero_api_calls"] = [v for v, n, z in q(
        "SELECT variant_id, COUNT(*), SUM(COALESCE(api_calls,0)=0) FROM run WHERE variant_id NOT IN ('B_NN','B_PYSR') GROUP BY 1") if n and z == n]
    res["portfolio_variance_discrepancy"] = [dict(variant=v, seed=s, exp=e, r2_train=a, r2_far=b) for v, s, e, a, b in q(
        "SELECT r.variant_id, r.seed, r.exp_id, c.r2_train, c.r2_far FROM run r JOIN result c USING(run_id) JOIN test_case t USING(case_id) "
        "WHERE t.description LIKE '%Portfolio Variance%' AND c.r2_train>0.99 AND c.r2_far<-1")]
    res["catastrophic_total"] = q("SELECT COUNT(*) FROM result WHERE catastrophic=1")[0][0]
    return res

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dryrun", action="store_true"); a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config" / "audit.yml").read_text())
    out = {"static": {"gt_leak_prompt_scan": scan(LEAK_TOKENS, PROMPT_CTX),
                      "test_residual_leak": scan(RESID_STD),
                      "dead_llm_path_files": check_use_llm_read()}}
    con = connect(dryrun=a.dryrun)
    out["data"] = data_checks(con)
    d = ROOT / "audit" / "mapping_issues-patches-fix-rate"; d.mkdir(parents=True, exist_ok=True)
    (d / ("checks_dryrun.json" if a.dryrun else "checks.json")).write_text(json.dumps(out, indent=1, default=str))
    # issues -> DB, fix-rate
    issues = yaml.safe_load((ROOT / "audit" / "issues" / "issues.yml").read_text())["issues"]
    for i in issues:
        con.execute("INSERT OR REPLACE INTO issue_patch VALUES(?,?,?,?,?,?)",
                    (i["id"], i["kind"], i["title"], i.get("patch"), int(i["status"] == "fixed"), ",".join(i.get("affects", []))))
    con.commit()
    by = collections.defaultdict(lambda: [0, 0])
    for i in issues:
        for e in (i.get("affects") or ["all"]):
            by[e][1] += 1; by[e][0] += i["status"] == "fixed"
    with open(d / "fix_rate.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["affects", "issues", "fixed", "fix_rate"])
        for e, (fx, n) in sorted(by.items()): w.writerow([e, n, fx, f"{fx / n:.2f}"])
    base = ROOT / "paper" / ("_dryrun" if a.dryrun else "") / "tables"; base.mkdir(parents=True, exist_ok=True)
    esc = lambda s: str(s).replace("#", "\\#").replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")
    rows = ["\\begin{tabular}{llp{6.8cm}ll}", "\\toprule", "ID & Kind & Issue & Status & Patch \\\\", "\\midrule"]
    rows += [f"{esc(i['id'])} & {esc(i['kind'])} & {esc(i['title'])} & {esc(i['status'])} & {esc(i.get('patch') or '--')} \\\\" for i in issues]
    nfix = sum(i["status"] == "fixed" for i in issues)
    rows += ["\\midrule", f"\\multicolumn{{5}}{{l}}{{Fix rate: {nfix}/{len(issues)} = {100 * nfix / len(issues):.0f}\\%}} \\\\", "\\bottomrule", "\\end{tabular}"]
    (base / "t12_audit.tex").write_text("\n".join(rows) + "\n")
    print(json.dumps({k: (v if k != "static" else {kk: len(vv) for kk, vv in v.items()}) for k, v in out.items()}, indent=1, default=str))
    print("checks declared in audit.yml:", cfg.get("checks"))

if __name__ == "__main__":
    main()
