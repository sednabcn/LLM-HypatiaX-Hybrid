#!/usr/bin/env python3
"""Export the repo's research state to docs/state.json (read by docs/index.html).

Sources: PROJECT_STATE.md (YAML header), audit/issues/issues.yml, SQLite DB (real if present, else
dry-run, else none), paper/references.bib, .github/scripts/assert_real_build.py (release gate).
data_status: REAL (results.sqlite) | DRYRUN (synthetic plumbing DB only) | NONE.
Never invents numbers: anything unavailable is null.
"""
from __future__ import annotations
import argparse, datetime as dt, json, os, pathlib, re, sqlite3, subprocess, sys
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

def project_state():
    p = ROOT / "PROJECT_STATE.md"
    if not p.exists():
        return {"missing": True}
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", p.read_text(), re.S)
    head = yaml.safe_load(m.group(1)) if m else {}
    head = {k: (str(v) if isinstance(v, dt.date) else v) for k, v in (head or {}).items()}
    return head

def audit():
    items = yaml.safe_load((ROOT / "audit/issues/issues.yml").read_text())["issues"]
    n = len(items); c = {"fixed": 0, "open": 0, "mitigated": 0}
    for i in items:
        c[i["status"]] = c.get(i["status"], 0) + 1
    return {"total": n, **c, "fix_rate": round(c["fixed"] / n, 3) if n else None, "items": items}

def runs(db_dir: pathlib.Path):
    for name, status in (("results.sqlite", "REAL"), ("results_dryrun.sqlite", "DRYRUN")):
        p = db_dir / name
        if p.exists():
            con = sqlite3.connect(p)
            rows = con.execute("select exp_id,count(*),count(distinct seed),count(distinct variant_id),"
                               "count(distinct case_id),coalesce(sum(is_synthetic),0) from run group by 1 order by 1").fetchall()
            sw = con.execute("select count(*) from sweep").fetchone()[0]
            return status, {"db": name, "sweep_rows": sw, "total": sum(r[1] for r in rows),
                            "by_exp": [dict(zip(("exp_id", "runs", "seeds", "variants", "cases", "synthetic"), r)) for r in rows]}
    return "NONE", {"db": None, "sweep_rows": 0, "total": 0, "by_exp": []}

def refs():
    t = (ROOT / "paper/references.bib").read_text()
    total = len(re.findall(r"^@\w+\{", t, re.M))
    ver = len(re.findall(r"status:\s*verified", t, re.I))
    return {"total": total, "verified": ver, "target": 100}

def gate():
    s = ROOT / ".github/scripts/assert_real_build.py"
    if not s.exists():
        return {"ok": None, "output": "assert_real_build.py missing"}
    r = subprocess.run([sys.executable, str(s)], capture_output=True, text=True, cwd=ROOT)
    return {"ok": r.returncode == 0, "output": (r.stdout + r.stderr).strip()[-600:]}

def git(*a):
    try:
        return subprocess.check_output(["git", *a], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/state.json"))
    ap.add_argument("--db-dir", default=os.environ.get("HYPATIAX_DB_DIR", str(ROOT / "rsc/db")))
    a = ap.parse_args()
    status, r = runs(pathlib.Path(a.db_dir))
    state = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": os.environ.get("GITHUB_SHA", git("rev-parse", "HEAD") or "")[:7] or None,
        "repo": os.environ.get("GITHUB_REPOSITORY"),
        "data_status": status,
        "project": project_state(),
        "audit": audit(),
        "runs": r,
        "refs": refs(),
        "release_gate": gate(),
    }
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(state, indent=1, default=str))
    au = state["audit"]
    print(f"[state] {status} | runs={r['total']} | audit {au['fixed']}/{au['total']} fixed | "
          f"refs {state['refs']['verified']}/{state['refs']['total']} verified | gate_ok={state['release_gate']['ok']} -> {out}")

if __name__ == "__main__":
    main()
