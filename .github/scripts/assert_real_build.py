#!/usr/bin/env python3
"""Release guard: fail if the paper would be built from synthetic data or unverified inputs."""
import pathlib, re, sqlite3, sys
R = pathlib.Path(__file__).resolve().parents[2]; bad = []
if "DRYRUN" in (R / "paper/gen_paths.tex").read_text(): bad.append("paper/gen_paths.tex says DRYRUN")
db = R / "rsc/db/results.sqlite"
if not db.exists(): bad.append("rsc/db/results.sqlite missing (no real results ingested)")
else:
    c = sqlite3.connect(db)
    if c.execute("select count(*) from run where is_synthetic=1").fetchone()[0]: bad.append("synthetic rows in real DB")
    if c.execute("select count(*) from run").fetchone()[0] == 0: bad.append("real DB is empty")
for f in (R / "paper/tables").glob("*.tex"):
    if re.search(r"SYNTHETIC|DRY-RUN", f.read_text()): bad.append(f"{f.name} contains synthetic marker")
rep = R / "audit/references_report.csv"
if not rep.exists(): bad.append("references not validated (audit/references_report.csv missing)")
if bad: sys.exit("RELEASE BLOCKED:\n  - " + "\n  - ".join(bad))
print("real-build guard: OK")
