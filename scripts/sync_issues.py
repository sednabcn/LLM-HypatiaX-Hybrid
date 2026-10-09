#!/usr/bin/env python3
"""Mirror audit/issues/issues.yml to GitHub Issues (issues.yml stays the source of truth).

  python scripts/sync_issues.py            # dry run: print what would change
  python scripts/sync_issues.py --apply    # create/update/close via the `gh` CLI (needs GH_TOKEN, issues:write)

Issue title = "[A-0xx] <title>"; labels: audit, kind:<kind>, status:<status>. status fixed -> issue closed.
Drift report: issues closed on GitHub while still open/mitigated in issues.yml are LISTED, never auto-edited
(change issues.yml through a PR, so the register stays reviewable).
"""
import argparse, json, pathlib, subprocess, sys
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

def gh(*args, check=True):
    import shutil
    if shutil.which("gh") is None:
        if "--apply" in sys.argv:
            sys.exit("gh CLI not found (needed for --apply)")
        return "[]"   # dry run without gh: pretend no issues exist yet
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=check, cwd=ROOT).stdout

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true"); a = ap.parse_args()
    items = yaml.safe_load((ROOT / "audit/issues/issues.yml").read_text())["issues"]
    existing = json.loads(gh("issue", "list", "--label", "audit", "--state", "all", "--limit", "300",
                             "--json", "number,title,state,labels") or "[]")
    by_id = {e["title"].split("]")[0].lstrip("["): e for e in existing if e["title"].startswith("[A-")}
    for it in items:
        title = f"[{it['id']}] {it['title']}"
        want = ["audit", f"kind:{it['kind']}", f"status:{it['status']}"]
        body = f"Affects: {', '.join(it.get('affects', []))}\nPatch: {it.get('patch', '-')}\n\n_Mirrored from audit/issues/issues.yml - edit there, not here._"
        e = by_id.get(it["id"])
        if e is None:
            print(f"CREATE  {title}")
            if a.apply:
                for l in want: gh("label", "create", l, "--force", check=False)
                gh("issue", "create", "--title", title, "--body", body, *sum((["--label", l] for l in want), []))
                if it["status"] == "fixed":
                    n = json.loads(gh("issue", "list", "--search", f"[{it['id']}] in:title", "--state", "open", "--json", "number"))[0]["number"]
                    gh("issue", "close", str(n))
            continue
        have = {l["name"] for l in e["labels"]}
        stale = {l for l in have if l.startswith("status:")} - set(want)
        if stale or not set(want) <= have or (it["status"] == "fixed") != (e["state"] == "CLOSED"):
            print(f"UPDATE  #{e['number']} {it['id']} -> {it['status']}")
            if a.apply:
                for l in want: gh("label", "create", l, "--force", check=False)
                gh("issue", "edit", str(e["number"]), "--title", title,
                   *sum((["--add-label", l] for l in want), []), *sum((["--remove-label", l] for l in stale), []))
                gh("issue", "close" if it["status"] == "fixed" else "reopen", str(e["number"]))
        if e["state"] == "CLOSED" and it["status"] != "fixed":
            print(f"DRIFT   #{e['number']} {it['id']} closed on GitHub but '{it['status']}' in issues.yml -> review & update issues.yml via PR")
    if not a.apply:
        print("(dry run - nothing changed; use --apply)")

if __name__ == "__main__":
    sys.exit(main())
