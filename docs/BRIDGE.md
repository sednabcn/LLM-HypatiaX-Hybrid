# GitHub <-> web interface bridge

| Direction | Mechanism |
|---|---|
| repo -> web | `pages_dashboard.yml` builds `docs/state.json` (via `scripts/export_state.py`) and publishes `docs/` to GitHub Pages |
| agent -> repo | PR from an `agent/<topic>` branch, or a `.patch` applied with `git am` (never direct to `main`) |
| your runs -> repo | push result JSONs to `results/<run-id>`; `ingest_results_branch.yml` validates and writes a job summary; then open a PR |
| audit <-> Issues | `python scripts/sync_issues.py [--apply]` (issues.yml is the source of truth; GitHub-side closures are reported as DRIFT) |
| session handoff | `PROJECT_STATE.md` (YAML header: step, next_action, blockers, updated) |

## One-time setup
1. Repo -> Settings -> Pages -> Source: **GitHub Actions**.
2. Settings -> Actions -> General -> allow workflows to read/write (needed only for `sync_issues.py --apply` in CI; locally use `gh auth login`).
3. Merge to `main`; the dashboard appears at `https://<user>.github.io/<repo>/`.
   Private repo: Pages visibility needs a plan that supports it; otherwise open the `state-and-db` artifact from the Actions run, or run `python -m http.server -d docs` after `python scripts/export_state.py`.

## Local preview
    # build a DB first: the 'Build DB' step in .github/workflows/pages_dashboard.yml, run locally
    python scripts/export_state.py && python -m http.server -d docs 8000

The banner shows REAL / DRYRUN / NONE from the DB that was actually read, so a synthetic build can never look like a result.
