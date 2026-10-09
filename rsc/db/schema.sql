-- rsc/db: experiments & results (SQLite). One row per (variant, case, seed) run.
CREATE TABLE variant   (variant_id TEXT PRIMARY KEY, name TEXT, entry_symbol TEXT, source_file TEXT, kind TEXT); -- kind: hybrid|baseline
CREATE TABLE protocol  (protocol_id TEXT PRIMARY KEY, domain TEXT, n_cases INTEGER);        -- defi|feynman|nguyen12|all30
CREATE TABLE experiment(exp_id TEXT PRIMARY KEY, step TEXT, protocol_id TEXT REFERENCES protocol, split TEXT, paper_section TEXT);
CREATE TABLE test_case (case_id TEXT PRIMARY KEY, protocol_id TEXT REFERENCES protocol, domain TEXT, subdomain TEXT, ground_truth TEXT, description TEXT);
CREATE TABLE run (run_id INTEGER PRIMARY KEY, exp_id TEXT REFERENCES experiment, variant_id TEXT REFERENCES variant,
                  case_id TEXT REFERENCES test_case, seed INTEGER, engine_version TEXT, git_sha TEXT, api_calls INTEGER,
                  wall_s REAL, status TEXT, decision TEXT, split TEXT, is_synthetic INTEGER DEFAULT 0, UNIQUE(exp_id, variant_id, case_id, seed));                                   -- decision: fitted_llm|ensemble|nn|llm|...
CREATE TABLE result (run_id INTEGER PRIMARY KEY REFERENCES run, r2_train REAL, r2_test_in REAL, r2_far REAL, extrap_tier INTEGER,
                     rmse REAL, mae REAL, near_perfect INTEGER, catastrophic INTEGER, expr TEXT);
CREATE TABLE winrate(exp_id TEXT, variant_id TEXT, metric TEXT, wins REAL, ties REAL, n INTEGER, ci_lo REAL, ci_hi REAL);
CREATE TABLE reference (bibkey TEXT PRIMARY KEY, status TEXT, doi TEXT, matched_title TEXT, score REAL, checked_on TEXT);
CREATE TABLE issue_patch (issue_id TEXT PRIMARY KEY, kind TEXT, description TEXT, patch_file TEXT, fixed INTEGER, affects_exp TEXT);

-- A1/A2 sweeps and ablations (normalised contract, see scripts/ingest_sweeps.py)
CREATE TABLE IF NOT EXISTS sweep (kind TEXT, variant_id TEXT, case_id TEXT, seed INTEGER, mode TEXT, level REAL,
  r2_far REAL, r2_test_in REAL, engine_version TEXT, is_synthetic INTEGER DEFAULT 0, source TEXT);
