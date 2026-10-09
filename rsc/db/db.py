"""SQLite helpers. Real results -> results.sqlite ; synthetic dry-run -> results_dryrun.sqlite (never mixed)."""
from __future__ import annotations
import os, pathlib, sqlite3
ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = pathlib.Path(__file__).with_name("schema.sql")

def db_path(dryrun: bool) -> pathlib.Path:
    return pathlib.Path(os.environ.get("HYPATIAX_DB_DIR", ROOT / "rsc" / "db")) / ("results_dryrun.sqlite" if dryrun else "results.sqlite")

def connect(dryrun: bool = False, reset: bool = False) -> sqlite3.Connection:
    p = db_path(dryrun)
    if reset and p.exists():
        p.unlink()
    new = not p.exists()
    con = sqlite3.connect(p)
    if new:
        # schema.sql uses '--' comments; executescript handles them.
        con.executescript(SCHEMA.read_text())
    return con
