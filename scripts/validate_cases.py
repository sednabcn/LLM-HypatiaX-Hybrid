#!/usr/bin/env python3
"""Validate benchmark case files (cases/P*.yml) before they are frozen.

Playbook step 1.1 / 3.3 / 3.4.

Usage:
  python scripts/validate_cases.py                 # validate every cases/P*.yml
  python scripts/validate_cases.py cases/P1_market_risk.yml
  python scripts/validate_cases.py --freeze P1     # validate, then write the hash to cases/FROZEN.json
  python scripts/validate_cases.py --check-frozen  # fail if a frozen protocol changed

Exit code 0 only when there are no errors. Warnings never fail the run, but --freeze
refuses to freeze while any warning remains (rule: "0 warnings unexplained").

What it checks (per case):
  E  required fields present, tier/domain/protocol values valid, id unique
  E  formula parses; every symbol is a declared variable or constant
  E  far range lies outside the train range on at least one variable
  E  target is finite on train AND far regions (sampled)
  E  target has real variance on the train region (catches constant-output bugs)
  E  derived cases name an existing `derived_from` id in the same file
  E  no duplicate functional form (canonical form + numeric fingerprint)
  W  extreme output scale (the 1e-10 guard class of bug)
  W  source_verified is not true / no source
  W  tractable is false
Per file:
  E  tier counts differ from the rule (default 10/10/10)
  E  fewer than --min-per-domain cases in a domain (only if the rule is enabled)

NOT checked here (moved to playbook step 5.3): correlation between generated inputs.
This validator samples inputs itself, so it cannot see a bug in the real data generator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import sympy as sp
import yaml

ROOT = Path(__file__).resolve().parents[1]
TIERS = ("canonical", "perturbed", "composed")
REQUIRED = ("id", "protocol", "tier", "domain", "family", "name", "formula",
            "variables", "source", "memorisation_risk")
RISKS = ("low", "medium", "high")
N_SAMPLES = 256
SCALE_HI, SCALE_LO = 1e8, 1e-8
VAR_REL_TOL = 1e-9          # std / (|mean| + eps) below this = "no real variance"


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, case: str, msg: str) -> None:
        self.errors.append(f"[{case}] {msg}")

    def warn(self, case: str, msg: str) -> None:
        self.warnings.append(f"[{case}] {msg}")


# ----------------------------------------------------------------------------- helpers
def _parse(formula: str, names: list[str]) -> sp.Expr:
    local = {n: sp.Symbol(n, real=True) for n in names}
    return sp.sympify(formula, locals=local)


def _sample(var_ranges: dict, which: str, rng: np.random.Generator, n: int = N_SAMPLES):
    out = {}
    for name, spec in var_ranges.items():
        lo, hi = spec[which]
        out[name] = rng.uniform(lo, hi, n)
    return out


def _scan_points(var_ranges: dict, which: str, n: int = 2001) -> list[dict]:
    """One-variable-at-a-time dense scans (others at midpoint) plus all-low/all-high corners."""
    mids = {k: sum(v[which]) / 2 for k, v in var_ranges.items()}
    out = []
    for k, v in var_ranges.items():
        d = {kk: np.full(n, mm) for kk, mm in mids.items()}
        d[k] = np.linspace(v[which][0], v[which][1], n)
        out.append(d)
    for side in (0, 1):
        out.append({k: np.array([v[which][side]]) for k, v in var_ranges.items()})
    return out


def _evaluate(expr: sp.Expr, consts: dict, names: list[str], data: dict) -> np.ndarray:
    expr = expr.subs({sp.Symbol(k, real=True): v for k, v in consts.items()})
    f = sp.lambdify([sp.Symbol(n, real=True) for n in names], expr, modules="numpy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = f(*[data[n] for n in names])
    return np.broadcast_to(np.asarray(y, dtype=float), (len(next(iter(data.values()))),))


def canonical_key(expr: sp.Expr, consts: dict, names: list[str]) -> str:
    """Structure-only key: constants substituted, variables renamed v0.. in sorted-name order."""
    e = expr.subs({sp.Symbol(k, real=True): v for k, v in consts.items()})
    ren = {sp.Symbol(n, real=True): sp.Symbol(f"v{i}") for i, n in enumerate(sorted(names))}
    return sp.srepr(sp.simplify(sp.expand(e.xreplace(ren))))


def numeric_fingerprint(expr: sp.Expr, consts: dict, names: list[str], train: dict) -> tuple | None:
    """Evaluate on a fixed shared grid (unit cube mapped per variable position) so that
    forms that differ only by variable names or algebraic arrangement collide."""
    if not names:
        return None
    rng = np.random.default_rng(12345)
    k = len(names)
    pts = rng.uniform(0.5, 2.0, (16, k))          # shared positive points, same for all cases
    order = sorted(names)
    data = {n: pts[:, i] for i, n in enumerate(order)}
    try:
        y = _evaluate(expr, consts, order, data)
    except Exception:
        return None
    if not np.all(np.isfinite(y)):
        return None
    return tuple(np.round(y, 8))


# ----------------------------------------------------------------------------- core
def validate_cases(cases: list[dict], tier_rule=(10, 10, 10), min_per_domain: int = 0,
                   check_tier_counts: bool = True) -> Report:
    rep = Report()
    rng = np.random.default_rng(0)
    ids: dict[str, dict] = {}
    forms: dict[str, str] = {}
    prints: dict[tuple, str] = {}

    for idx, c in enumerate(cases):
        cid = str(c.get("id", f"<case #{idx}>"))
        missing = [f for f in REQUIRED if f not in c or c[f] in (None, "", {})]
        if missing:
            rep.err(cid, f"missing required fields: {', '.join(missing)}")
            continue
        if cid in ids:
            rep.err(cid, "duplicate id")
            continue
        ids[cid] = c
        if c["tier"] not in TIERS:
            rep.err(cid, f"tier must be one of {TIERS}, got {c['tier']!r}")
        if c["memorisation_risk"] not in RISKS:
            rep.err(cid, f"memorisation_risk must be one of {RISKS}")
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", cid):
            rep.err(cid, "id may contain only letters, digits, _ and -")

        # variables / constants
        vars_ = c["variables"]
        consts = c.get("constants") or {}
        bad_var = False
        for vn, spec in vars_.items():
            try:
                tlo, thi = spec["train"]
                flo, fhi = spec["far"]
                assert tlo < thi and flo < fhi
            except Exception:
                rep.err(cid, f"variable {vn}: needs train:[lo,hi] and far:[lo,hi] with lo<hi")
                bad_var = True
        if bad_var:
            continue
        names = list(vars_)

        # formula
        try:
            expr = _parse(str(c["formula"]), names + list(consts))
        except Exception as e:  # noqa: BLE001
            rep.err(cid, f"formula does not parse: {e}")
            continue
        free = {s.name for s in expr.free_symbols}
        unknown = free - set(names) - set(consts)
        if unknown:
            rep.err(cid, f"undeclared symbols in formula: {sorted(unknown)}")
            continue
        unused = set(names) - free
        if unused:
            rep.err(cid, f"declared variables not used by the formula: {sorted(unused)}")
            continue

        # far range must actually be outside train
        outside = any(v["far"][0] < v["train"][0] or v["far"][1] > v["train"][1] for v in vars_.values())
        if not outside:
            rep.err(cid, "far range lies inside the train range on every variable (no extrapolation)")

        # finiteness and variance
        try:
            ytr = _evaluate(expr, consts, names, _sample(vars_, "train", rng))
            yfa = _evaluate(expr, consts, names, _sample(vars_, "far", rng))
        except Exception as e:  # noqa: BLE001
            rep.err(cid, f"formula cannot be evaluated: {e}")
            continue
        scan_bad = scan_ratio = 0.0
        try:
            for which, ysamp in (("train", ytr), ("far", yfa)):
                ref = float(np.median(np.abs(ysamp[np.isfinite(ysamp)]))) + 1e-300 if np.any(np.isfinite(ysamp)) else 1.0
                for d in _scan_points(vars_, which):
                    ys = _evaluate(expr, consts, names, d)
                    if not np.all(np.isfinite(ys)):
                        rep.err(cid, f"target not finite on {which} region (dense scan / corner)")
                        break
                    scan_ratio = max(scan_ratio, float(np.max(np.abs(ys))) / ref)
        except Exception:  # noqa: BLE001
            pass
        if scan_ratio > 1e3:
            rep.warn(cid, f"near-singular or extreme growth (max|y| / median|y| = {scan_ratio:.3g}); "
                          "confirm there is no pole inside the ranges")
        if not np.all(np.isfinite(ytr)):
            rep.err(cid, f"target not finite on train region ({np.sum(~np.isfinite(ytr))}/{N_SAMPLES} points)")
        if not np.all(np.isfinite(yfa)):
            rep.err(cid, f"target not finite on far region ({np.sum(~np.isfinite(yfa))}/{N_SAMPLES} points)")
        if np.all(np.isfinite(ytr)):
            sd, mu = float(np.std(ytr)), float(np.mean(np.abs(ytr)))
            if sd == 0 or sd / (mu + 1e-300) < VAR_REL_TOL:
                rep.err(cid, "target has no real variance on the train region (constant output)")
            amax = float(np.max(np.abs(ytr)))
            if amax > SCALE_HI or (0 < amax < SCALE_LO):
                rep.warn(cid, f"extreme output scale (max |y| = {amax:.3g}); R2 guards may misfire")

        # provenance
        if not c.get("source_verified", False):
            rep.warn(cid, "source_verified is not true (verify by search, then review as domain expert)")
        if c.get("tractable") is False:
            rep.warn(cid, "marked not tractable")

        # duplicates
        try:
            key = canonical_key(expr, consts, names)
            fp = numeric_fingerprint(expr, consts, names, vars_)
            dup_reported = False
            if key in forms:
                rep.err(cid, f"duplicate functional form of {forms[key]} (canonical form)")
                dup_reported = True
            else:
                forms[key] = cid
            if fp is not None:
                if fp in prints and not dup_reported:
                    rep.err(cid, f"duplicate functional form of {prints[fp]} (numeric fingerprint)")
                prints.setdefault(fp, cid)
        except Exception as e:  # noqa: BLE001
            rep.warn(cid, f"could not canonicalise formula for the duplicate check: {e}")

    # derived_from links (second pass, ids now known)
    for cid, c in ids.items():
        if c.get("tier") in ("perturbed", "composed"):
            src = c.get("derived_from")
            srcs = src if isinstance(src, list) else ([src] if src else [])
            if not srcs:
                rep.err(cid, f"{c['tier']} case needs derived_from (id of the original case)")
            for s in srcs:
                if s not in ids:
                    rep.err(cid, f"derived_from {s!r} does not exist in this file")
                elif s == cid:
                    rep.err(cid, "derived_from points to itself")

    # per-file rules
    if check_tier_counts:
        for proto in sorted({str(c.get("protocol")) for c in ids.values()}):
            sub = [c for c in ids.values() if str(c.get("protocol")) == proto]
            got = tuple(sum(1 for c in sub if c.get("tier") == t) for t in TIERS)
            if got != tuple(tier_rule):
                rep.err(proto, f"tier counts {dict(zip(TIERS, got))} differ from rule {dict(zip(TIERS, tier_rule))}")
    if min_per_domain:
        by_dom: dict[str, int] = {}
        for c in ids.values():
            by_dom[c["domain"]] = by_dom.get(c["domain"], 0) + 1
        for d, n in by_dom.items():
            if n < min_per_domain:
                rep.err(d, f"only {n} cases in domain (minimum {min_per_domain})")
    return rep


# ----------------------------------------------------------------------------- IO / freeze
class _Loader(yaml.SafeLoader):
    """PyYAML (YAML 1.1) reads 1.0e6 as a STRING because it wants 1.0e+6. Accept both."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?(?:[eE][-+]?[0-9]+)$|^[-+]?\.[0-9_]+(?:[eE][-+]?[0-9]+)?$"),
    list("-+0123456789."),
)


def load_file(path: Path) -> tuple[dict, list[dict]]:
    doc = yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader) or {}
    return doc.get("schema", {}), doc.get("cases") or []


def case_set_hash(cases: list[dict]) -> str:
    blob = json.dumps(sorted(cases, key=lambda c: str(c.get("id"))), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def frozen_path(root: Path) -> Path:
    return root / "cases" / "FROZEN.json"


def freeze(proto: str, root: Path, rep: Report, cases: list[dict], source_file: Path) -> None:
    if rep.errors or rep.warnings:
        raise SystemExit("refusing to freeze: resolve all errors and warnings first")
    fp = frozen_path(root)
    data = json.loads(fp.read_text()) if fp.exists() else {}
    if proto in data:
        raise SystemExit(f"{proto} is already frozen ({data[proto]['sha256'][:12]}); edits create a new version")
    data[proto] = {"sha256": case_set_hash(cases), "n_cases": len(cases),
                   "file": str(source_file.relative_to(root)), "version": 1}
    fp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"frozen {proto}: {data[proto]['sha256']}")


def check_frozen(root: Path) -> int:
    fp = frozen_path(root)
    if not fp.exists():
        print("no cases/FROZEN.json yet")
        return 0
    bad = 0
    for proto, info in json.loads(fp.read_text()).items():
        _, cases = load_file(root / info["file"])
        h = case_set_hash(cases)
        ok = h == info["sha256"]
        print(f"{proto}: {'OK' if ok else 'CHANGED AFTER FREEZE'}")
        bad += 0 if ok else 1
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--freeze", metavar="PROTO")
    ap.add_argument("--check-frozen", action="store_true")
    ap.add_argument("--tiers", default="10,10,10")
    ap.add_argument("--min-per-domain", type=int, default=0)
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args(argv)
    root = Path(a.root)

    if a.check_frozen:
        return check_frozen(root)

    files = [Path(f) for f in a.files] or sorted((root / "cases").glob("P*.yml"))
    if a.freeze:
        files = [f for f in files if f.name.startswith(a.freeze)]
        if not files:
            print(f"no case file for {a.freeze}")
            return 2
    rule = tuple(int(x) for x in a.tiers.split(","))
    rc = 0
    for f in files:
        _, cases = load_file(f)
        rep = validate_cases(cases, rule, a.min_per_domain)
        print(f"{f.name}: {len(cases)} cases, {len(rep.errors)} errors, {len(rep.warnings)} warnings")
        for m in rep.errors:
            print("  ERROR  ", m)
        for m in rep.warnings:
            print("  warning", m)
        rc |= 1 if rep.errors else 0
        if a.freeze and not rep.errors:
            freeze(a.freeze, root, rep, cases, f)
    return rc


if __name__ == "__main__":
    sys.exit(main())
