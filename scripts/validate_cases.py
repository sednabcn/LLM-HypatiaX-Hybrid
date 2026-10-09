#!/usr/bin/env python3
"""Validate the anchor case files (cases/P*.yml): the single, unique list of target formulas.

  python scripts/validate_cases.py                       # all cases/P*.yml
  python scripts/validate_cases.py --strict              # warnings also fail
  python scripts/validate_cases.py --freeze P1           # write sha256 to cases/FROZEN.json (needs a clean P1)
  python scripts/validate_cases.py --export-views DIR    # LLM-visible views: semantic + anonymous (no formula)

Every finding has a code. E_* fail the run, W_* are warnings.
Checks (each comes from a defect already seen in this project):
  E_MISSING/E_PARSE/E_SYMBOLS  schema, formula must parse, symbols declared
  E_DUP                        same functional STRUCTURE twice (constants/variables renamed) -> one case, not two
  E_CONST                      near-constant target (the Reserve-Ratio bug: ratio ~ 1.0 everywhere)
  E_NONFINITE                  NaN/inf anywhere in the train or far region
  E_NOEXTRAP                   no variable has a far range outside its train range
  E_LEAK                       description contains the formula, '=', '**' or '^'
  E_TIER/E_PARENT              tier rules: canonical has no parent, perturbed has 1, composed has >= 2 existing parents
  W_SCALE                      extreme output scale (the Newton-gravity R2 guard bug) without a scale_note
  W_CONSTLEAK                  description contains a declared constant's numeric value
Limits: structure-dedup canonicalises variable order (<= 6 variables) but not algebraic identities beyond
sympy.expand; formulas are parsed with sympy.sympify after a character whitelist -- fine for a trusted repo,
NOT sufficient for formulas submitted by external users (the app needs a proper sandbox).
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, pathlib, re, sys, zlib
import numpy as np, yaml, sympy as sp

ROOT = pathlib.Path(__file__).resolve().parents[1]
REQUIRED = ["id", "name", "family", "tier", "formula", "target", "variables", "description", "source", "memorisation_risk"]
TIERS = ("canonical", "perturbed", "composed")
RISKS = ("high", "medium", "low")
N_PTS = 2000
ALLOWED = re.compile(r"^[A-Za-z0-9_+\-*/^().,\s]+$")
SAFE = {n: getattr(sp, n) for n in ("sqrt", "exp", "log", "sin", "cos", "tan", "atan", "Abs", "Min", "Max", "erf", "pi")}
SAFE["Phi"] = lambda x: (1 + sp.erf(x / sp.sqrt(2))) / 2   # standard normal CDF

def parse(case):
    """-> (expr, findings). expr is None when the formula cannot be used."""
    f = []
    formula = str(case.get("formula", ""))
    if not ALLOWED.match(formula) or "__" in formula:
        return None, [("E_PARSE", "formula has characters outside the whitelist")]
    names = list(case.get("variables") or {}) + list(case.get("constants") or {})
    loc = {**SAFE, **{n: sp.Symbol(n) for n in names}}
    try:
        expr = sp.sympify(formula.replace("^", "**"), locals=loc)
    except Exception as e:
        return None, [("E_PARSE", f"formula does not parse: {e}")]
    free = {s.name for s in expr.free_symbols}
    if free - set(names):
        f.append(("E_SYMBOLS", f"undeclared symbols: {sorted(free - set(names))}"))
    if set(case.get("variables") or {}) - free:
        f.append(("W_UNUSED", f"declared but unused variables: {sorted(set(case['variables']) - free)}"))
    return expr, f

def structure_key(expr, var_names, const_names):
    vs = [sp.Symbol(n) for n in var_names]; cs = [sp.Symbol(n) for n in const_names]
    perms = itertools.permutations(range(len(vs))) if len(vs) <= 6 else [tuple(range(len(vs)))]
    best = None
    for p in perms:
        sub = {vs[i]: sp.Symbol(f"_v{p[i]}") for i in range(len(vs))}
        sub.update({cs[i]: sp.Symbol(f"_c{i}") for i in range(len(cs))})
        k = sp.srepr(sp.expand(expr.xreplace(sub)))
        best = k if best is None or k < best else best
    return best

def numeric(case, expr):
    f = []
    vars_ = case["variables"]; consts = {sp.Symbol(k): v for k, v in (case.get("constants") or {}).items()}
    rng = np.random.default_rng(zlib.crc32(case["id"].encode()))
    names = list(vars_)
    fn = sp.lambdify([sp.Symbol(n) for n in names], expr.xreplace(consts), modules=["scipy", "numpy"])
    out = {}
    for region in ("train", "far"):
        try:
            pts = [rng.uniform(*vars_[n][region], size=N_PTS) for n in names]
        except Exception:
            return [("E_MISSING", f"variables need train and far [lo, hi] ranges")], None
        with np.errstate(all="ignore"):
            y = np.broadcast_to(np.asarray(fn(*pts), dtype=float), (N_PTS,))
        out[region] = y
        if not np.all(np.isfinite(y)):
            f.append(("E_NONFINITE", f"{region} region: {int((~np.isfinite(y)).sum())}/{N_PTS} non-finite values"))
    ok = {r: y[np.isfinite(y)] for r, y in out.items()}
    if len(ok["train"]) and np.std(ok["train"]) < 1e-3 * max(np.abs(ok["train"]).max(), 1e-300):
        f.append(("E_CONST", "target is near-constant over the train region (relative std < 1e-3)"))
    ymax = max((np.abs(v).max() for v in ok.values() if len(v)), default=0.0)
    if (ymax < 1e-6 or ymax > 1e9) and not case.get("scale_note"):
        f.append(("W_SCALE", f"output scale {ymax:.2e} is extreme; add a scale_note and check the R2 guard"))
    return f, out

def check_case(case, all_ids):
    f = []
    miss = [k for k in REQUIRED if case.get(k) in (None, "", {}, [])]
    if miss:
        return [("E_MISSING", f"missing fields: {miss}")]
    if case["tier"] not in TIERS: f.append(("E_TIER", f"tier must be one of {TIERS}"))
    if case["memorisation_risk"] not in RISKS: f.append(("E_MISSING", f"memorisation_risk must be one of {RISKS}"))
    parents = case.get("derived_from") or []
    parents = [parents] if isinstance(parents, str) else list(parents)
    need = {"canonical": 0, "perturbed": 1, "composed": 2}.get(case["tier"])
    if need is not None and ((case["tier"] == "canonical" and parents) or (case["tier"] != "canonical" and len(parents) < need)
                             or (case["tier"] == "perturbed" and len(parents) != 1)):
        f.append(("E_TIER", f"tier '{case['tier']}' needs {'no' if need == 0 else need} parent(s) in derived_from, got {len(parents)}"))
    for p in parents:
        if p not in all_ids: f.append(("E_PARENT", f"derived_from '{p}' is not a case id"))
    d = str(case["description"])
    if "=" in d or "**" in d or "^" in d or str(case["formula"]) in d:
        f.append(("E_LEAK", "description contains '=', '**', '^' or the formula itself"))
    for k, v in (case.get("constants") or {}).items():
        if re.search(rf"(?<![\w.]){re.escape(str(v))}(?![\w.])", d):
            f.append(("W_CONSTLEAK", f"description contains the value of constant '{k}'"))
    expr, pf = parse(case); f += pf
    if expr is None or any(c.startswith("E_") for c, _ in pf):
        return f
    vars_ = case["variables"]
    try:
        if not any(v["far"][0] >= v["train"][1] or v["far"][1] <= v["train"][0] for v in vars_.values()):
            f.append(("E_NOEXTRAP", "no variable has a far range outside its train range"))
        nf, _ = numeric(case, expr); f += nf
    except (KeyError, TypeError, IndexError):
        f.append(("E_MISSING", "each variable needs train and far ranges as [lo, hi]"))
    return f

def load(paths):
    cases = []
    for p in paths:
        doc = yaml.safe_load(pathlib.Path(p).read_text()) or {}
        for c in doc.get("cases") or []:
            c = dict(c); c["_file"] = str(p); c["_protocol"] = doc.get("protocol"); cases.append(c)
    return cases

def validate(cases):
    findings = {}
    ids = [c.get("id") for c in cases]; idset = set(ids)
    seen_names = {}
    keys = {}
    for c in cases:
        cid = c.get("id", "?"); f = check_case(c, idset)
        if ids.count(cid) > 1: f.append(("E_DUP", f"duplicate id {cid}"))
        n = str(c.get("name", "")).lower()
        if n and n in seen_names: f.append(("E_DUP", f"same name as {seen_names[n]}"))
        seen_names[n] = cid
        if c.get("_protocol") and not str(cid).startswith(str(c["_protocol"]) + "-"):
            f.append(("E_MISSING", f"id must start with '{c['_protocol']}-'"))
        expr, _ = parse(c) if c.get("formula") and c.get("variables") else (None, None)
        if expr is not None:
            k = structure_key(expr, list(c["variables"]), list(c.get("constants") or {}))
            if k in keys: f.append(("E_DUP", f"same functional structure as {keys[k]} (use one case with a parameter)"))
            else: keys[k] = cid
        findings[cid] = f
    return findings

def freeze(proto, cases, findings):
    mine = [c for c in cases if c["_protocol"] == proto]
    errs = [(i, m) for c in mine for i, m in [(c["id"], x) for x in findings[c["id"]] if x[0].startswith("E_")]]
    problems = [f"{i}: {m}" for i, m in errs]
    if len(mine) != 30: problems.append(f"needs exactly 30 cases, has {len(mine)}")
    for t in TIERS:
        n = sum(c["tier"] == t for c in mine)
        if n != 10: problems.append(f"needs 10 '{t}' cases, has {n}")
    for c in mine:
        if c.get("source_verified") is not True: problems.append(f"{c['id']}: source_verified is not true")
    if problems: return problems, None
    body = [{k: v for k, v in c.items() if not k.startswith("_")} for c in sorted(mine, key=lambda c: c["id"])]
    h = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
    p = ROOT / "cases" / "FROZEN.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    if proto in cur and cur[proto]["sha256"] != h:
        return [f"{proto} is already frozen with a different hash; create a new protocol version instead of editing"], None
    cur[proto] = {"sha256": h, "n": len(mine)}; p.write_text(json.dumps(cur, indent=1))
    return [], h

def export_views(cases, out):
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True); leaks = []
    for c in cases:
        names = list(c["variables"])
        sem = {"id": c["id"], "name": c["name"], "description": c["description"], "target": c["target"],
               "variables": {n: {"train": c["variables"][n]["train"], **({"unit": c["variables"][n]["unit"]} if "unit" in c["variables"][n] else {})} for n in names}}
        anon = {"id": c["id"], "target": "y", "variables": {f"x{i+1}": {"train": c["variables"][n]["train"]} for i, n in enumerate(names)}}
        for kind, v in (("semantic", sem), ("anonymous", anon)):
            txt = json.dumps(v)
            if str(c["formula"]) in txt or str(c.get("source", "@@")) in txt: leaks.append((c["id"], kind))
            (out / f"{c['id']}.{kind}.json").write_text(json.dumps(v, indent=1))
    return leaks

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*"); ap.add_argument("--strict", action="store_true")
    ap.add_argument("--freeze"); ap.add_argument("--export-views")
    a = ap.parse_args(argv)
    paths = a.paths or sorted((ROOT / "cases").glob("P*.yml"))
    cases = load(paths); findings = validate(cases)
    ne = nw = 0
    for cid, fs in findings.items():
        for code, msg in fs:
            print(f"{code:12s} {cid}: {msg}"); ne += code.startswith("E_"); nw += code.startswith("W_")
    by = {}
    for c in cases:
        d = by.setdefault(c["_protocol"], {t: 0 for t in TIERS})
        d[c.get("tier")] = d.get(c.get("tier"), 0) + 1
    print(f"[cases] {len(cases)} cases | tiers per protocol: {by} | errors={ne} warnings={nw}")
    if a.freeze:
        probs, h = freeze(a.freeze, cases, findings)
        for p in probs: print("FREEZE-BLOCKED", p)
        if h: print(f"[freeze] {a.freeze} sha256={h}")
        else: return 1
    if a.export_views and not ne:
        leaks = export_views(cases, a.export_views)
        for cid, k in leaks: print(f"E_LEAK       {cid}: {k} view contains formula/source")
        if leaks: return 1
    return 1 if ne or (a.strict and nw) else 0

if __name__ == "__main__":
    sys.exit(main())
