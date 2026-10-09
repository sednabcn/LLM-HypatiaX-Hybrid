#!/usr/bin/env python3
"""Generate a SYNTHETIC results tree in the runner's JSON format, to test the pipeline end to end
(ingest -> stats -> tables -> figures -> paper build) on a machine with no PySR/Julia/API key.

!!  EVERY NUMBER HERE IS RANDOM.  Files carry  "synthetic": true ; ingest routes them to
!!  results_dryrun.sqlite ; tables/figures built from them are stamped DRY-RUN.  Never cite them.

The generator injects known artefacts so the audit checks have something to find:
  * one catastrophic far-R2 with perfect in-sample R2 (Portfolio Variance, cf. DESIGN I5)
  * one pre-v5.0 engine row for V4 (must be rejected, I6)
"""
from __future__ import annotations
import argparse, json, pathlib, random, datetime

METHODS = {  # registry idx -> (class name, mean far-R2 offset, spread)  -- arbitrary
    1: ("PureLLMBaselineMethod", 0.55, 0.35), 2: ("ImprovedNN", 0.20, 0.60),
    3: ("HybridDeFiMethod", 0.75, 0.25), 4: ("HybridAllDomainsMethod", 0.70, 0.28),
    5: ("SymbolicEngineMethod", 0.72, 0.30), 6: ("HybridSystemV50_2Method", 0.70, 0.30),
    7: ("EnsembleLLMNNMethod", 0.60, 0.30), 8: ("V4SelectorMethod", 0.78, 0.22),
    9: ("LLMPriorSeedingMethod", 0.74, 0.30), 10: ("PureSymbolicPySRMethod", 0.50, 0.40)}
EXPS = {
 "exp1": ("defi", False, [1,2,3,4,5,6,7,8,10]), "exp1_pca": ("defi", True, [1,2,3,4,5,6,7,8,10]),
 "exp1b": ("defi", False, [1,2,3,4,5,6,7,8,10]), "exp1b_pca": ("defi", True, [1,2,3,4,5,6,7,8,10]),
 "exp2": ("feynman", False, [1,2,3,4,5,6,7,10]), "exp3": ("nguyen12", False, [9,10]), "exp3b": ("nguyen12", False, [9,10])}
DOMAINS = {"defi": ["amm", "risk", "portfolio", "lending"], "feynman": ["physics", "chemistry", "biology"], "nguyen12": ["nguyen"]}
DECISIONS = {3: ["fitted_llm", "ensemble", "nn"], 4: ["llm", "ensemble", "nn"]}

SEEDS = {"exp1":[42],"exp1_pca":[42],"exp2":[42],"exp3":[42],"exp1b":[42,99,123,777,2024],"exp1b_pca":[42,99,123,777,2024],"exp3b":[99,123,777,2024]}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--cases", type=int, default=3, help="cases per domain"); a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    rng = random.Random(0)
    for exp, (proto, pca, regs) in EXPS.items():
        rows = []
        for dom in DOMAINS[proto]:
            ncase = 12 if proto == "nguyen12" else a.cases
            for c in range(ncase):
                desc = f"[SYNTHETIC] {proto}/{dom} case {c}"
                if proto == "defi" and dom == "portfolio" and c == 0:
                    desc = "[SYNTHETIC] Portfolio Variance"
                for seed in SEEDS[exp]:
                    res = {}
                    for r in regs:
                        name, mu, sd = METHODS[r]
                        far = mu - (0.15 if pca else 0) + rng.gauss(0, sd)
                        far = min(far, 1.0)
                        ok = rng.random() > 0.04
                        if desc.endswith("Portfolio Variance") and r == 3 and seed == 42:
                            far, tr = -882.9, 1.0
                        else:
                            tr = min(1.0, far + abs(rng.gauss(0.1, 0.08)))
                        res[name] = {"method": name, "success": ok, "r2": tr, "r2_train": tr, "r2_far": far if ok else None,
                                     "rmse": abs(rng.gauss(1, .5)), "mae": abs(rng.gauss(.8, .4)), "elapsed": abs(rng.gauss(30, 10)),
                                     "decision": rng.choice(DECISIONS[r]) if r in DECISIONS else None,
                                     "engine_version": "v5.4" if r in (5, 6, 9) else None,
                                     "expr": "x0*x1" if ok else None, "extrap_tier": 1 + int(max(0.0, min(far, 0.999)) * 5)}
                        if r == 5 and exp == "exp1b" and seed == 2024 and c == 1:
                            res[name]["engine_version"] = "v4.2.1"          # must be rejected
                    rows.append({"description": desc, "domain": dom, "family": proto, "protocol_source": proto,
                                 "seed": seed, "timestamp": datetime.datetime.now().isoformat(), "results": res})
        d = pathlib.Path(a.out) / "hybrid_comparison" / exp; d.mkdir(parents=True, exist_ok=True)
        (d / f"hybrid_comparison_results_{proto}{'_pca' if pca else ''}.json").write_text(json.dumps(
            {"synthetic": True, "protocol": proto, "pca": pca, "methods": regs, "n_cases": len(rows), "rows": rows}, indent=1))
    # sweeps / ablation in the normalised contract (columns of scripts/ingest_sweeps.py)
    import csv
    sw = pathlib.Path(a.out) / "sweeps"; sw.mkdir(exist_ok=True)
    with open(sw / "sweeps.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["kind", "variant", "case", "seed", "mode", "level", "r2_far", "r2_test_in", "engine_version", "synthetic"])
        for v in ("V1", "V2", "V3", "B_NN"):
            for lvl in (0.0, 0.01, 0.05, 0.1):
                for c in range(6):
                    for sd in (42, 43):
                        w.writerow(["noise", v, f"[SYNTHETIC] noise case {c}", sd, "", lvl, 0.8 - 3 * lvl + rng.gauss(0, .1), 0.9 - 2 * lvl, "", 1])
            for n in (50, 100, 200, 500):
                for c in range(6):
                    for sd in (42, 43):
                        w.writerow(["sample", v, f"[SYNTHETIC] sample case {c}", sd, "", n, 0.4 + 0.1 * (n ** .25) + rng.gauss(0, .1), 0.9, "", 1])
        for c in range(12):
            for sd in (42, 99):
                for mode in ("seeded", "unseeded"):
                    w.writerow(["ablation", "V7", f"[SYNTHETIC] nguyen ablation {c}", sd, mode, "", 0.7 + (0.05 if mode == "seeded" else 0) + rng.gauss(0, .2), 0.9, "v5.4", 1])
                for mode in ("none", "seed", "hybrid", "fallback"):
                    w.writerow(["ablation", "V4", f"[SYNTHETIC] nguyen ablation {c}", sd, mode, "", 0.65 + rng.gauss(0, .2), 0.9, "v5.4", 1])
    print("synthetic tree written to", a.out)

if __name__ == "__main__":
    main()
