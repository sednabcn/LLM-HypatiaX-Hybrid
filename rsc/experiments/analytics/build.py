#!/usr/bin/env python3
"""Build every generated artefact of the paper from the SQLite DB + config.

    python build.py            -> reads results.sqlite, writes paper/tables, paper/figures
    python build.py --dryrun   -> reads results_dryrun.sqlite, writes paper/_dryrun/{tables,figures}
                                  and stamps all captions/figures "DRY-RUN (synthetic)"
Also writes paper/gen_paths.tex (\\tabdir, \\figdir, \\datastatus) that main.tex inputs.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, warnings
import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "rsc" / "db")); sys.path.insert(0, str(HERE))
from db import connect          # noqa: E402
import stats as S               # noqa: E402
warnings.filterwarnings("ignore", category=RuntimeWarning)

def tex(s):  # escape for LaTeX text
    return str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")

def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "--"
    if isinstance(x, float) and np.isinf(x): return "$-\\infty$"
    if isinstance(x, (int, np.integer)): return str(x)
    return f"{x:.{nd}f}" if abs(x) < 1e4 else f"{x:.1e}"

def write_table(path, header, rows, align=None, caption_note=""):
    align = align or "l" + "r" * (len(header) - 1)
    lines = ["\\begin{tabular}{%s}" % align, "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}"]
    pathlib.Path(path).write_text("\n".join(lines) + "\n")

def watermark(fig, dry):
    if dry:
        fig.text(0.5, 0.5, "DRY-RUN  synthetic data", fontsize=26, color="red", alpha=0.18, ha="center", va="center", rotation=25)

def load(con):
    q = """SELECT r.run_id, r.exp_id, r.variant_id, r.case_id, r.seed, r.engine_version, r.wall_s, r.api_calls, r.status, r.decision,
                  r.split, c.r2_train, c.r2_test_in, c.r2_far, c.extrap_tier, c.rmse, c.mae, c.near_perfect, c.catastrophic,
                  t.domain, t.description
           FROM run r JOIN result c ON c.run_id=r.run_id JOIN test_case t ON t.case_id=r.case_id"""
    return pd.read_sql_query(q, con)

# ---------------------------------------------------------------- static tables (T1,T3,T11,T14)
ADV_LIM = {  # from HypatiaX static-analysis report (PDF 1, sections 5.1/5.2); results column filled by 5.x
 "V1": ("Refit of LLM constants lets the LLM get only the form right; explicit non-finite guards", "Gate uses training R$^2$, thresholds (0.85/0.05) hand-tuned; no leakage check"),
 "V2": ("\\texttt{force\\_llm} encodes a physics-family prior; ties favour the symbolic form", "Same train-R$^2$ gate; NN-win blend needs $R^2_{nn}>0.90$"),
 "V3": ("Precision-weighted blend, no hard gate (smooth)", "Weights from residual std; earlier version leaked test residuals (Fix 7); weighting differs from V1/V2"),
 "V4": ("LLM and PySR refine one expression; four modes with explicit thresholds", "Dead LLM path before v5.0 (use\\_llm no-op); non-deterministic search"),
 "V5": ("Retry + physics-template fallback chain", "Physics fallback off by default; template library is domain specific"),
 "V6": ("Edge-validation slice and extrapolation guard choose among 5 candidates", "DeFi-specific candidate pool; not defined outside DeFi"),
 "V7": ("Seeds PySR population from LLM prior (Nguyen-12)", "Needs a PySR build exposing \\texttt{guesses}; no analogue on other protocols"),
 "B_LLM": ("Interpretable closed form", "Fails when form unknown; exec() of LLM text"),
 "B_NN": ("Fits arbitrary in-distribution structure", "Poor extrapolation (by construction of far splits)"),
 "B_PYSR": ("No LLM dependency", "Cold-start search cost; not in the runner registry (A-009)")}

def static_tables(out, dry):
    v = yaml.safe_load((ROOT / "config/variants.yml").read_text())["variants"]
    write_table(out / "t1_variants.tex", ["ID", "Name", "Kind", "Registry"],
                [[tex(k), tex(x["name"]), x["kind"], fmt(x["registry"]) if x["registry"] else "--"] for k, x in v.items()], "llll")
    e = yaml.safe_load((ROOT / "config/experiments.yml").read_text())["experiments"]
    write_table(out / "t3_experiments.tex", ["Step", "Protocol", "Split", "Registry methods", "\\S"],
                [[tex(k), x["protocol"], x["split"], ",".join(map(str, x["registry"])), x["section"]] for k, x in e.items()], "lllll")
    write_table(out / "t11_adv_lim.tex", ["ID", "Advantage", "Limitation"],
                [[tex(k), a, b] for k, (a, b) in ADV_LIM.items()], "lp{5.2cm}p{5.2cm}")
    m = yaml.safe_load((ROOT / "config/model_config.yml").read_text()); ex = yaml.safe_load((ROOT / "config/post-process.yml").read_text())
    rows = [["LLM model", tex(m["llm"]["model"])], ["LLM temperature", str(m["llm"]["temperature"])],
            ["PySR populations / generations", f"{m['pysr']['populations']} / {m['pysr']['generations']}"],
            ["PySR timeout (s)", str(m["pysr"]["timeout_s"])], ["NN optimiser / scheduler", tex(m["nn"]["optimizer"] + " / " + m["nn"]["scheduler"])],
            ["Bootstrap resamples", str(ex["stats"]["bootstrap_ci"])], ["Near-perfect / catastrophic", f"$R^2>{ex['stats']['near_perfect_r2']}$ / $R^2<{ex['stats']['catastrophic_r2']}$"],
            ["Tie tolerance (win rate)", str(S.TIE_EPS)], ["Seeds", "per step (Table~\\ref{tab:exps}); 42--46 in full mode"]]
    write_table(out / "t14_hparams.tex", ["Parameter", "Value"], rows, "ll")

# ---------------------------------------------------------------- data tables
def data_tables(df, out, figs, dry):
    exps = list(yaml.safe_load((ROOT / "config/experiments.yml").read_text())["experiments"].keys())
    summary = {}
    # T2 coverage: variant x experiment, #cases
    cov = df.groupby(["variant_id", "exp_id"]).case_id.nunique().unstack("exp_id").reindex(columns=exps)
    write_table(out / "t2_coverage.tex", ["Variant"] + [tex(e.replace("hybrid_", "")) for e in exps],
                [[tex(v)] + [fmt(int(x)) if pd.notna(x) else "n/a" for x in r] for v, r in cov.iterrows()], "l" + "r" * len(exps))
    for exp in exps:
        if exp not in set(df.exp_id): continue
        m = S.case_scores(df, exp); tag = exp.replace("hybrid_", "")
        if len(m) == 0: continue
        if m.shape[1] < 2:
            print(f"[skip] {exp}: only {m.shape[1]} method -> win rate undefined (no baseline in registry, audit A-005)"); continue
        wr = S.winrate_table(m); summary[exp] = dict(n_cases=len(m), winrate=wr.to_dict("records"))
        write_table(out / f"t4_winrate_{tag}.tex", ["Variant", "Wins", "Ties", "$n$", "Win rate", "95\\% CI"],
                    [[tex(r.variant_id), fmt(r.wins, 1), fmt(int(r.ties)), fmt(int(r.n)), fmt(r.winrate), f"[{fmt(r.ci_lo)}, {fmt(r.ci_hi)}]"] for r in wr.itertuples()], "lrrrrr")
        pw = S.pairwise_winrate(m)
        write_table(out / f"t5_pairwise_{tag}.tex", ["row beats col"] + [tex(c) for c in pw.columns],
                    [[tex(i)] + [fmt(x, 2) for x in r] for i, r in pw.iterrows()], "l" + "r" * len(pw.columns))
        fn = S.friedman_nemenyi(m); summary[exp]["friedman"] = fn
        if "avg_rank" in fn:
            write_table(out / f"t5b_ranks_{tag}.tex", ["Variant", "Avg. rank"],
                        [[tex(k), fmt(x, 2)] for k, x in sorted(fn["avg_rank"].items(), key=lambda kv: kv[1])], "lr")
            (out / f"friedman_{tag}.tex").write_text(f"$\\chi^2_F={fn['chi2']:.2f}$, $p={fn['p']:.3g}$, Nemenyi CD$={fn['cd']:.2f}$ ($n={fn['n']}$ cases, $k={fn['k']}$ methods)")
        if len(m.columns) >= 2:
            wh = S.wilcoxon_holm(m); wh.to_csv(out / f"wilcoxon_holm_{tag}.csv", index=False)
        # T6 domain table
        d = df[(df.exp_id == exp) & (df.status == "ok")].groupby(["domain", "variant_id", "case_id"]).r2_far.median().reset_index()
        piv = d.groupby(["domain", "variant_id"]).r2_far.apply(
            lambda s: f"{s.median():.2f} [{s.quantile(.25):.2f},{s.quantile(.75):.2f}]").unstack("variant_id")
        write_table(out / f"t6_domain_{tag}.tex", ["Domain"] + [tex(c) for c in piv.columns],
                    [[tex(i)] + [x if isinstance(x, str) else "--" for x in r] for i, r in piv.iterrows()], "l" + "r" * len(piv.columns))
        # T7 near-perfect / catastrophic
        g = df[(df.exp_id == exp) & (df.status == "ok") & df.r2_far.notna()].groupby("variant_id").agg(
            n=("run_id", "count"), npf=("near_perfect", "mean"), catas=("catastrophic", "mean"))
        write_table(out / f"t7_rates_{tag}.tex", ["Variant", "$n$ runs", "Near-perfect", "Catastrophic"],
                    [[tex(i), fmt(int(r.n)), fmt(r.npf), fmt(r.catas)] for i, r in g.iterrows()], "lrrr")
        # T13 cost
        c = df[df.exp_id == exp].groupby("variant_id").agg(wall=("wall_s", "median"), api=("api_calls", "mean"))
        write_table(out / f"t13_cost_{tag}.tex", ["Variant", "Median wall (s)", "Mean API calls"],
                    [[tex(i), fmt(r.wall, 1), fmt(r.api, 1)] for i, r in c.iterrows()], "lrr")
    # T8 decisions
    dd = df[df.decision.notna()].groupby(["variant_id", "decision"]).size().unstack("decision").fillna(0)
    if len(dd):
        sh = dd.div(dd.sum(axis=1), axis=0)
        write_table(out / "t8_decisions.tex", ["Variant"] + [tex(c) for c in sh.columns],
                    [[tex(i)] + [fmt(x, 2) for x in r] for i, r in sh.iterrows()], "l" + "r" * len(sh.columns))
    for name, why in (("t9_seeded", "A1 ablation (exp1_ablation.py) has no ingest path yet: audit A-011"),
                      ("t10_noise_sample", "A2 sweeps (run_noise_sweep/run_sample_complexity) have no ingest path yet: audit A-010")):
        if not (out / f"{name}.tex").exists() or dry:
            (out / f"{name}.tex").write_text("\\emph{Pending: %s.}\n" % tex(why))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary

# ---------------------------------------------------------------- sweeps (A1, A2)
def load_sweep(con):
    try:
        return pd.read_sql_query("SELECT * FROM sweep", con)
    except Exception:
        return pd.DataFrame()

def sweep_tables_figs(sw, out, figs, dry):
    if sw.empty:
        return False
    # T9: ablation. per (variant, mode) median far R2 over case medians; paired seeded-vs-unseeded Wilcoxon (V7)
    ab = sw[sw.kind == "ablation"]
    if len(ab):
        cm = ab.groupby(["variant_id", "mode", "case_id"]).r2_far.median().reset_index()
        rows = []
        for (v, m), g in cm.groupby(["variant_id", "mode"]):
            rows.append([tex(v), tex(m), fmt(len(g)), f"{g.r2_far.median():.3f}", f"[{g.r2_far.quantile(.25):.3f},{g.r2_far.quantile(.75):.3f}]"])
        v7 = cm[cm.variant_id == "V7"].pivot(index="case_id", columns="mode", values="r2_far").dropna()
        extra = ""
        if len(v7) >= 5:
            from scipy import stats as sps
            d = v7["seeded"] - v7["unseeded"]
            p = 1.0 if np.allclose(d, 0) else sps.wilcoxon(v7["seeded"], v7["unseeded"]).pvalue
            extra = f"V7 seeded$-$unseeded: median diff {d.median():.3f}, Wilcoxon $p={p:.3g}$ ($n={len(v7)}$ cases)"
        write_table(out / "t9_seeded.tex", ["Variant", "Mode", "$n$ cases", "Median $\\rfar$", "IQR"], rows, "llrrr")
        if extra:
            with open(out / "t9_seeded.tex", "a") as f: f.write("\\\\[2pt]\\footnotesize " + extra + "\n")
    # T10 + F7/F8
    for kind, name, figname, xl in (("noise", "noise", "f7_noise.pdf", "noise level"), ("sample", "sample", "f8_sample_complexity.pdf", "n_train")):
        d = sw[sw.kind == kind]
        if d.empty: continue
        cm = d.groupby(["variant_id", "level", "case_id"]).r2_far.median().reset_index()
        piv = cm.groupby(["variant_id", "level"]).r2_far.median().unstack("level")
        write_table(out / f"t10_{name}.tex", ["Variant"] + [fmt(c, 2) if kind == "noise" else str(int(c)) for c in piv.columns],
                    [[tex(i)] + [fmt(x, 3) for x in r] for i, r in piv.iterrows()], "l" + "r" * piv.shape[1])
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        for v, r in piv.iterrows(): ax.plot(piv.columns, r.values, marker="o", label=v)
        if kind == "sample": ax.set_xscale("log")
        ax.set_xlabel(xl); ax.set_ylabel("median far R$^2$"); ax.legend(fontsize=7); fig.tight_layout(); watermark(fig, dry); fig.savefig(figs / figname); plt.close(fig)
    nn = (out / "t10_noise.tex").exists(); ns = (out / "t10_sample.tex").exists()
    parts = []
    if nn: parts.append("\\textbf{Noise sweep}\\\\[2pt]\\input{\\tabdir/t10_noise}")
    if ns: parts.append("\\textbf{Sample size}\\\\[2pt]\\input{\\tabdir/t10_sample}")
    (out / "t10_noise_sample.tex").write_text("\\\\[4pt]".join(parts) + "\n" if parts else "\\emph{no sweep data}\n")
    return True

# ---------------------------------------------------------------- figures
def figures(df, figs, dry):
    exps = [e for e in yaml.safe_load((ROOT / "config/experiments.yml").read_text())["experiments"] if e in set(df.exp_id)]
    # F3 win rate with CI
    fig, axs = plt.subplots(1, len(exps), figsize=(3.0 * len(exps), 3.4), sharey=True, squeeze=False)
    for ax, exp in zip(axs[0], exps):
        m = S.case_scores(df, exp)
        if len(m) == 0: continue
        wr = S.winrate_table(m, n_boot=2000)
        ax.bar(wr.variant_id, wr.winrate, yerr=[wr.winrate - wr.ci_lo, wr.ci_hi - wr.winrate], capsize=2, color="#4c78a8")
        ax.set_title(exp.replace("hybrid_", ""), fontsize=9); ax.tick_params(axis="x", rotation=60, labelsize=7)
    axs[0][0].set_ylabel("win rate (far-extrap. R$^2$)"); fig.tight_layout(); watermark(fig, dry); fig.savefig(figs / "f3_winrate.pdf"); plt.close(fig)
    # F4 heatmaps
    for exp in exps:
        d = df[(df.exp_id == exp) & (df.status == "ok")].groupby(["domain", "variant_id", "case_id"]).r2_far.median().reset_index()
        piv = d.groupby(["domain", "variant_id"]).r2_far.median().unstack("variant_id")
        if piv.empty: continue
        fig, ax = plt.subplots(figsize=(1 + 0.7 * piv.shape[1], 0.8 + 0.5 * piv.shape[0]))
        im = ax.imshow(piv.values.astype(float), vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, fontsize=7)
        ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=7)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, label="median far R$^2$ (clipped colour)"); fig.tight_layout(); watermark(fig, dry)
        fig.savefig(figs / f"f4_heatmap_{exp.replace('hybrid_', '')}.pdf"); plt.close(fig)
    # F5 extrapolation ladder
    if df.extrap_tier.notna().any():
        t = df[df.extrap_tier.notna()].groupby(["variant_id", "extrap_tier"]).size().unstack("extrap_tier").fillna(0)
        t = t.div(t.sum(axis=1), axis=0); fig, ax = plt.subplots(figsize=(5, 3)); t.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
        ax.set_ylabel("share of runs"); ax.legend(title="tier", fontsize=6); fig.tight_layout(); watermark(fig, dry); fig.savefig(figs / "f5_ladder.pdf"); plt.close(fig)
    else: print("[skip] F5: no extrap_tier in data (field not found in runner JSON; see ALIASES in ingest)")
    # F6 train vs far
    s = df[(df.status == "ok") & df.r2_far.notna() & df.r2_train.notna()]
    if len(s):
        fig, ax = plt.subplots(figsize=(4.2, 3.6)); ax.scatter(s.r2_train, s.r2_far, s=6, alpha=.4); ax.set_yscale("symlog", linthresh=1)
        ax.axhline(-1, color="r", lw=.6, ls="--"); ax.set_xlabel("train R$^2$"); ax.set_ylabel("far R$^2$ (symlog)")
        for r in s[s.r2_far < -100].drop_duplicates("case_id").itertuples():
            ax.annotate(r.description[:28], (r.r2_train, r.r2_far), fontsize=6, xytext=(-80, 8), textcoords="offset points", arrowprops=dict(arrowstyle="->", lw=.5))
        fig.tight_layout(); watermark(fig, dry); fig.savefig(figs / "f6_train_vs_far.pdf"); plt.close(fig)
    # F9 gate decisions
    dd = df[df.decision.notna()].groupby(["variant_id", "decision"]).size().unstack("decision").fillna(0)
    if len(dd):
        dd = dd.div(dd.sum(axis=1), axis=0); fig, ax = plt.subplots(figsize=(3.6, 3)); dd.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("share"); fig.tight_layout(); watermark(fig, dry); fig.savefig(figs / "f9_decisions.pdf"); plt.close(fig)
    # F10 random vs PCA paired difference
    pairs = [("hybrid_exp1", "hybrid_exp1_pca"), ("hybrid_exp1b", "hybrid_exp1b_pca")]
    fig, ax = plt.subplots(figsize=(5, 3.2)); drawn = False
    for a, b in pairs:
        if a in set(df.exp_id) and b in set(df.exp_id):
            ma, mb = S.case_scores(df, a), S.case_scores(df, b)
            vs = [v for v in ma.columns if v in mb.columns]; idx = ma.index.intersection(mb.index)
            if len(idx) == 0: continue
            diff = (mb.loc[idx, vs] - ma.loc[idx, vs]).replace([np.inf, -np.inf], np.nan)
            ax.boxplot([diff[v].dropna() for v in vs], tick_labels=vs, showfliers=False); drawn = True; break
    if drawn:
        ax.axhline(0, color="k", lw=.5); ax.set_ylabel("far R$^2$: PCA - random"); ax.tick_params(axis="x", rotation=45, labelsize=7)
        fig.tight_layout(); watermark(fig, dry); fig.savefig(figs / "f10_random_vs_pca.pdf")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dryrun", action="store_true"); a = ap.parse_args()
    base = ROOT / "paper" / ("_dryrun" if a.dryrun else "")
    out, figs = base / "tables", base / "figures"; out.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    static_tables(out, a.dryrun)
    con = connect(dryrun=a.dryrun); df = load(con)
    if df.empty:
        sys.exit("no runs in DB -- run scripts/ingest_results.py first")
    data_tables(df, out, figs, a.dryrun); figures(df, figs, a.dryrun)
    sweep_tables_figs(load_sweep(con), out, figs, a.dryrun)
    # audit table (T12) is written by audit_checks.py; ensure a file exists
    if not (out / "t12_audit.tex").exists():
        (out / "t12_audit.tex").write_text("\\emph{Run audit\\_checks.py to generate.}\n")
    rel = "_dryrun/" if a.dryrun else ""
    (ROOT / "paper" / "gen_paths.tex").write_text(
        f"\\newcommand{{\\tabdir}}{{{rel}tables}}\n\\newcommand{{\\figdir}}{{{rel}figures}}\n"
        f"\\newcommand{{\\datastatus}}{{{'DRYRUN' if a.dryrun else 'REAL'}}}\n"
        f"\\{'' if a.dryrun else 'let\\ifdryrun\\iffalse\\fi%\n\\relax%\n'}{'def\\ifdryrun{\\iftrue}' if a.dryrun else ''}\n" if False else
        f"\\newcommand{{\\tabdir}}{{{rel}tables}}\n\\newcommand{{\\figdir}}{{{rel}figures}}\n\\newcommand{{\\datastatus}}{{{'DRYRUN' if a.dryrun else 'REAL'}}}\n")
    print("built", base)

if __name__ == "__main__":
    main()
