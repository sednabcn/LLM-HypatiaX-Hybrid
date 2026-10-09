"""Win-rate and significance statistics (DESIGN.md section 3).

Definitions
-----------
* per-case score  = median over seeds of r2_far (failed / missing run -> -inf, i.e. it loses)
* compared set    = methods that have a row for the case; a case enters an experiment's
                    comparison only if EVERY method of that experiment has a row (same-cases rule)
* win             = highest r2_far; every method within TIE_EPS (0.01) of the best, when >1 such
                    methods exist, gets 0.5 (so with 3-way ties win scores can sum to >1; reported)
* win rate        = mean win score over cases, percentile bootstrap CI (10k resamples, case level)
* Friedman + Nemenyi CD (Demsar 2006) and pairwise Wilcoxon-Holm on per-case scores
"""
from __future__ import annotations
import itertools, math
import numpy as np, pandas as pd
from scipy import stats as sps

TIE_EPS = 0.01

def case_scores(df: pd.DataFrame, exp_id: str, metric: str = "r2_far") -> pd.DataFrame:
    """df: joined run+result rows. Returns cases x variants matrix of per-case median score."""
    d = df[df.exp_id == exp_id].copy()
    d["score"] = d[metric].where(d.status == "ok", -np.inf).fillna(-np.inf)
    m = d.groupby(["case_id", "variant_id"])["score"].median().unstack("variant_id")
    return m.dropna(axis=0, how="any")            # same-cases rule

def win_scores(m: pd.DataFrame, eps: float = TIE_EPS) -> pd.DataFrame:
    best = m.max(axis=1)
    near = m.ge(best - eps, axis=0) & np.isfinite(m)
    k = near.sum(axis=1)
    w = near.astype(float)
    w[k > 1] = w[k > 1] * 0.5
    w[k == 1] = near[k == 1].astype(float)
    return w

def bootstrap_ci(x: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))

def winrate_table(m: pd.DataFrame, n_boot: int = 10000) -> pd.DataFrame:
    w = win_scores(m)
    rows = []
    for v in m.columns:
        lo, hi = bootstrap_ci(w[v].values, n_boot)
        rows.append(dict(variant_id=v, wins=w[v].sum(), ties=int(((w[v] == 0.5)).sum()), n=len(m),
                         winrate=w[v].mean(), ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows).sort_values("winrate", ascending=False).reset_index(drop=True)

def pairwise_winrate(m: pd.DataFrame, eps: float = TIE_EPS) -> pd.DataFrame:
    """P[row beats col] over cases; |diff|<eps counts 0.5."""
    vs = list(m.columns); out = pd.DataFrame(np.nan, index=vs, columns=vs)
    for a, b in itertools.permutations(vs, 2):
        d = (m[a] - m[b]).replace([np.inf, -np.inf], np.nan)
        both_inf = ~np.isfinite(m[a]) & ~np.isfinite(m[b])
        s = np.where(np.isfinite(m[a]) & ~np.isfinite(m[b]), 1.0,
            np.where(~np.isfinite(m[a]) & np.isfinite(m[b]), 0.0,
            np.where(both_inf, 0.5, np.where(d.abs() < eps, 0.5, (d > 0).astype(float)))))
        out.loc[a, b] = float(np.mean(s))
    return out

def _finite_rank_matrix(m: pd.DataFrame) -> np.ndarray:
    # rank 1 = best; -inf ranks last. Higher score is better.
    return m.rank(axis=1, ascending=False, method="average").values

def friedman_nemenyi(m: pd.DataFrame, alpha: float = 0.05) -> dict:
    n, k = m.shape
    if n < 3 or k < 3:
        return dict(n=n, k=k, note="need >=3 cases and >=3 methods")
    r = _finite_rank_matrix(m)
    chi2, p = sps.friedmanchisquare(*[r[:, j] for j in range(k)])
    avg = r.mean(axis=0)
    q = sps.studentized_range.ppf(1 - alpha, k, np.inf)
    cd = q / math.sqrt(2) * math.sqrt(k * (k + 1) / (6 * n))
    return dict(n=n, k=k, chi2=float(chi2), p=float(p), cd=float(cd),
                avg_rank=dict(zip(m.columns, map(float, avg))))

def wilcoxon_holm(m: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    vs = list(m.columns); pairs = []
    for a, b in itertools.combinations(vs, 2):
        x, y = m[a].values, m[b].values
        fin = np.isfinite(x) & np.isfinite(y)
        # non-finite handled via ranks: replace -inf by (min finite - 1) so signs stay valid
        floor = np.nanmin(np.where(np.isfinite(m.values), m.values, np.nan)) - 1
        x = np.where(np.isfinite(x), x, floor); y = np.where(np.isfinite(y), y, floor)
        if np.allclose(x, y):
            p = 1.0
        else:
            try:
                p = float(sps.wilcoxon(x, y, zero_method="wilcox").pvalue)
            except ValueError:
                p = 1.0
        pairs.append([a, b, p, float(np.median(x - y))])
    df = pd.DataFrame(pairs, columns=["a", "b", "p", "median_diff"]).sort_values("p").reset_index(drop=True)
    mtests = len(df)
    adj = np.maximum.accumulate([(mtests - i) * p for i, p in enumerate(df.p)])
    df["p_holm"] = np.minimum(adj, 1.0)
    df["significant"] = df.p_holm < alpha
    return df
