#!/usr/bin/env python3
"""
synthetic_catalog.py -- 40 SYNTHETIC cases for validating the pipeline
=======================================================================
WHY THIS EXISTS
    The real protocol modules (experiment_protocol_defi.py,
    experiment_protocol_benchmark_v2.py, experiment_protocol_nguyen12.py,
    experiment_protocol_all_30.py) live in LLM-HypatiaX-REPRO and are NOT in
    this repo. The 7-case fallback catalogue inside experiment_protocol_hybrid.py
    has 1-3 cases per protocol -- far too few for a win rate to mean anything.
    This module provides a bigger, *clearly synthetic* catalogue so that every
    downstream stage (runner -> ingest -> stats -> tables/figures -> paper)
    can be exercised and tested end to end before any API money is spent.

WHAT IT IS NOT
    * Not the real DeFi-73 / Feynman-30 / Nguyen-12 / all-30 suites. The DeFi,
      Feynman and all30 cases are *analogues* written for this repo; only the
      12 Nguyen equations follow the published definitions (Uy et al. 2011).
    * Any number computed from it is PIPELINE-VALIDATION data. Every case is
      tagged meta["synthetic"] = True and the ingest step stores
      run.data_origin = "synthetic"; the paper exporter stamps a watermark on
      every table/figure built from such rows.

Interface is identical to the real adapters:
    adapter.get_all_domains() -> list[str]
    adapter.load_test_data(domain, num_samples=None, noise_level=None,
                           seed=None) -> list[(description, X, y, var_names, meta)]

Descriptions are qualitative on purpose (no formula, no constants), so the
ground-truth-leak scan in rsc/tools/audit_checks.py can pass on them honestly.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Optional

import numpy as np

_FUNCS = {"sqrt": np.sqrt, "log": np.log, "exp": np.exp, "sin": np.sin,
          "cos": np.cos, "abs": np.abs}


@dataclass(frozen=True)
class Case:
    id: str
    protocol: str            # defi | feynman | nguyen12 | all30
    family: str              # ratio_product | physics_broad | nguyen_* | multi_domain_breadth
    domain: str
    description: str         # qualitative; must not contain the formula
    gt: str                  # ground truth, python/numpy syntax (NEVER shown to a method)
    vars: tuple
    train: tuple             # ((lo, hi), ...) per variable
    far: tuple               # ((lo, hi), ...) per variable (extrapolation region)
    noise: float = 0.0       # relative noise on y_train (fraction of std)
    n_train: int = 120

    def f(self, X: np.ndarray) -> np.ndarray:
        ns = {v: X[:, i] for i, v in enumerate(self.vars)}
        ns.update(_FUNCS)
        ns["pi"] = np.pi
        with np.errstate(all="ignore"):
            out = eval(compile(ast.parse(self.gt, mode="eval"), "<gt>", "eval"),
                       {"__builtins__": {}}, ns)
        return np.asarray(out, dtype=float) * np.ones(len(X))

    def sample(self, ranges, n, rng):
        return np.column_stack([rng.uniform(lo, hi, n) for lo, hi in ranges])


def _c(id, protocol, family, domain, desc, gt, vars, train, far, noise=0.0, n=120):
    return Case(id, protocol, family, domain, desc, gt, tuple(vars),
                tuple(train), tuple(far), noise, n)


_D = "defi"; _RP = "ratio_product"
_F = "feynman"; _PB = "physics_broad"
_N = "nguyen12"
_A = "all30"; _MB = "multi_domain_breadth"

CASES: list[Case] = [
    # ---- DeFi analogues (ratio / product structure, extrapolation-heavy) ----
    _c("DEFI-01", _D, _RP, "amm", "Constant-product AMM: tokens received for a given input amount, from the two pool reserves.",
       "y_res*dx/(x_res+dx)", ["x_res", "y_res", "dx"], [(10, 50), (10, 50), (1, 10)], [(50, 100), (50, 100), (10, 30)], 0.01),
    _c("DEFI-02", _D, _RP, "amm", "Impermanent loss of a liquidity position as a function of the relative price change of the pair.",
       "2*sqrt(r)/(1+r)-1", ["r"], [(0.5, 2.0)], [(2.0, 6.0)], 0.01),
    _c("DEFI-03", _D, _RP, "amm", "Fractional price impact of a swap relative to the size of the input reserve.",
       "dx/(x_res+dx)", ["x_res", "dx"], [(10, 50), (1, 10)], [(50, 100), (10, 40)], 0.01),
    _c("DEFI-04", _D, _RP, "lending", "Liquidation price of a collateralised loan given debt, collateral amount and loan-to-value limit.",
       "debt/(coll*ltv)", ["debt", "coll", "ltv"], [(100, 500), (1, 5), (0.5, 0.9)], [(500, 900), (5, 8), (0.3, 0.5)], 0.01),
    _c("DEFI-05", _D, _RP, "lending", "Health factor of a loan position from collateral value, liquidation threshold and debt.",
       "coll*ltv/debt", ["coll", "ltv", "debt"], [(1, 5), (0.5, 0.9), (0.5, 3)], [(5, 9), (0.3, 0.5), (3, 6)], 0.01),
    _c("DEFI-06", _D, _RP, "lending", "Utilisation rate of a lending pool from borrowed and idle funds.",
       "borrows/(cash+borrows)", ["borrows", "cash"], [(10, 60), (10, 60)], [(60, 150), (60, 150)], 0.01),
    _c("DEFI-07", _D, _RP, "staking", "Value of a staked position compounding continuously over time.",
       "p*exp(r*t)", ["p", "r", "t"], [(1, 5), (0.02, 0.1), (1, 5)], [(1, 5), (0.02, 0.1), (5, 12)], 0.01),
    _c("DEFI-08", _D, _RP, "staking", "Effective annual yield obtained from a nominal rate compounded monthly.",
       "(1+r/12)**12-1", ["r"], [(0.02, 0.2)], [(0.2, 0.6)], 0.01),
    _c("DEFI-09", _D, _RP, "derivatives", "Optimal bet fraction (Kelly criterion) from win probability and payoff ratio.",
       "(p*b-(1-p))/b", ["p", "b"], [(0.4, 0.7), (1, 3)], [(0.7, 0.9), (3, 6)], 0.01),
    _c("DEFI-10", _D, _RP, "derivatives", "Leverage-like ratio of an exposure to the margin left after a haircut.",
       "a/(b-c+5)", ["a", "b", "c"], [(1, 10), (1, 5), (0, 3)], [(10, 20), (5, 8), (3, 5)], 0.01),
    _c("DEFI-11", _D, _RP, "portfolio", "Variance of a two-asset portfolio from weights, volatilities and correlation (synthetic analogue of the Portfolio Variance stress case).",
       "w**2*s1**2+(1-w)**2*s2**2+2*w*(1-w)*rho*s1*s2", ["w", "s1", "s2", "rho"],
       [(0.1, 0.9), (0.1, 0.4), (0.1, 0.4), (-0.5, 0.8)], [(0.1, 0.9), (0.4, 0.9), (0.4, 0.9), (-0.5, 0.8)], 0.01),
    _c("DEFI-12", _D, _RP, "portfolio", "Sharpe ratio from expected return, risk-free rate and volatility.",
       "(mu-rf)/sigma", ["mu", "rf", "sigma"], [(0.05, 0.2), (0.0, 0.04), (0.1, 0.4)], [(0.2, 0.5), (0.0, 0.04), (0.02, 0.1)], 0.01),

    # ---- Feynman-style physics analogues ----
    _c("FEY-01", _F, _PB, "feynman_mechanics", "Kinetic energy of a moving body from its mass and speed.",
       "0.5*m*v**2", ["m", "v"], [(1, 5), (1, 5)], [(5, 10), (5, 10)]),
    _c("FEY-02", _F, _PB, "feynman_mechanics", "Gravitational attraction between two masses separated by a distance.",
       "6.674*m1*m2/r**2", ["m1", "m2", "r"], [(1, 5), (1, 5), (1, 4)], [(5, 10), (5, 10), (4, 8)]),
    _c("FEY-03", _F, _PB, "feynman_em", "Electrostatic force between two point charges.",
       "q1*q2/r**2", ["q1", "q2", "r"], [(1, 4), (1, 4), (1, 4)], [(4, 8), (4, 8), (4, 8)]),
    _c("FEY-04", _F, _PB, "feynman_mechanics", "Period of a simple pendulum from its length and the local gravitational acceleration.",
       "2*pi*sqrt(L/g_acc)", ["L", "g_acc"], [(0.5, 3), (8, 11)], [(3, 9), (11, 25)]),
    _c("FEY-05", _F, _PB, "feynman_optics", "Refracted-ray quantity from an index of refraction and an angle of incidence.",
       "n1*sin(th)", ["n1", "th"], [(1, 2), (0.1, 1.0)], [(2, 3), (1.0, 2.5)]),
    _c("FEY-06", _F, _PB, "feynman_thermo", "Pressure of an ideal gas from amount, temperature and volume.",
       "8.314*n_mol*T/V", ["n_mol", "T", "V"], [(1, 5), (200, 400), (1, 5)], [(5, 10), (400, 800), (0.5, 1)]),
    _c("FEY-07", _F, _PB, "feynman_thermo", "Saturating reaction rate of an enzyme as substrate concentration rises.",
       "Vmax*S/(Km+S)", ["Vmax", "S", "Km"], [(1, 5), (0.5, 5), (0.5, 3)], [(5, 10), (5, 20), (3, 6)]),
    _c("FEY-08", _F, _PB, "feynman_relativity", "Relativistic mass increase as a body approaches the speed of light (units with c = 1).",
       "m0/sqrt(1-v**2)", ["m0", "v"], [(1, 5), (0.1, 0.6)], [(5, 8), (0.6, 0.9)]),
    _c("FEY-09", _F, _PB, "feynman_thermo", "Spectral energy density of thermal radiation versus photon energy and temperature.",
       "x**3/(exp(x/T)-1)", ["x", "T"], [(0.5, 3), (1, 3)], [(3, 6), (3, 6)]),
    _c("FEY-10", _F, _PB, "feynman_mechanics", "Displacement of a harmonic oscillator at a later time.",
       "A*sin(w*t)", ["A", "w", "t"], [(1, 3), (0.5, 2), (0.5, 3)], [(3, 5), (2, 3), (3, 6)]),

    # ---- Nguyen-12 (published definitions; Uy et al. 2011) ----
    _c("NG-01", _N, "nguyen_polynomial", "nguyen_polynomial", "Nguyen F1: cubic polynomial of one variable.", "x**3+x**2+x", ["x"], [(-1, 1)], [(1, 2)], 0.0, 100),
    _c("NG-02", _N, "nguyen_polynomial", "nguyen_polynomial", "Nguyen F2: quartic polynomial of one variable.", "x**4+x**3+x**2+x", ["x"], [(-1, 1)], [(1, 2)], 0.0, 100),
    _c("NG-03", _N, "nguyen_polynomial", "nguyen_polynomial", "Nguyen F3: quintic polynomial of one variable.", "x**5+x**4+x**3+x**2+x", ["x"], [(-1, 1)], [(1, 2)], 0.0, 100),
    _c("NG-04", _N, "nguyen_polynomial", "nguyen_polynomial", "Nguyen F4: sextic polynomial of one variable.", "x**6+x**5+x**4+x**3+x**2+x", ["x"], [(-1, 1)], [(1, 2)], 0.0, 100),
    _c("NG-05", _N, "nguyen_transcendental", "nguyen_transcendental", "Nguyen F5: product of trigonometric terms, shifted.", "sin(x**2)*cos(x)-1", ["x"], [(-1, 1)], [(1, 2)], 0.0, 100),
    _c("NG-06", _N, "nguyen_transcendental", "nguyen_transcendental", "Nguyen F6: sum of two sine terms.", "sin(x)+sin(x+x**2)", ["x"], [(-1, 1)], [(1, 2)], 0.0, 100),
    _c("NG-07", _N, "nguyen_transcendental", "nguyen_transcendental", "Nguyen F7: sum of two logarithms.", "log(x+1)+log(x**2+1)", ["x"], [(0, 2)], [(2, 4)], 0.0, 100),
    _c("NG-08", _N, "nguyen_transcendental", "nguyen_transcendental", "Nguyen F8: square root of one variable.", "sqrt(x)", ["x"], [(0, 4)], [(4, 9)], 0.0, 100),
    _c("NG-09", _N, "nguyen_bivariate", "nguyen_bivariate", "Nguyen F9: sum of sines of two variables.", "sin(x)+sin(y**2)", ["x", "y"], [(0, 1), (0, 1)], [(1, 2), (1, 2)], 0.0, 100),
    _c("NG-10", _N, "nguyen_bivariate", "nguyen_bivariate", "Nguyen F10: product of sine and cosine.", "2*sin(x)*cos(y)", ["x", "y"], [(0, 1), (0, 1)], [(1, 2), (1, 2)], 0.0, 100),
    _c("NG-11", _N, "nguyen_bivariate", "nguyen_bivariate", "Nguyen F11: one variable raised to the other.", "x**y", ["x", "y"], [(0.1, 1), (0.1, 1)], [(1, 2), (1, 2)], 0.0, 100),
    _c("NG-12", _N, "nguyen_bivariate", "nguyen_bivariate", "Nguyen F12: mixed polynomial of two variables.", "x**4-x**3+y**2/2-y", ["x", "y"], [(0, 1), (0, 1)], [(1, 2), (1, 2)], 0.0, 100),

    # ---- all30 analogues (multi-domain breadth) ----
    _c("ALL-01", _A, _MB, "mechanics", "Elastic energy stored in a stretched spring.", "0.5*k_s*x**2", ["k_s", "x"], [(1, 5), (0.5, 3)], [(5, 10), (3, 6)], 0.01),
    _c("ALL-02", _A, _MB, "chemistry", "Temperature dependence of a reaction rate constant.", "A*exp(-Ea/T)", ["A", "Ea", "T"], [(1, 5), (1, 5), (1, 5)], [(5, 10), (1, 5), (0.3, 1)], 0.01),
    _c("ALL-03", _A, _MB, "chemistry", "Remaining amount in a first-order decay process.", "n0*exp(-k_d*t)", ["n0", "k_d", "t"], [(1, 5), (0.1, 1), (0.5, 3)], [(1, 5), (0.1, 1), (3, 8)], 0.01),
    _c("ALL-04", _A, _MB, "biology", "Saturating cooperative response of a receptor to ligand concentration.", "x**2/(K**2+x**2)", ["x", "K"], [(0.1, 3), (0.5, 2)], [(3, 8), (0.5, 2)], 0.01),
    _c("ALL-05", _A, _MB, "biology", "Logistic population growth over time toward a carrying capacity.", "Kc/(1+exp(-r*(t-t0)))", ["Kc", "r", "t", "t0"],
       [(5, 20), (0.3, 1.0), (0, 10), (3, 7)], [(5, 20), (0.3, 1.0), (10, 20), (3, 7)], 0.01),
    _c("ALL-06", _A, _MB, "mechanics", "Viscous drag force on a small sphere moving through a fluid.", "6*pi*eta*r*v", ["eta", "r", "v"], [(0.5, 2), (0.5, 2), (0.5, 3)], [(2, 4), (2, 4), (3, 8)], 0.01),
]
CASE_BY_ID = {c.id: c for c in CASES}


def ground_truth(case_id: str) -> str:
    """Used ONLY by SimulatedLLM (an explicit error-model oracle) and by the
    audit; never passed to a method as data."""
    return CASE_BY_ID[case_id].gt


class SyntheticAdapter:
    """Drop-in for the real protocol adapters (same two methods)."""

    def __init__(self, source: str):
        self.source = source
        self._cases = [c for c in CASES if c.protocol == source]

    def get_all_domains(self) -> list:
        return sorted({c.domain for c in self._cases})

    def load_test_data(self, domain: str, num_samples: Optional[int] = None,
                       noise_level: Optional[float] = None, seed: Optional[int] = None) -> list:
        seed = 42 if seed is None else seed
        out = []
        for c in (c for c in self._cases if c.domain == domain):
            rng = np.random.default_rng(abs(hash((c.id, seed))) % (2 ** 32) if False else
                                        (sum(map(ord, c.id)) * 1000003 + seed) % (2 ** 32))
            n = num_samples or c.n_train
            Xtr = c.sample(c.train, n, rng)
            Xte = c.sample(c.train, 40, rng)
            Xfar = c.sample(c.far, 40, rng)
            ytr, yte, yfar = c.f(Xtr), c.f(Xte), c.f(Xfar)
            nz = c.noise if noise_level is None else noise_level
            if nz > 0:
                ytr = ytr + rng.normal(0, nz * np.std(ytr), len(ytr))
            meta = {"id": c.id, "family": c.family, "protocol_source": c.protocol,
                    "domain": c.domain, "subdomain": c.domain, "synthetic": True,
                    "ground_truth": c.gt, "extrapolation_test": True,
                    "X_test": Xte, "y_test": yte, "X_extrap": Xfar, "y_extrap": yfar,
                    "noise_level": nz}
            out.append((c.description, Xtr, ytr, list(c.vars), meta))
        return out


def build_synthetic_adapters() -> dict:
    return {p: SyntheticAdapter(p) for p in ("defi", "feynman", "nguyen12", "all30")}


if __name__ == "__main__":
    import collections
    cnt = collections.Counter(c.protocol for c in CASES)
    print("synthetic catalogue:", dict(cnt), "total", len(CASES))
    for c in CASES:
        X = c.sample(c.train, 50, np.random.default_rng(0))
        Xf = c.sample(c.far, 50, np.random.default_rng(0))
        y, yf = c.f(X), c.f(Xf)
        assert np.all(np.isfinite(y)) and np.all(np.isfinite(yf)), c.id
    print("all ground truths finite on train and far ranges")
