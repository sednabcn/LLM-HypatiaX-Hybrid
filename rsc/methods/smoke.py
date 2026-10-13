"""Smoke run: every variant on one synthetic, extrapolative task.   python -m rsc.methods.smoke [--search auto|pysr|substitute]

With no API key this uses the scripted fixture LLM and (without PySR) the substitute search. The numbers it prints
verify WIRING ONLY. They are not results and must never be compared with the paper.
"""
from __future__ import annotations
import argparse
import numpy as np

from .common import Task, r2, safe_predict
from .env import ApiEnv
from .llm import FixtureLLM
from .stack import build_stack
from .variants import REGISTRY, make


def synthetic_task(seed: int = 42, n: int = 120):
    rng = np.random.default_rng(seed)
    f = lambda X: 2.0 * X[:, 0] + 0.5 * X[:, 1] ** 2
    Xtr = rng.uniform(1, 5, (n, 2)); Xfar = rng.uniform(5, 9, (40, 2))
    task = Task(Xtr, f(Xtr) + rng.normal(0, 0.05, n), ["a", "b"], "linear in a plus half of b squared",
                domain="synthetic", seed=seed, case_id="synthetic", X_test_features=Xfar)
    return task, Xfar, f(Xfar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", default="auto", choices=["auto", "pysr", "substitute"])
    ap.add_argument("--live", action="store_true", help="use the LLM from config/api_env.yml instead of fixtures")
    a = ap.parse_args()
    env = ApiEnv.load()
    llm = None if a.live else FixtureLLM({"default": [{"equation": "2*a + 0.4*b**2", "confidence": 0.7, "reasoning": "fixture"}]})
    stack = build_stack(env, search=a.search, llm=llm)
    task, Xf, yf = synthetic_task()
    print(f"llm={stack.llm.backend} search={stack.search.engine} nn={stack.provenance()['nn_backend']}  (WIRING CHECK, not results)")
    print(f"{'variant':12s} {'decision':24s} {'train_r2':>9s} {'far_r2':>9s} {'llm':>4s} {'srch':>4s}")
    for vid in REGISTRY:
        try:
            r = make(vid).run(task, stack)
            print(f"{vid:12s} {r.decision:24s} {r.train_r2:9.4f} {r2(yf, safe_predict(r, Xf)):9.4f} {r.api_calls:4d} {r.search_calls:4d}")
        except Exception as e:
            print(f"{vid:12s} ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
