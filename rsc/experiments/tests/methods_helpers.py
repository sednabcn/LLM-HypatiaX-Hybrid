"""Shared helpers for the test_methods_*.py tests (imported by name; pytest puts this directory on sys.path)."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))            # makes `import rsc.methods` work without installing anything

import numpy as np  # noqa: E402

from rsc.methods.common import Task  # noqa: E402
from rsc.methods.env import ApiEnv  # noqa: E402
from rsc.methods.llm import FixtureLLM  # noqa: E402
from rsc.methods.stack import build_stack  # noqa: E402


def make_task(fn=lambda X: 4.0 * X[:, 0], n=80, names=("x",), seed=42, case_id="t", Xfar=None):
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 5, (n, len(names)))
    return Task(X, fn(X), list(names), "test relation", seed=seed, case_id=case_id, X_test_features=Xfar)


def stack_with(equations, **kw):
    """Stack whose LLM always answers with `equations` (list of strings) -- or [] for 'no answer'."""
    fx = [{"equation": e, "confidence": 0.5, "reasoning": ""} for e in equations]
    return build_stack(ApiEnv(), search="substitute", llm=FixtureLLM({"default": fx}), **kw)
