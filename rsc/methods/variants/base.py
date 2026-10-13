from __future__ import annotations
import time
from typing import ClassVar

import numpy as np

from ..common import FitResult, Task


class Variant:
    """One interface for every method: fit(task, stack) -> FitResult.
    `fit` never receives the test labels. It must not raise on a bad LLM answer (that is a normal outcome,
    recorded in the result); it may raise on configuration errors (missing PySR guesses, budget, strict temperature)."""
    id: ClassVar[str] = "?"
    kind: ClassVar[str] = "hybrid"
    name: ClassVar[str] = "?"

    def __init__(self, **params):
        self.params = params

    def p(self, key, default):
        return self.params.get(key, default)

    def fit(self, task: Task, stack) -> FitResult:        # pragma: no cover - interface
        raise NotImplementedError

    def run(self, task: Task, stack) -> FitResult:
        """fit() plus the accounting every result row needs (API calls and wall time of THIS fit)."""
        c0 = stack.llm.api_calls
        t0 = time.time()
        res = self.fit(task, stack)
        res.api_calls = stack.llm.api_calls - c0
        res.info.setdefault("seconds", round(time.time() - t0, 3))
        res.info.setdefault("variant_params", dict(self.params))
        return res


def nan_predict(n):
    return np.full(n, np.nan)
