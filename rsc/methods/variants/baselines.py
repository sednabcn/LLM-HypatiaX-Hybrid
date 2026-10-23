from __future__ import annotations
import numpy as np

from ..common import FitResult, Task, r2, try_predict
from ..hypotheses import propose, best_usable
from .base import Variant, nan_predict


class PureLLM(Variant):
    id, kind, name = "B_LLM", "baseline", "Pure LLM"

    def fit(self, task: Task, stack) -> FitResult:
        h = best_usable(propose(stack, task, n=self.p("n_candidates", 1)))
        if h is None:
            return FitResult(self.id, "llm_unusable", lambda X: nan_predict(len(X)), float("-inf"),
                             info={"llm_failed": True})
        eq = h.equation

        def predict(X):
            p = try_predict(eq, X, task.names)
            return p if p is not None else nan_predict(len(X))
        return FitResult(self.id, "llm", predict, h.r2, equation=eq)


class NeuralNet(Variant):
    id, kind, name = "B_NN", "baseline", "Neural network"

    def fit(self, task: Task, stack) -> FitResult:
        nn = stack.new_nn(task).fit(task.X, task.y)
        return FitResult(self.id, "nn", nn.predict, r2(task.y, nn.predict(task.X)),
                         info={"nn_backend": nn.backend, "nn_seed": nn.seed})


class PurePySR(Variant):
    id, kind, name = "B_PYSR", "baseline", "Pure symbolic search (V4 mode=none)"

    def fit(self, task: Task, stack) -> FitResult:
        r = stack.search.run(task.X, task.y, task.names, seed=task.seed, unary=self.p("unary", ("sq", "sqrt", "log", "inv", "sin", "cos")))
        return FitResult(self.id, "search", r.predict, r.r2, equation=r.expression, engine=r.engine, search_calls=1)
