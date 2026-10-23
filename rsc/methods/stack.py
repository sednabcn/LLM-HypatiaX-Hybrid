"""Stack = the three imported components (llm / nn / pysr) + optional retriever, built from config.

    stack = build_stack()                       # development: auto mode, substitute search if no PySR
    stack = build_stack(strict=True)            # anchor / paper runs: live LLM, real PySR, torch NN or refuse
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field

from .env import ApiEnv, EnvError
from .llm import get_client
from .nn import NNBaseline
from .search import PySRSearch, SubstituteSearch
from . import pysr as pysr_mod


@dataclass
class Stack:
    llm: object
    search: object
    env: ApiEnv
    retriever: object | None = None

    def new_nn(self, task, hidden=64, epochs: int = 400) -> NNBaseline:
        return NNBaseline(task.description, epochs=epochs, hidden=hidden, run_seed=task.seed)

    def provenance(self) -> dict:
        """Store this next to every result row so a number can always be traced to what produced it."""
        try:
            import torch  # noqa: F401
            nn_backend = "torch"
        except ImportError:
            nn_backend = "sklearn"
        return {"llm": self.llm.meta(), "search_engine": self.search.engine, "nn_backend": nn_backend,
                "rag": self.retriever is not None, "api_env": self.env.public_dict()}


def build_stack(env: ApiEnv | None = None, search: str = "auto", retriever=None, llm=None,
                strict: bool = False, pysr_populations: int | None = None, pysr_generations: int | None = None,
                pysr_timeout_s: int | None = None) -> Stack:
    env = env or ApiEnv.load()
    if strict and env.mode != "live":
        raise EnvError("strict stack requires api_env.yml mode: live")
    client = llm if llm is not None else get_client(env)
    if strict and getattr(client, "backend", "") != "live":
        raise EnvError(f"strict stack requires the live LLM, got backend={getattr(client, 'backend', '?')}")
    if pysr_populations is not None:
        os.environ["PYSR_POPULATIONS"] = str(pysr_populations)
    if pysr_generations is not None:
        os.environ["PYSR_GENERATIONS"] = str(pysr_generations)
    use_pysr = search == "pysr" or (search == "auto" and pysr_mod.available())
    if strict and not pysr_mod.available():
        raise EnvError("strict stack requires PySR (+ Julia)")
    engine = PySRSearch(niterations=pysr_generations, timeout_s=pysr_timeout_s) if use_pysr else SubstituteSearch()
    stack = Stack(llm=client, search=engine, env=env, retriever=retriever)
    if strict and stack.provenance()["nn_backend"] != "torch":
        raise EnvError("strict stack requires torch: the paper's NN numbers come from the torch path")
    return stack
