"""The shared 'ask the LLM for candidate formulas' step used by every LLM-based variant."""
from __future__ import annotations
import json, re
from dataclasses import dataclass

import numpy as np

from .common import Task, r2, try_predict, equation_ok


@dataclass
class Hyp:
    equation: str
    confidence: float = 0.5
    reasoning: str = ""
    r2: float = float("-inf")          # on TRAINING data
    pred: np.ndarray | None = None     # training predictions (None if the equation failed to evaluate)


def clean_equation(eq: str) -> str:
    eq = str(eq or "")
    if "=" in eq:
        eq = eq.split("=", 1)[1]
    eq = eq.strip().replace("^", "**")
    return re.sub(r"\b(np|numpy|math)\.", "", eq)


def build_prompt(task: Task, n: int = 1, context: str = "") -> str:
    ctx = f"{context}\n" if context else ""
    case = f"# case: {task.case_id}\n" if task.case_id else ""      # lets scripted test LLMs key on the case
    return (f"{case}You are an expert scientific equation discovery system.\n"
            f"Domain: {task.domain}\nDescription: {task.description}\n"
            f"Variables: {', '.join(task.names)}\n{ctx}"
            f"Generate {n} candidate equations using Python syntax (** / * / + / -) and only the functions "
            f"sqrt, log, exp, sin, cos, tan, abs.\n"
            'Return ONLY a JSON array: [{"equation": "y = ...", "confidence": 0.9, "reasoning": "..."}]')


def parse_hypotheses(text: str) -> list:
    try:
        if "```" in text:
            body = text.split("```", 2)[1]
            body = body[4:] if body.lstrip().startswith("json") else body
        else:
            body = text[text.find("["): text.rfind("]") + 1]
        data = json.loads(body)
        return [Hyp(clean_equation(c.get("equation", "")), float(c.get("confidence", 0.5)), str(c.get("reasoning", "")))
                for c in data if isinstance(c, dict)]
    except Exception:
        return []


def propose(stack, task: Task, n: int = 1) -> list:
    """Ask the LLM, parse, and score every candidate on the training data. Unusable equations are kept
    with pred=None / r2=-inf so callers can see (and count) failures instead of crashing on them."""
    context = ""
    if getattr(stack, "retriever", None) is not None:
        context = stack.retriever.context(f"{task.description} {task.domain} {' '.join(task.names)}")
    hyps = parse_hypotheses(stack.llm.complete(build_prompt(task, n, context)))
    for h in hyps:
        if equation_ok(h.equation, task.names):
            h.pred = try_predict(h.equation, task.X, task.names)
        if h.pred is not None:
            h.r2 = r2(task.y, h.pred)
        else:
            h.pred, h.r2 = None, float("-inf")
    return hyps


def best_usable(hyps: list) -> Hyp | None:
    ok = [h for h in hyps if h.pred is not None and np.isfinite(h.r2)]
    return max(ok, key=lambda h: h.r2) if ok else None
