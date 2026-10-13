"""Optional RAG for the LLM step: retrieve reference formula cards and put them in the prompt.

Offline and dependency-free (TF-IDF + cosine) so a run is reproducible and costs no API calls.
OFF by default everywhere: turning it on changes the experiment, so it must be a declared ablation.

Leakage is the main risk of RAG in a benchmark (the knowledge base can contain the answer), so:
  * view_for_case(case_id, gt_formula) returns a retriever with that case's card and any card whose
    formula equals / contains the ground truth removed
  * audit_prompt_leak(prompt, gt_formula) is a final check the harness can run on every prompt sent
The Task object has no ground-truth field; the harness passes gt_formula to these two functions only.
"""
from __future__ import annotations
import math, pathlib, re
from dataclasses import dataclass
from collections import Counter

import yaml


@dataclass(frozen=True)
class Card:
    id: str
    name: str
    domain: str
    formula: str
    variables: tuple
    description: str
    source: str = ""


def normalize_formula(s: str) -> str:
    s = re.sub(r"^\s*[A-Za-z_]\w*\s*=\s*", "", s or "")
    return re.sub(r"\s+", "", s.replace("^", "**")).lower()


def _tokens(text: str) -> list:
    return re.findall(r"[a-z0-9_]+", (text or "").lower())


def leaks(gt_formula: str, card: Card) -> bool:
    g, c = normalize_formula(gt_formula), normalize_formula(card.formula)
    return bool(g) and (g == c or (len(g) > 6 and g in c))


def audit_prompt_leak(prompt: str, gt_formula: str) -> bool:
    """True if the ground-truth formula text appears in the prompt (after normalisation)."""
    g = normalize_formula(gt_formula)
    return bool(g) and len(g) > 3 and g in re.sub(r"\s+", "", prompt.replace("^", "**")).lower()


class Retriever:
    def __init__(self, cards: list, k: int = 3):
        self.cards, self.k = list(cards), k
        docs = [Counter(_tokens(f"{c.name} {c.domain} {c.description} {' '.join(c.variables)}")) for c in self.cards]
        n = len(docs)
        df = Counter(t for d in docs for t in d)
        self._idf = {t: math.log((n + 1) / (v + 1)) + 1.0 for t, v in df.items()}
        self._vecs = [self._vec(d) for d in docs]

    @classmethod
    def from_yaml(cls, path, k: int = 3) -> "Retriever":
        raw = yaml.safe_load(pathlib.Path(path).read_text()) or {}
        cards = [Card(id=str(c["id"]), name=c["name"], domain=c.get("domain", ""), formula=c["formula"],
                      variables=tuple(c.get("variables", [])), description=c.get("description", ""),
                      source=c.get("source", "")) for c in raw.get("cards", [])]
        return cls(cards, k=k)

    def _vec(self, counts: Counter) -> dict:
        v = {t: (1 + math.log(c)) * self._idf.get(t, 1.0) for t, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def view_for_case(self, case_id: str | None = None, gt_formula: str | None = None) -> "Retriever":
        keep = [c for c in self.cards
                if not (case_id and c.id == case_id) and not (gt_formula and leaks(gt_formula, c))]
        return Retriever(keep, k=self.k)

    def retrieve(self, query: str, k: int | None = None) -> list:
        q = self._vec(Counter(_tokens(query)))
        scored = sorted(((sum(w * v.get(t, 0.0) for t, w in q.items()), c.id, c)
                         for v, c in zip(self._vecs, self.cards)), key=lambda t: (-t[0], t[1]))
        return [c for s, _, c in scored[: (k or self.k)] if s > 0]

    def context(self, query: str, k: int | None = None) -> str:
        cards = self.retrieve(query, k)
        if not cards:
            return ""
        lines = [f"- {c.name} ({c.domain}): {c.formula}   [vars: {', '.join(c.variables)}]" for c in cards]
        return ("Reference formulas from a knowledge base (may be irrelevant; use only if they truly apply):\n"
                + "\n".join(lines))
