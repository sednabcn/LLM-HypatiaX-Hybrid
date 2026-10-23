import pathlib
import pytest

from rsc.methods.retrieval import Retriever, Card, audit_prompt_leak, leaks, normalize_formula
from rsc.methods.hypotheses import build_prompt, propose
from rsc.methods.stack import build_stack
from rsc.methods.env import ApiEnv
from rsc.methods.llm import FixtureLLM
from methods_helpers import make_task, stack_with

KB = pathlib.Path(__file__).resolve().parents[3] / "config" / "formula_kb.example.yml"


def test_retrieves_relevant_card_first():
    r = Retriever.from_yaml(KB, k=2)
    top = r.retrieve("enzyme reaction rate saturating with substrate concentration")
    assert top[0].id == "kb_michaelis"
    assert r.retrieve("zzzz qqqq") == []                     # nothing relevant -> no padding with random cards


def test_retrieval_is_deterministic():
    r = Retriever.from_yaml(KB)
    assert [c.id for c in r.retrieve("interest on a balance")] == [c.id for c in r.retrieve("interest on a balance")]


def test_leak_filter_removes_the_answer():
    r = Retriever.from_yaml(KB)
    assert any(c.id == "kb_ohm" for c in r.retrieve("current through a resistor"))
    view = r.view_for_case(case_id="other", gt_formula="y = V/R")          # the ground truth equals a card
    assert all(c.id != "kb_ohm" for c in view.cards)
    assert not any(leaks("V/R", c) for c in view.cards)
    assert any(c.id == "kb_ohm" for c in r.cards)                          # original retriever untouched
    view2 = r.view_for_case(case_id="kb_ohm")                              # the card of the case itself
    assert all(c.id != "kb_ohm" for c in view2.cards)


def test_prompt_audit_catches_a_leak():
    assert audit_prompt_leak("Reference: P*(1+r/n)**(n*t)", "P * (1 + r/n) ^ (n*t)")
    assert not audit_prompt_leak("Reference: 0.5*m*v**2", "n*R*T/V")


def test_rag_is_off_by_default_and_changes_the_prompt_only_when_on():
    task = make_task()
    s_off = stack_with(["x"])
    propose(s_off, task)
    assert "Reference formulas" not in s_off.llm.prompts[-1]
    assert s_off.provenance()["rag"] is False
    s_on = stack_with(["x"], retriever=Retriever.from_yaml(KB))
    t2 = make_task()
    object.__setattr__(t2, "description", "current through a resistor from voltage and resistance")
    propose(s_on, t2)
    assert "Reference formulas" in s_on.llm.prompts[-1] and "V/R" in s_on.llm.prompts[-1]
    assert s_on.provenance()["rag"] is True


def test_no_context_when_nothing_relevant():
    t = make_task()
    object.__setattr__(t, "description", "zzzz qqqq")
    s = stack_with(["x"], retriever=Retriever.from_yaml(KB))
    propose(s, t)
    assert "Reference formulas" not in s.llm.prompts[-1]
