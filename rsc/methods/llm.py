"""Single LLM client (replaces the 3 duplicated _create_message_deterministic copies, report s7.1).

Changes vs the version you uploaded (everything else is kept):
  * settings come from config/api_env.yml through ApiEnv (one source of truth)
  * `temperature` rejection is a policy: strict_temperature=true raises; false retries without it and
    RECORDS temperature_used=None, so a result row can never claim a temperature that was not applied
  * the freeze-cache key now includes temperature and max_tokens (before, changing temperature
    silently returned stale cached text)
  * token usage and optional USD budget are tracked; BudgetExceeded stops the run
  * a client can be injected (tests) and `meta()` returns what must be stored next to every result
  * FixtureLLM: scripted answers for tests / smoke runs; MockLLM now returns the JSON format the parser expects
"""
from __future__ import annotations
import hashlib, json, os, pathlib, re
from typing import Callable

from .env import ApiEnv


class BudgetExceeded(RuntimeError):
    pass


class TemperatureRejected(RuntimeError):
    pass


class _Base:
    name = "base"
    backend = "base"
    api_calls = 0

    def meta(self) -> dict:
        return {"backend": self.backend, "api_calls": self.api_calls}


class MockLLM(_Base):
    """Deterministic offline stand-in. NOT a language model: results from it are not scientific results."""
    name = "mock"
    backend = "mock"

    def __init__(self):
        self.api_calls = 0

    def complete(self, prompt: str, **_) -> str:
        self.api_calls += 1
        m = re.search(r"^Variables:\s*(.+)$", prompt, flags=re.M)
        names = [v.strip() for v in m.group(1).split(",")] if m else ["x0"]
        h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
        coef = 1 + h % 5
        expr = f"{coef}.0*{names[0]}*{names[1]}" if len(names) > 1 else f"{coef}.0*{names[0]}"
        return json.dumps([{"equation": f"y = {expr}", "confidence": 0.1, "reasoning": "mock"}])

    def meta(self) -> dict:
        return {"backend": "mock", "api_calls": self.api_calls, "model": "mock",
                "temperature_requested": None, "temperature_used": None}


class FixtureLLM(_Base):
    """Scripted answers: fixtures[case_id or 'default'] -> list of {equation, confidence, reasoning}.
    Used by tests and the smoke run so gate logic is exercised without an API key."""
    name = "fixture"
    backend = "fixture"

    def __init__(self, fixtures: dict, case_getter: Callable[[str], str | None] | None = None):
        self.fixtures = fixtures
        self.api_calls = 0
        self.prompts: list[str] = []
        self._case = case_getter or (lambda p: (re.search(r"^# case: (.+)$", p, flags=re.M) or [None, None])[1])

    def complete(self, prompt: str, **_) -> str:
        self.api_calls += 1
        self.prompts.append(prompt)
        cid = self._case(prompt)
        return json.dumps(self.fixtures.get(cid, self.fixtures.get("default", [])))

    def meta(self) -> dict:
        return {"backend": "fixture", "api_calls": self.api_calls, "model": "fixture",
                "temperature_requested": None, "temperature_used": None}


class LLMClient(_Base):
    name = "anthropic"
    backend = "live"

    def __init__(self, env: ApiEnv | None = None, client=None, model: str | None = None,
                 temperature: float | None = None):
        self.env = env or ApiEnv.load()
        self.model = model or self.env.model
        self.temperature = self.env.temperature if temperature is None else temperature
        self.temperature_used: float | None = self.temperature
        self.temperature_dropped = False
        self.api_calls = 0
        self.cache_hits = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.client = client if client is not None else self._make_client()
        self._cache_path = self.env.freeze_cache_path()
        p = pathlib.Path(self._cache_path) if self._cache_path else None
        self._cache = json.loads(p.read_text()) if p and p.exists() else {}

    # ---- construction -------------------------------------------------
    def _make_client(self):
        import anthropic  # lazy: only needed for live runs
        key = self.env.api_key()
        if not key:
            raise RuntimeError(f"{self.env.api_key_env} is not set")
        kw = dict(api_key=key, timeout=self.env.timeout_s, max_retries=self.env.max_retries)
        if self.env.base_url():
            kw["base_url"] = self.env.base_url()
        return anthropic.Anthropic(**kw)

    # ---- accounting ---------------------------------------------------
    def cost_usd(self) -> float | None:
        if self.env.price_in is None or self.env.price_out is None:
            return None
        return (self.tokens_in * self.env.price_in + self.tokens_out * self.env.price_out) / 1e6

    def _check_budget(self) -> None:
        if self.env.budget_usd is None:
            return
        c = self.cost_usd()
        if c is not None and c >= self.env.budget_usd:
            raise BudgetExceeded(f"spent ${c:.4f} >= budget ${self.env.budget_usd:.2f}")

    def meta(self) -> dict:
        return {"backend": "live", "model": self.model, "temperature_requested": self.temperature,
                "temperature_used": self.temperature_used, "temperature_dropped": self.temperature_dropped,
                "api_calls": self.api_calls, "cache_hits": self.cache_hits,
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out, "cost_usd": self.cost_usd()}

    # ---- the call -----------------------------------------------------
    def complete(self, prompt: str, max_tokens: int | None = None) -> str:
        max_tokens = max_tokens or self.env.max_tokens
        key = hashlib.sha256(f"{self.model}|{self.temperature}|{max_tokens}|{prompt}".encode()).hexdigest()
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self._check_budget()
        kw = dict(model=self.model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
        try:
            r = self.client.messages.create(temperature=self.temperature, **kw)
        except Exception as e:
            if "temperature" not in str(e).lower():
                raise
            if self.env.strict_temperature:
                raise TemperatureRejected(
                    f"{self.model} rejected temperature={self.temperature} and strict_temperature=true: {e}") from e
            self.temperature_dropped, self.temperature_used = True, None
            r = self.client.messages.create(**kw)
        self.api_calls += 1
        u = getattr(r, "usage", None)
        self.tokens_in += int(getattr(u, "input_tokens", 0) or 0)
        self.tokens_out += int(getattr(u, "output_tokens", 0) or 0)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        if self._cache_path:
            self._cache[key] = text
            tmp = pathlib.Path(self._cache_path + ".tmp")
            tmp.write_text(json.dumps(self._cache))
            os.replace(tmp, self._cache_path)
        return text


def get_client(env: ApiEnv | None = None):
    """Live client if the environment says so, otherwise the mock. Check `.backend` before trusting results."""
    env = env or ApiEnv.load()
    return LLMClient(env) if env.resolved_mode() == "live" else MockLLM()
