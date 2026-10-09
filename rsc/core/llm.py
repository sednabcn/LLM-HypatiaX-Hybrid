"""Single LLM client (replaces the 3 duplicated _create_message_deterministic copies, report s7.1).

* model/temperature come from config/model_config.yml
* temperature=0 is dropped automatically for models that reject it (retry once)
* optional freeze cache (HYPATIAX_LLM_FREEZE_CACHE=path.json): identical (model, prompt) -> identical text
* MockLLM: deterministic offline stand-in for smoke runs without ANTHROPIC_API_KEY
Every call increments `api_calls` so the DB column run.api_calls is real, not guessed.
"""
from __future__ import annotations
import hashlib, json, os, pathlib
import yaml
_CFG = yaml.safe_load((pathlib.Path(__file__).resolve().parents[2] / "config" / "model_config.yml").read_text())["llm"]

class MockLLM:
    name = "mock"
    def __init__(self): self.api_calls = 0
    def complete(self, prompt: str, **_) -> str:
        self.api_calls += 1
        h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
        return f"def formula(x0, x1):\n    return {1 + h % 5}.0 * x0 * x1\n"

class LLMClient:
    def __init__(self, model: str | None = None, temperature: float | None = None):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model or _CFG["model"]; self.temperature = _CFG["temperature"] if temperature is None else temperature
        self.api_calls = 0; self._cache_path = os.environ.get("HYPATIAX_LLM_FREEZE_CACHE")
        self._cache = json.loads(pathlib.Path(self._cache_path).read_text()) if self._cache_path and pathlib.Path(self._cache_path).exists() else {}
    def complete(self, prompt: str, max_tokens: int = 1500) -> str:
        key = hashlib.sha256(f"{self.model}|{prompt}".encode()).hexdigest()
        if key in self._cache: return self._cache[key]
        kw = dict(model=self.model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
        try:
            r = self.client.messages.create(temperature=self.temperature, **kw)
        except Exception as e:                                   # models that deprecate `temperature`
            if "temperature" not in str(e).lower(): raise
            r = self.client.messages.create(**kw)
        self.api_calls += 1
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        if self._cache_path:
            self._cache[key] = text; pathlib.Path(self._cache_path).write_text(json.dumps(self._cache))
        return text

def get_client():
    return LLMClient() if os.environ.get("ANTHROPIC_API_KEY") else MockLLM()
