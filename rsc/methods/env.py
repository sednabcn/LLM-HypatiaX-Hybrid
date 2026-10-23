"""API environment loader. One typed object built from config/api_env.yml + the process environment.

Design rules (each one is covered by a test):
  * the key is looked up, never stored in the config, never included in public_dict()/repr
  * mode=live with no key is an error, not a silent downgrade to the mock
  * a budget without prices is an error (we do not invent prices)
"""
from __future__ import annotations
import os
import pathlib
from dataclasses import dataclass, field
from typing import Mapping

import yaml

ROOT = pathlib.Path(os.environ.get("HX_ROOT") or pathlib.Path(__file__).resolve().parents[2])   # repo root: rsc/methods/env.py -> parents[2]
DEFAULT_CONFIG = ROOT / "config" / "api_env.yml"


class EnvError(RuntimeError):
    pass


def _read_dotenv(path: pathlib.Path) -> dict:
    out: dict = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if v:
            out[k.strip()] = v
    return out


@dataclass
class ApiEnv:
    provider: str = "anthropic"
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url_env: str | None = "ANTHROPIC_BASE_URL"
    dotenv: str | None = ".env"
    mode: str = "auto"
    model: str = "claude-sonnet-5-5"
    temperature: float = 0.0
    strict_temperature: bool = True
    max_tokens: int = 1500
    timeout_s: float = 60.0
    max_retries: int = 3
    budget_usd: float | None = None
    price_in: float | None = None
    price_out: float | None = None
    freeze_cache_env: str = "HYPATIAX_LLM_FREEZE_CACHE"
    _root: pathlib.Path = field(default=ROOT, repr=False)

    # ---- construction -------------------------------------------------
    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "ApiEnv":
        path = pathlib.Path(path or os.environ.get("HX_API_ENV") or DEFAULT_CONFIG)
        raw = yaml.safe_load(path.read_text()) if path.is_file() else {}
        raw = raw or {}
        price = raw.get("price_usd_per_mtok") or {}
        env = cls(
            provider=raw.get("provider", "anthropic"),
            api_key_env=raw.get("api_key_env", "ANTHROPIC_API_KEY"),
            base_url_env=raw.get("base_url_env"),
            dotenv=raw.get("dotenv"),
            mode=raw.get("mode", "auto"),
            model=raw.get("model", "claude-sonnet-5-5"),
            temperature=float(raw.get("temperature", 0.0)),
            strict_temperature=bool(raw.get("strict_temperature", True)),
            max_tokens=int(raw.get("max_tokens", 1500)),
            timeout_s=float(raw.get("timeout_s", 60)),
            max_retries=int(raw.get("max_retries", 3)),
            budget_usd=raw.get("budget_usd"),
            price_in=price.get("input"),
            price_out=price.get("output"),
            freeze_cache_env=raw.get("freeze_cache_env", "HYPATIAX_LLM_FREEZE_CACHE"),
            _root=path.resolve().parents[1] if path.is_file() else ROOT,
        )
        env.validate()
        return env

    def validate(self) -> None:
        if self.mode not in ("auto", "live", "mock"):
            raise EnvError(f"mode must be auto|live|mock, got {self.mode!r}")
        if self.provider != "anthropic":
            raise EnvError(f"only provider=anthropic is implemented, got {self.provider!r}")
        if self.budget_usd is not None and (self.price_in is None or self.price_out is None):
            raise EnvError("budget_usd is set but price_usd_per_mtok.input/output are null: "
                           "fill in the prices from your console (they are not guessed here)")

    # ---- secrets ------------------------------------------------------
    def _dotenv_values(self) -> dict:
        if not self.dotenv:
            return {}
        p = pathlib.Path(self.dotenv)
        return _read_dotenv(p if p.is_absolute() else self._root / p)

    def api_key(self, environ: Mapping[str, str] | None = None) -> str | None:
        environ = os.environ if environ is None else environ
        return environ.get(self.api_key_env) or self._dotenv_values().get(self.api_key_env) or None

    def base_url(self, environ: Mapping[str, str] | None = None) -> str | None:
        if not self.base_url_env:
            return None
        environ = os.environ if environ is None else environ
        return environ.get(self.base_url_env) or self._dotenv_values().get(self.base_url_env) or None

    def freeze_cache_path(self, environ: Mapping[str, str] | None = None) -> str | None:
        environ = os.environ if environ is None else environ
        return environ.get(self.freeze_cache_env) or None

    # ---- mode ---------------------------------------------------------
    def resolved_mode(self, environ: Mapping[str, str] | None = None) -> str:
        has_key = self.api_key(environ) is not None
        if self.mode == "live":
            if not has_key:
                raise EnvError(f"mode=live but {self.api_key_env} is not set (environment or {self.dotenv})")
            return "live"
        if self.mode == "mock":
            return "mock"
        return "live" if has_key else "mock"

    def public_dict(self) -> dict:
        """Safe to write into result rows / logs: never contains the key."""
        return {"provider": self.provider, "model": self.model, "temperature": self.temperature,
                "strict_temperature": self.strict_temperature, "mode": self.mode,
                "max_tokens": self.max_tokens, "budget_usd": self.budget_usd}
