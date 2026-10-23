"""PySR wrapper: guarded import, capability probe for `guesses` (V7 needs it; report s4.6 says V4 'seed' mode only
works if the installed build exposes it -- silently ignoring it would repeat the use_llm dead-path bug)."""
from __future__ import annotations
import inspect, os

def available() -> bool:
    try:
        import pysr  # noqa: F401
        return True
    except Exception:
        return False

def supports_guesses() -> bool:
    if not available(): return False
    from pysr import PySRRegressor
    return "guesses" in inspect.signature(PySRRegressor.__init__).parameters

def make_regressor(seed: int = 42, generations: int | None = None, guesses=None, **kw):
    """Raise (never ignore) when guesses are requested but unsupported."""
    if not available(): raise RuntimeError("PySR/Julia not installed: real experiments need them (config/installation.yml)")
    from pysr import PySRRegressor
    if guesses and not supports_guesses():
        raise RuntimeError("installed PySR has no `guesses` argument: refusing to run 'seeded' as unseeded (audit A-020)")
    args = dict(populations=int(os.environ.get("PYSR_POPULATIONS", 30)), niterations=int(generations or os.environ.get("PYSR_GENERATIONS", 10000)),
                random_state=seed, deterministic=True, parallelism="serial", **kw)
    if guesses: args["guesses"] = guesses
    return PySRRegressor(**args)
