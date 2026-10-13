"""Registry keyed by the FROZEN variant ids of config/variants.yml (B_LLM, B_NN, B_PYSR, V1..V7).
make("V4", mode="seed") builds a variant with overrides; "V4_seed" etc. are shorthands for the other V4 modes."""
from __future__ import annotations

from .baselines import PureLLM, NeuralNet, PurePySR
from .gates import V1EnhancedHybridDeFi, V2HybridAllDomains, V3Ensemble
from .search_variants import V4SymbolicWithLLM, V5RetryDiscovery, V7LLMSeededSearch
from .validation_selected import V6ValidationSelected
from .base import Variant

REGISTRY = {
    "B_LLM": (PureLLM, {}), "B_NN": (NeuralNet, {}), "B_PYSR": (PurePySR, {}),
    "V1": (V1EnhancedHybridDeFi, {}), "V2": (V2HybridAllDomains, {}), "V3": (V3Ensemble, {}),
    "V4": (V4SymbolicWithLLM, {"mode": "hybrid"}),
    "V4_seed": (V4SymbolicWithLLM, {"mode": "seed"}), "V4_fallback": (V4SymbolicWithLLM, {"mode": "fallback"}),
    "V5": (V5RetryDiscovery, {}), "V6": (V6ValidationSelected, {}), "V7": (V7LLMSeededSearch, {}),
}


def make(variant_id: str, **overrides) -> Variant:
    cls, defaults = REGISTRY[variant_id]
    v = cls(**{**defaults, **overrides})
    v.id = variant_id.split("_")[0] if variant_id.startswith("V4_") else variant_id
    return v


__all__ = ["REGISTRY", "make", "Variant"]
