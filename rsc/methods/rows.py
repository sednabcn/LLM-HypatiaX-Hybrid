"""FitResult -> one result row carrying what PLAYBOOK_ANCHOR rule 4 (comparability key) and the ingest guards need.

Field names follow the repo docs (seed, engine_version, api_calls, wall_s, decision, r2_far, llm model + temperature,
case-set hash, split, threshold, config hash). scripts/ingest_results.py is NOT in the zip this was built against:
confirm the exact names against it before the first smoke run (PLAYBOOK 6.1).

Rules enforced here, not left to the caller:
  * r2_far is RECOMPUTED from the raw far-range predictions; the method's own train_r2 is stored separately.
  * engine_version, case_set_hash, split and threshold have no defaults: a row cannot be made without them.
  * a row produced with the mock/fixture LLM, the substitute search or the sklearn NN is marked synthetic=True
    (with the reasons), so the ingest guard "synthetic rows refused" can see it. Nothing here ever claims a row is REAL;
    only ingest does.
"""
from __future__ import annotations
import hashlib, json

from .common import FitResult, Task, r2, safe_predict

# First alias of each id in the FROZEN config/variants.yml (the regexes ingest matches against `method`).
METHOD_NAME = {"B_LLM": "PureLLM", "B_NN": "ImprovedNN", "B_PYSR": "pysr_none",
               "V1": "EnhancedHybridDeFi", "V2": "HybridAllDomains", "V3": "EnsembleLLMNN",
               "V4": "SymbolicEngine", "V5": "HybridSystemV50", "V6": "V4Selector", "V7": "LLMPriorSeeding"}


def config_hash(stack, fit: FitResult) -> str:
    blob = {"api_env": stack.env.public_dict(), "variant_params": fit.info.get("variant_params", {}),
            "search_engine": stack.search.engine, "rag": stack.retriever is not None}
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:16]


def result_row(variant_id: str, task: Task, fit: FitResult, stack, *, X_far, y_far, engine_version: str,
               case_set_hash: str, split: str, threshold: float, X_test=None, y_test=None) -> dict:
    prov = stack.provenance()
    llm = prov["llm"]
    reasons = []
    if llm.get("backend") != "live":
        reasons.append(f"llm_backend={llm.get('backend')}")
    if prov["search_engine"] != "pysr":
        reasons.append(f"search_engine={prov['search_engine']}")
    if prov["nn_backend"] != "torch":
        reasons.append(f"nn_backend={prov['nn_backend']}")
    cfg = config_hash(stack, fit)
    row = {
        "method_id": variant_id, "method": METHOD_NAME[variant_id.split("_")[0] if variant_id.startswith("V4_") else variant_id],
        "case_id": task.case_id, "seed": task.seed, "split": split, "threshold": threshold,
        "engine_version": engine_version, "case_set_hash": case_set_hash, "config_hash": cfg,
        "llm_model": llm.get("model"), "temperature_requested": llm.get("temperature_requested"),
        "temperature_used": llm.get("temperature_used"),
        "decision": fit.decision, "equation": fit.equation,
        "r2_far": r2(y_far, safe_predict(fit, X_far)),
        "r2_test": r2(y_test, safe_predict(fit, X_test)) if X_test is not None else None,
        "train_r2_reported": fit.train_r2, "selects_on": fit.info.get("selects_on"),
        "api_calls": fit.api_calls, "search_calls": fit.search_calls, "wall_s": fit.info.get("seconds"),
        "synthetic": bool(reasons), "synthetic_reasons": reasons,
        "provenance": prov,
    }
    row["comparability_key"] = [case_set_hash, split, task.seed, threshold, engine_version,
                                f"{row['llm_model']}@{row['temperature_requested']}", cfg]
    row["info"] = {k: v for k, v in fit.info.items() if k != "variant_params"}
    return row
