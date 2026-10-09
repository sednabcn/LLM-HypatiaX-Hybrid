#!/usr/bin/env bash
# Source me:  source scripts/env.sh [smoke|full]    (A-003, A-018)
# Points the copied scripts at (a) this repo's layout and (b) a checkout of the REAL LLM-HypatiaX-REPRO,
# which provides experiment_protocol_*.py, run_comparative_suite_benchmark_{v2,pca}.py and hypatiax.*
export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPRO_ROOT="${REPRO_ROOT:-$REPO_ROOT/external/LLM-HypatiaX-REPRO}"   # git clone https://github.com/sednabcn/LLM-HypatiaX-REPRO
export EXPERIMENTS_DIR="$REPO_ROOT/rsc/experiments/benchmarks"
export RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results}"
export GIT_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)"
REAL_BENCH="$REPRO_ROOT/hypatiax/experiments/benchmarks"
export PYTHONPATH="$REPO_ROOT/rsc/generation/variants:$EXPERIMENTS_DIR:$REAL_BENCH:$REPRO_ROOT/hypatiax/protocols:$REPRO_ROOT:$REPRO_ROOT/hypatiax:${PYTHONPATH:-}"
case "${1:-smoke}" in
  smoke) export HYBRID_SAMPLE_CASES=2 PYSR_GENERATIONS=200 EFFECTIVE_SEEDS=42 ;;
  full)  export EFFECTIVE_SEEDS="${EFFECTIVE_SEEDS:-42,43,44,45,46}" PYSR_BASELINE=1 ;;
esac
echo "[env] REPRO_ROOT=$REPRO_ROOT mode=${1:-smoke} seeds=${EFFECTIVE_SEEDS:-} results=$RESULTS_DIR"
[[ -d "$REAL_BENCH" ]] || echo "[env] WARNING: real repo not found at $REPRO_ROOT -> run_hybrid_all.sh will fail at import (A-016)"
