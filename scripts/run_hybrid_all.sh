#!/usr/bin/env bash
#
# run_hybrid_all.sh
# ==================
# Step runner for the 7 hybrid-comparison experiments (hybrid_exp1 ..
# hybrid_exp3b), each a thin wrapper around run_comparative_hybrid_methods.py
# against the matching protocol. Harness functions (log/warn/die, run(),
# _STEP_ORDER, --step/--from/--dry-run) are copied verbatim from run_all.sh
# -- same conventions, same env var contract (METHOD_TIMEOUT, PYSR_*,
# JULIA_NUM_THREADS, RESULTS_DIR, EXPERIMENTS_DIR, SHARD_IDS/TASK_IDS) --
# so this can sit next to run_all.sh without inventing a second harness
# style.
#
# Which methods run on which step is NOT a free choice here: it mirrors
# the variant x experiment matrix already computed from the capability
# manifest (python experiment_protocol_hybrid.py --list-coverage).
# hybrid_exp1/1b (defi, non-pca)     -> methods 1 2 3 4 5 6 7 8
# hybrid_exp1_pca/1b_pca (defi, pca) -> methods 1 2 3 4 5 6 7 8  (assumed*)
# hybrid_exp2 (feynman)              -> methods 1 2 3 4 5 6 7
# hybrid_exp3/3b (nguyen12)          -> method  9
# * "assumed" = decision #3 (pca.py as pure fork), not yet byte-verified;
#   see experiment_protocol_hybrid.py --list-coverage legend.
#
# Run:
#   ./run_hybrid_all.sh                       # all 7 steps, in order
#   ./run_hybrid_all.sh --step hybrid_exp3    # just one step
#   ./run_hybrid_all.sh --from hybrid_exp2    # this step onward
#   ./run_hybrid_all.sh --dry-run             # print commands, run nothing

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
_RESULTS_RAW="${RESULTS_DIR:-${REPO_ROOT}/hypatiax/data/results}"
RESULTS_DIR="$(cd "$(dirname "${_RESULTS_RAW}")" 2>/dev/null && pwd)/$(basename "${_RESULTS_RAW}")" \
  || RESULTS_DIR="${REPO_ROOT}/hypatiax/data/results"
export RESULTS_DIR
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${REPO_ROOT}/hypatiax/experiments/benchmarks}"

# ── same env var contract run_all.sh already exports; the method classes
#    imported by run_comparative_hybrid_methods.py read these directly, so
#    they must not diverge from run_all.sh's names/defaults. ─────────────
export PYSR_GENERATIONS="${PYSR_GENERATIONS:-10000}"
export PYSR_POPULATIONS="${PYSR_POPULATIONS:-30}"
export PYSR_SEED="${PYSR_SEED:-42}"
export METHOD_TIMEOUT="${METHOD_TIMEOUT:-900}"
export LLM_METHOD_TIMEOUT="${LLM_METHOD_TIMEOUT:-120}"
export PYSR_FIT_WALL_TIMEOUT="${PYSR_FIT_WALL_TIMEOUT:-1200}"
export PYSR_FIT_GRACE_SECS="${PYSR_FIT_GRACE_SECS:-120}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-4}"
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes
export JOB_DEADLINE="${JOB_DEADLINE:-19800}"

# ── CI wiring contract ────────────────────────────────────────────────
# ci_runner_hybrid.yml's "plan" job computes PROTOCOL / PCA_FLAG / METHODS
# / EFFECTIVE_SEEDS per step and exports them as job-level env vars to the
# worker before it calls this script with --step. Each per-step block
# below now reads those (falling back to its own hardcoded default when a
# var is unset, so direct/local `./run_hybrid_all.sh --step ...` without
# the CI environment still behaves exactly as before). Previously these
# were computed by the yaml and only echoed, never actually forwarded --
# this line plus the per-step ${VAR:-default} / ${VAR-default} reads below
# close that gap.
export EFFECTIVE_SEEDS="${EFFECTIVE_SEEDS:-}"

# Smoke-scale knob (plan doc decision #5): unset = full scale. CI's smoke
# job sets HYBRID_SAMPLE_CASES=2 so a PR check doesn't spend full API budget.
HYBRID_SAMPLE_CASES="${HYBRID_SAMPLE_CASES:-}"
_SAMPLE_FLAG=""
[[ -n "${HYBRID_SAMPLE_CASES}" ]] && _SAMPLE_FLAG="--sample-cases ${HYBRID_SAMPLE_CASES}"

ONLY_STEP=""
FROM_STEP=""
DRY_RUN=false

_STEP_ORDER="hybrid_exp1 hybrid_exp1_pca hybrid_exp1b hybrid_exp1b_pca hybrid_exp2 hybrid_exp3 hybrid_exp3b"

while [[ $# -gt 0 ]]; do
  case $1 in
    --step)    ONLY_STEP="$2"; shift 2 ;;
    --from)    FROM_STEP="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *)
      BARE="$1"; shift
      if [[ " $_STEP_ORDER " == *" ${BARE} "* ]]; then
        ONLY_STEP="$BARE"
      else
        echo "Unknown arg: ${BARE}"
        echo "  Valid step names: ${_STEP_ORDER}"
        echo "  Flags: --step <step> | --from <step> | --dry-run"
        exit 1
      fi
      ;;
  esac
done

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[run_hybrid_all]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()  { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

run() {
  local step="$1" desc="$2"; shift 2
  [[ -n "$ONLY_STEP" && "$ONLY_STEP" != "$step" ]] && return 0
  if [[ -n "$FROM_STEP" ]]; then
    local skip=true
    for s in $_STEP_ORDER; do
      [[ "$s" == "$FROM_STEP" ]] && skip=false
      [[ "$s" == "$step"      ]] && break
    done
    [[ "$skip" == true ]] && return 0
  fi
  echo ""
  log "=== STEP: ${step} -- ${desc} ==="
  if [[ "$DRY_RUN" == true ]]; then
    echo "    [dry-run] $*"
  else
    "$@"
    log "--- DONE: ${step} ---"
  fi
}

# ── hybrid_exp1 : defi, full case set, methods 1-8 ──────────────────────
_EXP1_PROTOCOL="${PROTOCOL:-defi}"
_EXP1_METHODS="${METHODS:-1 2 3 4 5 6 7 8 ${PYSR_BASELINE:+10}}"
_EXP1_PCA="${PCA_FLAG-}"
run hybrid_exp1 "Hybrid comparison: defi, full case set (methods 1-8)" bash -c "
  cd '${REPO_ROOT}'
  mkdir -p '${RESULTS_DIR}/hybrid_comparison/exp1'
  python3 '${EXPERIMENTS_DIR}/run_comparative_hybrid_methods.py' \
    --protocol '${_EXP1_PROTOCOL}' \
    --methods ${_EXP1_METHODS} \
    --seeds \"${EFFECTIVE_SEEDS:-42}\" \
    ${_EXP1_PCA} \
    ${_SAMPLE_FLAG} \
    --output-dir '${RESULTS_DIR}/hybrid_comparison/exp1' \
    2>&1 | tee '${RESULTS_DIR}/hybrid_comparison/exp1/hybrid_exp1_run.log'
"

# ── hybrid_exp1_pca : defi, PCA 40/60 split, methods 1-8 (assumed*) ─────
_EXP1PCA_PROTOCOL="${PROTOCOL:-defi}"
_EXP1PCA_METHODS="${METHODS:-1 2 3 4 5 6 7 8 ${PYSR_BASELINE:+10}}"
_EXP1PCA_PCA="${PCA_FLAG---pca}"
run hybrid_exp1_pca "Hybrid comparison: defi, PCA 40/60 split (methods 1-8, assumed*)" bash -c "
  cd '${REPO_ROOT}'
  mkdir -p '${RESULTS_DIR}/hybrid_comparison/exp1_pca'
  python3 '${EXPERIMENTS_DIR}/run_comparative_hybrid_methods.py' \
    --protocol '${_EXP1PCA_PROTOCOL}' ${_EXP1PCA_PCA} \
    --methods ${_EXP1PCA_METHODS} \
    --seeds \"${EFFECTIVE_SEEDS:-42}\" \
    ${_SAMPLE_FLAG} \
    --output-dir '${RESULTS_DIR}/hybrid_comparison/exp1_pca' \
    2>&1 | tee '${RESULTS_DIR}/hybrid_comparison/exp1_pca/hybrid_exp1_pca_run.log'
"

# ── hybrid_exp1b : defi, seed sweep, methods 1-8 ────────────────────────
_EXP1B_PROTOCOL="${PROTOCOL:-defi}"
_EXP1B_METHODS="${METHODS:-1 2 3 4 5 6 7 8 ${PYSR_BASELINE:+10}}"
_EXP1B_PCA="${PCA_FLAG-}"
run hybrid_exp1b "Hybrid comparison: defi, seed sweep (methods 1-8)" bash -c "
  cd '${REPO_ROOT}'
  mkdir -p '${RESULTS_DIR}/hybrid_comparison/exp1b'
  _SHARD_TASKS='${SHARD_IDS:-${TASK_IDS:-}}'
  # SHARD_IDS/TASK_IDS arrive as a JSON array (e.g. [\"portfolio_seed42\",
  # \"portfolio_seed99\"]), not a bare space-separated list -- strip the
  # JSON punctuation before splitting, or the quotes/brackets/commas make
  # every line fail the ^portfolio_seedNN\$ anchor and sharding silently
  # falls back to the full default seed list every time.
  _SHARD_SEEDS=\$(printf '%s' \"\${_SHARD_TASKS}\" | tr -d '[]\"' | tr ', ' '\n\n' | grep -oE '^portfolio_seed[0-9]+\$' | sed 's/^portfolio_seed//' | paste -sd, -)
  if [[ -z \"\${_SHARD_SEEDS}\" ]]; then
    if [[ -n \"${EFFECTIVE_SEEDS}\" ]]; then
      echo '  [hybrid_exp1b] No portfolio_seedNN task IDs in SHARD_IDS/TASK_IDS -- using EFFECTIVE_SEEDS override.'
      _SHARD_SEEDS='${EFFECTIVE_SEEDS}'
    else
      echo '  [hybrid_exp1b] No portfolio_seedNN task IDs in SHARD_IDS/TASK_IDS -- using full default seed list.'
      _SHARD_SEEDS='42,99,123,777,2024'
    fi
  fi
  python3 '${EXPERIMENTS_DIR}/run_comparative_hybrid_methods.py' \
    --protocol '${_EXP1B_PROTOCOL}' \
    --methods ${_EXP1B_METHODS} \
    --seeds \"\${_SHARD_SEEDS}\" \
    ${_EXP1B_PCA} \
    ${_SAMPLE_FLAG} \
    --output-dir '${RESULTS_DIR}/hybrid_comparison/exp1b' \
    2>&1 | tee '${RESULTS_DIR}/hybrid_comparison/exp1b/hybrid_exp1b_run.log'
"

# ── hybrid_exp1b_pca : defi, PCA split, seed sweep (assumed*) ───────────
_EXP1BPCA_PROTOCOL="${PROTOCOL:-defi}"
_EXP1BPCA_METHODS="${METHODS:-1 2 3 4 5 6 7 8 ${PYSR_BASELINE:+10}}"
_EXP1BPCA_PCA="${PCA_FLAG---pca}"
run hybrid_exp1b_pca "Hybrid comparison: defi, PCA split, seed sweep (methods 1-8, assumed*)" bash -c "
  cd '${REPO_ROOT}'
  mkdir -p '${RESULTS_DIR}/hybrid_comparison/exp1b_pca'
  _SHARD_TASKS='${SHARD_IDS:-${TASK_IDS:-}}'
  # See hybrid_exp1b above: strip JSON array punctuation before splitting.
  _SHARD_SEEDS=\$(printf '%s' \"\${_SHARD_TASKS}\" | tr -d '[]\"' | tr ', ' '\n\n' | grep -oE '^portfolio_seed[0-9]+\$' | sed 's/^portfolio_seed//' | paste -sd, -)
  if [[ -z \"\${_SHARD_SEEDS}\" ]]; then
    if [[ -n \"${EFFECTIVE_SEEDS}\" ]]; then
      echo '  [hybrid_exp1b_pca] No portfolio_seedNN task IDs in SHARD_IDS/TASK_IDS -- using EFFECTIVE_SEEDS override.'
      _SHARD_SEEDS='${EFFECTIVE_SEEDS}'
    else
      echo '  [hybrid_exp1b_pca] No portfolio_seedNN task IDs in SHARD_IDS/TASK_IDS -- using full default seed list.'
      _SHARD_SEEDS='42,99,123,777,2024'
    fi
  fi
  python3 '${EXPERIMENTS_DIR}/run_comparative_hybrid_methods.py' \
    --protocol '${_EXP1BPCA_PROTOCOL}' ${_EXP1BPCA_PCA} \
    --methods ${_EXP1BPCA_METHODS} \
    --seeds \"\${_SHARD_SEEDS}\" \
    ${_SAMPLE_FLAG} \
    --output-dir '${RESULTS_DIR}/hybrid_comparison/exp1b_pca' \
    2>&1 | tee '${RESULTS_DIR}/hybrid_comparison/exp1b_pca/hybrid_exp1b_pca_run.log'
"

# ── hybrid_exp2 : feynman/benchmark_v2, per-domain, methods 1-7 ─────────
_EXP2_PROTOCOL="${PROTOCOL:-feynman}"
_EXP2_METHODS="${METHODS:-1 2 3 4 5 6 7 ${PYSR_BASELINE:+10}}"
run hybrid_exp2 "Hybrid comparison: feynman, per-domain (methods 1-7)" bash -c "
  cd '${REPO_ROOT}'
  mkdir -p '${RESULTS_DIR}/hybrid_comparison/exp2'
  python3 '${EXPERIMENTS_DIR}/run_comparative_hybrid_methods.py' \
    --protocol '${_EXP2_PROTOCOL}' \
    --domain all \
    --methods ${_EXP2_METHODS} \
    --seeds \"${EFFECTIVE_SEEDS:-42}\" \
    ${_SAMPLE_FLAG} \
    --output-dir '${RESULTS_DIR}/hybrid_comparison/exp2' \
    2>&1 | tee '${RESULTS_DIR}/hybrid_comparison/exp2/hybrid_exp2_run.log'
"

# ── hybrid_exp3 : nguyen12, seed 42, method 9 only ──────────────────────
run hybrid_exp3 "Hybrid comparison: nguyen12, seed 42 (method 9 only)" bash -c "
  cd '${REPO_ROOT}'
  mkdir -p '${RESULTS_DIR}/hybrid_comparison/exp3'
  python3 '${EXPERIMENTS_DIR}/run_comparative_hybrid_methods.py' \
    --protocol nguyen12 \
    --methods 9 ${PYSR_BASELINE:+10} \
    --seeds 42 \
    ${_SAMPLE_FLAG} \
    --output-dir '${RESULTS_DIR}/hybrid_comparison/exp3' \
    2>&1 | tee '${RESULTS_DIR}/hybrid_comparison/exp3/hybrid_exp3_run.log'
"

# ── hybrid_exp3b : nguyen12, multi-seed, method 9 only ──────────────────
run hybrid_exp3b "Hybrid comparison: nguyen12, multi-seed (method 9 only)" bash -c "
  cd '${REPO_ROOT}'
  mkdir -p '${RESULTS_DIR}/hybrid_comparison/exp3b'
  _SHARD_TASKS='${SHARD_IDS:-${TASK_IDS:-}}'
  _SHARD_SEEDS=\$(echo \"\${_SHARD_TASKS}\" | tr ' ' '\n' | grep -oE '^[0-9]+\$' | paste -sd, -)
  if [[ -z \"\${_SHARD_SEEDS}\" ]]; then
    echo '  [hybrid_exp3b] No numeric task IDs in SHARD_IDS/TASK_IDS -- using full default seed list.'
    _SHARD_SEEDS='99,123,777,2024'
  fi
  python3 '${EXPERIMENTS_DIR}/run_comparative_hybrid_methods.py' \
    --protocol nguyen12 \
    --methods 9 ${PYSR_BASELINE:+10} \
    --seeds \"\${_SHARD_SEEDS}\" \
    ${_SAMPLE_FLAG} \
    --output-dir '${RESULTS_DIR}/hybrid_comparison/exp3b' \
    2>&1 | tee '${RESULTS_DIR}/hybrid_comparison/exp3b/hybrid_exp3b_run.log'
"

log "All requested hybrid steps complete."
