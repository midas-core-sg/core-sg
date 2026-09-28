#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv310/bin/python}"
THREADS="${THREADS:-1}"
BEANS_CSV="${BEANS_CSV:-}"
SEEDS_CSV="${SEEDS_CSV:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30}"

export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export NUMBA_NUM_THREADS="$THREADS"

BEANS_ARGS=()
if [[ -n "$BEANS_CSV" ]]; then
  if [[ ! -f "$BEANS_CSV" ]]; then
    echo "Beans CSV not found: $BEANS_CSV" >&2
    exit 1
  fi
  BEANS_ARGS=(--real-csv "beans=$BEANS_CSV" --csv-target-column beans:Class)
fi

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
for seed in "${SEEDS[@]}"; do
  seed="${seed//[[:space:]]/}"
  [[ -z "$seed" ]] && continue
  "$PYTHON_BIN" benchmarking/quality/quality_comparison.py \
    --output-dir benchmarking/quality/results \
    --benchmark-groups synthetic,real \
    --methods hdbscan_generic,optimized_hdbscan,score_sg,score_sg_random \
    --runtime-sample-sizes 5000,10000 \
    --runtime-dimensions 32 \
    --runtime-centers 10 \
    --sample-sizes 5000,10000 \
    --dimensions 2,32 \
    --distributions gaussian,gaussian_sparse \
    --k-max 50 \
    --k-min 2 \
    --seeds "$seed" \
    --random-state 42 \
    --resume \
    "${BEANS_ARGS[@]}" \
    "$@"
done
