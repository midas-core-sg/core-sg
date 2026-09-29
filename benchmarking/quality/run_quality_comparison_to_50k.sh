#!/usr/bin/env bash

################################################################################
#
# CoreSG Quality Benchmark - Quick Runner (ARI and HAI to 50k)
#
# Simplified version of run_article_quality_to_50k.sh with predefined settings
# for the standard article quality experiments.
#
# Usage:
#   ./run_quality_comparison_to_50k.sh
#   THREADS=4 ./run_quality_comparison_to_50k.sh
#   SKIP_EXISTING=1 ./run_quality_comparison_to_50k.sh --limit 10
#
# Environment Variables (all optional):
#   PYTHON_BIN       - Path to Python interpreter (default: python3)
#   THREADS          - Number of threads (default: 1)
#   OUTPUT_DIR       - Results output directory (default: benchmarking/quality/results)
#   SKIP_EXISTING    - Skip configurations with existing results (default: 0)
#   LOG_LEVEL        - Logging verbosity: DEBUG, INFO, WARNING, ERROR (default: INFO)
#   BEANS_CSV        - Optional path to the Dry Bean CSV file
#   SEEDS_CSV        - Comma-separated seeds (default: 1..30)
#
# Additional Arguments:
#   Pass any arguments after the script name to quality_comparison.py
#   Example: ./run_quality_comparison_to_50k.sh --limit 10 --stop-on-error
#
################################################################################

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Thread control
PYTHON_BIN="${PYTHON_BIN:-python3}"
THREADS="${THREADS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarking/quality/results}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
BEANS_CSV="${BEANS_CSV:-}"
SEEDS_CSV="${SEEDS_CSV:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30}"

# Export thread settings to all numerical libraries
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export NUMBA_NUM_THREADS="$THREADS"

echo "================================== CoreSG Quality Benchmark =================================="
echo "Python: $PYTHON_BIN"
echo "Threads: $THREADS"
echo "Output: $OUTPUT_DIR"
echo "Skip existing: $SKIP_EXISTING"
echo "Log level: $LOG_LEVEL"
echo "Beans CSV: ${BEANS_CSV:-none}"
echo "Seeds: $SEEDS_CSV"
echo "Extra args: $*"
echo "=============================================================================================="

BEANS_ARGS=()
if [[ -n "$BEANS_CSV" ]]; then
  if [[ ! -f "$BEANS_CSV" ]]; then
    echo "Beans CSV not found: $BEANS_CSV" >&2
    exit 1
  fi
  BEANS_ARGS=(--real-csv "beans=$BEANS_CSV" --csv-target-column beans:Class)
fi

# Run the quality comparison for the article's configuration:
# - Groups: synthetic distributions and built-in real datasets
# - Methods: all four (HDBSCAN, Optimized HDBSCAN, ScoreSG, ScoreSG Random)
# - Sample sizes: 5k, 10k, 20k, 30k, 40k, 50k (capped at 50k for exact HDBSCAN)
# - Dimensions: 2, 10, 20, 32, 64, 128 (diagnostic)
# - Distributions: gaussian and gaussian_sparse
# - Seeds: one independent execution per seed
# - k parameters: k_min=2, k_max=50

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
for seed in "${SEEDS[@]}"; do
  seed="${seed//[[:space:]]/}"
  [[ -z "$seed" ]] && continue
  echo "Running seed: $seed"
  "$PYTHON_BIN" benchmarking/quality/quality_comparison.py \
    --output-dir "$OUTPUT_DIR" \
    --benchmark-groups synthetic,real \
    --methods hdbscan_generic,optimized_hdbscan,score_sg,score_sg_random \
    --sample-sizes 5000,10000,20000,30000,40000,50000 \
    --dimensions 2,10,20,32,64,128 \
    --distributions gaussian,gaussian_sparse \
    --k-min 2 \
    --k-max 50 \
    --seeds "$seed" \
    --random-state 42 \
    --resume \
    --log-level "$LOG_LEVEL" \
    "${BEANS_ARGS[@]}" \
    "$@"
done

echo "=============================================================================================="
echo "✓ Quality benchmark completed"
echo "Results saved to: $OUTPUT_DIR"
echo "Key outputs:"
echo "  - quality_comparison_by_k.csv"
echo "  - quality_comparison_summary.csv"
echo "  - quality_comparison_manifest.json"
echo "=============================================================================================="
