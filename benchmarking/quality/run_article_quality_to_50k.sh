#!/usr/bin/env bash

################################################################################
#
# CoreSG Quality Benchmark Runner (ARI and HAI)
# 
# This script runs clustering quality experiments for the article to 50k samples.
# It measures quality metrics (ARI and HAI) across various configurations:
# - Gaussian and sparse Gaussian synthetic distributions
# - Multiple dimensions
# - Multiple sample sizes (capped at 50k)
# - Compares: HDBSCAN, Optimized HDBSCAN, ScoreSG, ScoreSG Random
#
# Features:
#   - Respects SKIP_EXISTING=1 to resume partial runs
#   - Environment variable overrides for all major parameters
#   - Hooks for disabling specific method families
#   - Thread control via THREADS environment variable
#
################################################################################

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

################################################################################
# Configuration Parameters (override via environment variables)
################################################################################

# Sample sizes for quality experiments (capped at 50k for exact HDBSCAN)
SAMPLES_CSV="${SAMPLES_CSV:-5000,10000,20000,40000,50000}"

# Dimensions to test
DIMENSIONS_CSV="${DIMENSIONS_CSV:-2,10,32,64}"

# Distributions to test
# Options: gaussian, poisson, chi_square, gamma, beta, von_mises, gumbel, logistic, gaussian_sparse
DISTRIBUTIONS_CSV="${DISTRIBUTIONS_CSV:-gaussian,gaussian_sparse}"

# Runtime-specific parameters (for the "runtime" benchmark group)
# These match the runtime experiments configuration
RUNTIME_SAMPLES_CSV="${RUNTIME_SAMPLES_CSV:-5000,10000,20000,30000,40000,50000}"
RUNTIME_DIMENSIONS_CSV="${RUNTIME_DIMENSIONS_CSV:-20}"
RUNTIME_CENTERS="${RUNTIME_CENTERS:-10}"

# Clustering parameters
K_MAX="${K_MAX:-50}"
K_MIN="${K_MIN:-2}"

# Random seeds for repetitions. RANDOM_STATE is only a fallback for datasets
# without a seed; quality_comparison.py uses the current execution seed for
# stochastic ScoreSG components when one is available.
RANDOM_STATE="${RANDOM_STATE:-42}"
SEEDS_CSV="${SEEDS_CSV:-1,2,3,4,5,6,7,8,9,10}"

# Python interpreter
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Output directory for results
OUTPUT_DIR="${OUTPUT_DIR:-benchmarking/quality/results}"

# Optional Dry Bean CSV dataset path. When set, the dataset is added as
# --real-csv beans=... and the Class column is used as the external target.
BEANS_CSV="${BEANS_CSV:-}"

# Logging level
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Number of threads (exported to numpy/sklearn/etc)
THREADS="${THREADS:-1}"

# Toggle method families:
#   Set to 0 to skip running that method family
#   RUN_HDBSCAN_FAMILY=1 runs: hdbscan_generic, optimized_hdbscan
#   RUN_SCORESG_FAMILY=1 runs: score_sg, score_sg_random
RUN_HDBSCAN_FAMILY="${RUN_HDBSCAN_FAMILY:-1}"
RUN_SCORESG_FAMILY="${RUN_SCORESG_FAMILY:-1}"

# Resume mode: set SKIP_EXISTING=1 to resume a partial run
# (skips configurations that have results files already)
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# Separated clusters configuration (for special separated test cases)
SEPARATED_CLUSTERS="${SEPARATED_CLUSTERS:-6}"
SEPARATED_CLUSTER_SCALE="${SEPARATED_CLUSTER_SCALE:-0.65}"
SEPARATED_CLUSTER_SEPARATION="${SEPARATED_CLUSTER_SEPARATION:-8.0}"

# Optional real datasets to include, disabled by default for the article runs.
REAL_DATASETS_CSV="${REAL_DATASETS_CSV:-}"

################################################################################
# Thread Control (exported to external libraries)
################################################################################

export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export NUMBA_NUM_THREADS="$THREADS"

################################################################################
# Helper Functions
################################################################################

log_info() {
  echo "[$SCRIPT_NAME] INFO: $*" >&2
}

log_error() {
  echo "[$SCRIPT_NAME] ERROR: $*" >&2
}

log_separator() {
  echo "================================================================================" >&2
}

# Parse CSV lists from environment/parameters
parse_csv_list() {
  local csv="$1"
  local IFS=','
  read -r -a arr <<< "$csv"
  for item in "${arr[@]}"; do
    echo -n "${item#"${item%%[![:space:]]*}"} "
  done
  echo
}

# Validate that a sample size does not exceed 50k (exact HDBSCAN limit)
validate_sample_size() {
  local n_samples="$1"
  if (( n_samples > 50000 )); then
    log_error "n_samples=$n_samples exceeds 50,000 (exact HDBSCAN limit)"
    exit 1
  fi
}

# Build a descriptive label for the run configuration
# If benchmarks have single values, keep label simple; otherwise add all parameters
make_config_label() {
  local base_label="$1"
  local n_dimensions="$2"
  local n_samples="$3"
  
  # If all parameters are fixed (single value), use just the base label
  if [[ "${#DIMS[@]}" -eq 1 && "${#SAMPLES[@]}" -eq 1 ]]; then
    printf '%s' "$base_label"
  else
    # Otherwise encode dimensions and samples in the label
    printf '%s_d%d_n%d' "$base_label" "$n_dimensions" "$n_samples"
  fi
}

# Validate script prerequisites
validate_script() {
  if [[ ! -f "benchmarking/quality/quality_comparison.py" ]]; then
    log_error "quality_comparison.py not found. Please run from repository root."
    exit 1
  fi
  
  if ! command -v "$PYTHON_BIN" &> /dev/null; then
    log_error "Python interpreter not found: $PYTHON_BIN"
    exit 1
  fi
}

# Display configuration summary
print_config_summary() {
  log_separator
  log_info "Configuration Summary"
  log_separator
  log_info "Samples: $SAMPLES_CSV"
  log_info "Dimensions: $DIMENSIONS_CSV"
  log_info "Distributions: $DISTRIBUTIONS_CSV"
  log_info "Seeds: $SEEDS_CSV"
  log_info "Real datasets: ${REAL_DATASETS_CSV:-none}"
  log_info "Beans CSV: ${BEANS_CSV:-none}"
  log_info "Runtime Parameters:"
  log_info "  - Samples: $RUNTIME_SAMPLES_CSV"
  log_info "  - Dimensions: $RUNTIME_DIMENSIONS_CSV"
  log_info "  - Centers: $RUNTIME_CENTERS"
  log_info "Clustering Parameters: k_min=$K_MIN, k_max=$K_MAX"
  log_info "Random State: $RANDOM_STATE"
  log_info "Threads: $THREADS"
  log_info "Output Directory: $OUTPUT_DIR"
  log_info "Run HDBSCAN family: $RUN_HDBSCAN_FAMILY"
  log_info "Run ScoreSG family: $RUN_SCORESG_FAMILY"
  log_info "Skip existing: $SKIP_EXISTING"
  log_separator
}

################################################################################
# Main Execution
################################################################################

SCRIPT_NAME="$(basename "$0")"

# Validate prerequisites
validate_script

# Parse CSV parameters into arrays
read -r -a SAMPLES <<< "$(parse_csv_list "$SAMPLES_CSV")"
read -r -a DIMS <<< "$(parse_csv_list "$DIMENSIONS_CSV")"
read -r -a DISTS <<< "$(parse_csv_list "$DISTRIBUTIONS_CSV")"
read -r -a RUNTIME_SAMPLES <<< "$(parse_csv_list "$RUNTIME_SAMPLES_CSV")"
read -r -a RUNTIME_DIMS <<< "$(parse_csv_list "$RUNTIME_DIMENSIONS_CSV")"
read -r -a SEEDS <<< "$(parse_csv_list "$SEEDS_CSV")"

REAL_DATASET_ARGS=()
if [[ -n "$REAL_DATASETS_CSV" ]]; then
  REAL_DATASET_ARGS=(--real-datasets "$REAL_DATASETS_CSV")
else
  REAL_DATASET_ARGS=(--no-builtin-real)
fi

BEANS_ARGS=()
if [[ -n "$BEANS_CSV" ]]; then
  if [[ ! -f "$BEANS_CSV" ]]; then
    log_error "Beans CSV not found: $BEANS_CSV"
    exit 1
  fi
  BEANS_ARGS=(--real-csv "beans=$BEANS_CSV" --csv-target-column beans:Class)
fi

METHODS_CSV=""
if [[ "$RUN_HDBSCAN_FAMILY" == "1" && "$RUN_SCORESG_FAMILY" == "1" ]]; then
  METHODS_CSV="hdbscan_generic,optimized_hdbscan,score_sg,score_sg_random"
elif [[ "$RUN_HDBSCAN_FAMILY" == "1" ]]; then
  METHODS_CSV="hdbscan_generic,optimized_hdbscan"
elif [[ "$RUN_SCORESG_FAMILY" == "1" ]]; then
  METHODS_CSV="hdbscan_generic,score_sg,score_sg_random"
else
  log_error "At least one method family must be enabled."
  exit 1
fi

# Print configuration
print_config_summary

# Create output directory
mkdir -p "$OUTPUT_DIR"

log_info "Starting quality benchmark runs..."
log_separator

################################################################################
# Main Loop: Iterate over configurations
#
# Structure:
#   For each sample size
#     For each dimension
#       For each distribution
#         For each seed
#           Run one benchmark command for the current seed
################################################################################

TOTAL_CONFIGS=0
COMPLETED_CONFIGS=0
SKIPPED_CONFIGS=0

for n_samples in "${SAMPLES[@]}"; do
  # Validate against exact HDBSCAN limit
  validate_sample_size "$n_samples"

  for n_dimensions in "${DIMS[@]}"; do
    for distribution in "${DISTS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        seed="${seed//[[:space:]]/}"
        [[ -z "$seed" ]] && continue
        config_label="$(make_config_label "synthetic_${distribution}" "$n_dimensions" "$n_samples")_seed${seed}"

        ((++TOTAL_CONFIGS))

        log_separator
        log_info "Configuration: $config_label"
        log_info "Parameters: dist=$distribution, dim=$n_dimensions, n=$n_samples, seed=$seed"
        log_info "Methods: $METHODS_CSV"

        "$PYTHON_BIN" benchmarking/quality/quality_comparison.py \
          --output-dir "$OUTPUT_DIR" \
          --benchmark-groups synthetic \
          --methods "$METHODS_CSV" \
          --sample-sizes "$n_samples" \
          --dimensions "$n_dimensions" \
          --distributions "$distribution" \
          --k-min "$K_MIN" \
          --k-max "$K_MAX" \
          --seeds "$seed" \
          --random-state "$RANDOM_STATE" \
          --resume \
          "$@" || log_error "Methods failed for $config_label"

        ((++COMPLETED_CONFIGS))
        log_info "Configuration completed: $config_label"
      done
    done
  done
done

if [[ -n "$REAL_DATASETS_CSV" || -n "$BEANS_CSV" ]]; then
  for seed in "${SEEDS[@]}"; do
    seed="${seed//[[:space:]]/}"
    [[ -z "$seed" ]] && continue
    ((++TOTAL_CONFIGS))
    log_separator
    log_info "Configuration: real datasets seed$seed"
    log_info "Parameters: datasets=$REAL_DATASETS_CSV, seed=$seed"
    log_info "Methods: $METHODS_CSV"

    "$PYTHON_BIN" benchmarking/quality/quality_comparison.py \
      --output-dir "$OUTPUT_DIR" \
      --benchmark-groups real \
      --methods "$METHODS_CSV" \
      "${REAL_DATASET_ARGS[@]}" \
      --k-min "$K_MIN" \
      --k-max "$K_MAX" \
      --seeds "$seed" \
      --random-state "$RANDOM_STATE" \
      --resume \
      "${BEANS_ARGS[@]}" \
      "$@" || log_error "Methods failed for real datasets seed$seed"

    ((++COMPLETED_CONFIGS))
  done
fi

################################################################################
# Summary Report
################################################################################

log_separator
log_info "Benchmark Run Complete"
log_separator
log_info "Total configurations: $TOTAL_CONFIGS"
log_info "Completed: $COMPLETED_CONFIGS"
log_info "Skipped (existing): $SKIPPED_CONFIGS"
log_info "Results directory: $OUTPUT_DIR"
log_info "Summary files:"
log_info "  - $OUTPUT_DIR/quality_comparison_by_k.csv"
log_info "  - $OUTPUT_DIR/quality_comparison_summary.csv"
log_info "  - $OUTPUT_DIR/quality_comparison_manifest.json"
log_separator

if [[ $COMPLETED_CONFIGS -eq 0 ]]; then
  log_error "No configurations were completed (all may have been skipped)"
  exit 1
fi

log_info "✓ Benchmark suite finished successfully"
exit 0
