#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Runtime grid used in the paper's common comparison, capped at 50k samples.
SAMPLES_CSV="${SAMPLES_CSV:-5000,10000,20000,30000,40000,50000}"
FEATURES_CSV="${FEATURES_CSV:-20}"
CENTERS_CSV="${CENTERS_CSV:-10}"
SEEDS_CSV="${SEEDS_CSV:-42}"

K_MAX="${K_MAX:-50}"
REPETITIONS="${REPETITIONS:-30}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarking/run_time/results}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Toggle method families without editing the script:
#   RUN_CORESG_HDBSCAN=0 disables CoreSG/HDBSCAN runs.
#   RUN_SCORESG=0 disables ScoreSG runs.
RUN_CORESG_HDBSCAN="${RUN_CORESG_HDBSCAN:-1}"
RUN_SCORESG="${RUN_SCORESG:-1}"

# Set SKIP_EXISTING=1 to resume a partially completed grid.
SKIP_EXISTING="${SKIP_EXISTING:-0}"

IFS=',' read -r -a SAMPLES <<< "$SAMPLES_CSV"
IFS=',' read -r -a FEATURES_LIST <<< "$FEATURES_CSV"
IFS=',' read -r -a CENTERS_LIST <<< "$CENTERS_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

label_for_config() {
  local base_label="$1"
  local n_features="$2"
  local centers="$3"
  local seed="$4"

  if [[ "${#FEATURES_LIST[@]}" -eq 1 && "${#CENTERS_LIST[@]}" -eq 1 && "${#SEEDS[@]}" -eq 1 ]]; then
    printf '%s' "$base_label"
  else
    printf '%s_d%s_c%s_seed%s' "$base_label" "$n_features" "$centers" "$seed"
  fi
}

run_python_benchmark() {
  local script_path="$1"
  local label="$2"
  local n_samples="$3"
  local n_features="$4"
  local centers="$5"
  local seed="$6"
  local output_path="${OUTPUT_DIR}/${label}_${n_samples}.csv"

  if [[ "$SKIP_EXISTING" == "1" && -f "$output_path" ]]; then
    echo "Skipping existing result: $output_path"
    return
  fi

  echo "Running ${script_path} | n=${n_samples} | d=${n_features} | centers=${centers} | seed=${seed} | repetitions=${REPETITIONS}"
  "$PYTHON_BIN" "$script_path" \
    --label "$label" \
    --output-dir "$OUTPUT_DIR" \
    --n-samples "$n_samples" \
    --n-features "$n_features" \
    --centers "$centers" \
    --seed "$seed" \
    --k-max "$K_MAX" \
    --repetitions "$REPETITIONS" \
    --log-level "$LOG_LEVEL"
}

for seed in "${SEEDS[@]}"; do
  for n_features in "${FEATURES_LIST[@]}"; do
    for centers in "${CENTERS_LIST[@]}"; do
      core_label="$(label_for_config "results" "$n_features" "$centers" "$seed")"
      score_label="$(label_for_config "score_sg" "$n_features" "$centers" "$seed")"

      for n_samples in "${SAMPLES[@]}"; do
        if (( n_samples > 50000 )); then
          echo "Refusing n-samples=${n_samples}: this runner is capped at 50000."
          exit 1
        fi

        if [[ "$RUN_CORESG_HDBSCAN" == "1" ]]; then
          run_python_benchmark \
            "benchmarking/run_time/scripts/script.py" \
            "$core_label" \
            "$n_samples" \
            "$n_features" \
            "$centers" \
            "$seed"
        fi

        if [[ "$RUN_SCORESG" == "1" ]]; then
          run_python_benchmark \
            "benchmarking/run_time/scripts/score_sg_script.py" \
            "$score_label" \
            "$n_samples" \
            "$n_features" \
            "$centers" \
            "$seed"
        fi
      done
    done
  done
done
