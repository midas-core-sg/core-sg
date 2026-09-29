#!/usr/bin/env bash

set -euo pipefail

samples=(1000 5000 10000 20000 30000 40000 50000 60000)

for n in "${samples[@]}"; do
  echo "Running benchmark with n-samples=${n}"
  python3 benchmarking/run_time/scripts/script.py --label core_sg_cython --n-samples "${n}"
done
