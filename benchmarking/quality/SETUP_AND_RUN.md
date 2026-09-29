# CoreSG Quality Benchmark - Setup and Execution Guide

## Overview

This guide explains how to set up and run the CoreSG quality benchmarks (ARI and HAI metrics) on a new machine. These experiments measure clustering quality across various configurations, comparing HDBSCAN, Optimized HDBSCAN, ScoreSG, and ScoreSG Random methods.

**Key Characteristics:**
- Metric: ARI (Adjusted Rand Index) vs. HDBSCAN reference, and HAI (Hierarchy Agreement Index)
- Sample sizes: 5K, 10K, 20K, 30K, 40K, 50K (capped at 50K for exact HDBSCAN)
- Dimensions: 2, 10, 20, 32, 64, 128
- Distributions: 9 synthetic distributions
- Total estimated runtime: **Several hours to days** (depending on machine specs and threads)

---

## Part 1: Prerequisites

### System Requirements

- **OS:** Linux, macOS, or Windows (with WSL)
- **Python:** 3.10 or 3.11 (3.12+ not recommended for now)
- **RAM:** Minimum 8GB (16GB+ recommended for parallel runs)
- **Disk Space:** ~2GB for source code and results

### Software Requirements

```bash
# Core dependencies (typically pre-installed)
pip install numpy scipy scikit-learn pandas numba numexpr

# Algorithm dependencies
pip install hdbscan scikit-learn

# Optional: For faster approximate KNN (used by ScoreSG)
pip install pynndescent

# Optional: For Cython builds (if not pre-built)
pip install cython

# Development (optional, for development work)
pip install pytest hypothesis
```

---

## Part 2: Repository Setup

### 2.1 Clone and Navigate

```bash
# Clone the CoreSG repository
git clone <repository-url> core-sg
cd core-sg
```

### 2.2 Python Environment Setup

#### Option A: Virtual Environment (Recommended)

```bash
# Create virtual environment with Python 3.10
python3.10 -m venv .venv310

# Activate
source .venv310/bin/activate  # Linux/macOS
# or: .venv310\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Option B: Conda Environment

```bash
conda create -n core-sg python=3.10
conda activate core-sg
```

### 2.3 Install CoreSG

```bash
# Install in development mode (builds Cython extensions)
pip install -e .

# Verify installation
python -c "from core_sg import CoreSG; print('CoreSG imported successfully')"
```

### 2.4 Verify Installation

```bash
# Check all required packages
python -c "
import numpy, scipy, sklearn, pandas, hdbscan, numba
from core_sg import CoreSG
from core_sg.edges import build_knng_vectors
from core_sg.score_sg import build_approximate_knn_graph
print('✓ All dependencies installed successfully')
"
```

---

## Part 3: Understanding Configuration Options

### Available Scripts

#### Script 1: `run_quality_comparison_to_50k.sh` (Recommended - Simpler)

**Best for:** Standard article configuration, quick setup

```bash
./benchmarking/quality/run_quality_comparison_to_50k.sh
```

**Configuration:**
- ✓ Fixed to article's exact specifications
- ✓ Default: All 9 distributions, all 6 dimensions, all 6 sample sizes
- ✓ 4 methods: HDBSCAN, Optimized HDBSCAN, ScoreSG, ScoreSG Random
- ✓ Easy to pass additional arguments

**Customization via environment variables:**
```bash
THREADS=4 ./benchmarking/quality/run_quality_comparison_to_50k.sh
SKIP_EXISTING=1 ./benchmarking/quality/run_quality_comparison_to_50k.sh  # Resume
"./benchmarking/quality/run_quality_comparison_to_50k.sh --limit 10    # Limit configurations
```

#### Script 2: `run_article_quality_to_50k.sh` (Advanced - Full Control)

**Best for:** Custom configurations, fine-grained control

```bash
./benchmarking/quality/run_article_quality_to_50k.sh
```

**Configuration via environment variables:**
```bash
# Sample sizes (comma-separated, max 50000)
export SAMPLES_CSV="5000,10000,20000,30000,40000,50000"

# Dimensions to test
export DIMENSIONS_CSV="2,10,20,32,64,128"

# Distributions (subset of: gaussian, poisson, chi_square, gamma, beta, von_mises, gumbel, logistic, gaussian_sparse)
export DISTRIBUTIONS_CSV="gaussian,gaussian_sparse"

# Clustering parameters
export K_MIN=2
export K_MAX=50
export RANDOM_STATE=42

# Thread control
export THREADS=4

# Output directory
export OUTPUT_DIR="benchmarking/quality/results"

# Resume mode (skip existing outputs)
export SKIP_EXISTING=1

# Which method families to run
export RUN_HDBSCAN_FAMILY=1      # runs: hdbscan_generic, optimized_hdbscan
export RUN_SCORESG_FAMILY=1      # runs: score_sg, score_sg_random

# Log verbosity
export LOG_LEVEL="INFO"  # or DEBUG, WARNING, ERROR

./benchmarking/quality/run_article_quality_to_50k.sh
```

---

## Part 4: Running the Experiments

### Quick Start (5-10 minutes to test)

```bash
# Activate environment
source .venv310/bin/activate

# Run with minimal set (test)
SAMPLES_CSV="5000" DIMENSIONS_CSV="2" DISTRIBUTIONS_CSV="gaussian" \
  PYTHON_BIN=".venv310/bin/python" ./benchmarking/quality/run_article_quality_to_50k.sh
```

### Standard Full Run (Article Configuration - Several Hours)

```bash
# Activate environment
source .venv310/bin/activate

# Run with all article configurations
THREADS=4 PYTHON_BIN=".venv310/bin/python" SKIP_EXISTING=1 \
  ./benchmarking/quality/run_quality_comparison_to_50k.sh
```

### Recommended Production Setup (Multi-Core)

```bash
# For a powerful server with many cores (e.g., 16 cores)
THREADS=8 \           # Leave some headroom for OS
  PYTHON_BIN=".venv310/bin/python" \
  SKIP_EXISTING=1 \
  LOG_LEVEL="INFO" \
  ./benchmarking/quality/run_quality_comparison_to_50k.sh
```

### Running in Background (Recommended for Remote Machines)

```bash
# Capture output to log file
nohup bash -c '
  source .venv310/bin/activate
  THREADS=4 ./benchmarking/quality/run_quality_comparison_to_50k.sh
' > benchmark.log 2>&1 &

# Monitor progress
tail -f benchmark.log

# Check if still running
ps aux | grep quality_comparison.py
```

### Resuming Interrupted Runs

```bash
# Set SKIP_EXISTING=1 to resume from last completed configuration
SKIP_EXISTING=1 THREADS=4 PYTHON_BIN=".venv310/bin/python" \
  ./benchmarking/quality/run_quality_comparison_to_50k.sh
```

---

## Part 5: Understanding Outputs

### Output Directory Structure

```
benchmarking/quality/results/
├── quality_comparison_by_k.csv           # Detailed results by k value
├── quality_comparison_summary.csv        # Summary statistics
├── quality_comparison_manifest.json      # Metadata and run info
└── _runs/                                # Individual configuration outputs
    ├── synthetic_gaussian_d2_n5000_seed42/
    ├── synthetic_gaussian_d2_n10000_seed42/
    ├── synthetic_poisson_d2_n5000_seed42/
    └── ...
```

### Main Output Files

#### `quality_comparison_by_k.csv`
Complete results for every k value tested. Columns include:
- `benchmark_group`: type of benchmark (synthetic, real)
- `dataset_name`: name of dataset/distribution
- `method`: clustering method used
- `i_mds`: minimum min_samples parameter (k_min)
- `j_mds`: k value (min_samples in HDBSCAN)
- `ari_vs_hdbscan_generic`: Adjusted Rand Index vs. exact HDBSCAN reference
- `hai`: Hierarchy Agreement Index
- `mst_edge_jaccard`: MST edge overlap metric
- Other metadata

#### `quality_comparison_summary.csv`
Aggregated statistics per dataset/method. Useful for:
- Comparing method performance across all k values
- Identifying which methods perform best on each distribution

#### `quality_comparison_manifest.json`
Metadata including:
- When runs completed
- Which configurations were successful/failed
- Run duration
- System information

### Viewing Results

```bash
# Quick overview
head -20 benchmarking/quality/results/quality_comparison_by_k.csv

# Count successful configurations
grep -c "success\|completed" benchmarking/quality/results/quality_comparison_manifest.json

# Filter results for specific distribution
grep "gaussian," benchmarking/quality/results/quality_comparison_by_k.csv | head

# Compare ARI across methods for gaussian_sparse at 50K samples
grep "gaussian_sparse.*50000" benchmarking/quality/results/quality_comparison_by_k.csv
```

---

## Part 6: Performance and Optimization

### Expected Runtime Estimates

These are **approximate** on a modern machine:

| Configuration | Single Core | 4 Cores | 8 Cores |
|---|---|---|---|
| Single distribution (9 dims × 6 sizes) | ~30 min | ~10 min | ~6 min |
| All 9 distributions | ~4.5 hrs | ~1.5 hrs | ~1 hr |
| All + Real datasets | ~6 hrs | ~2 hrs | ~1.5 hrs |

**Key Factors:**
- Exact HDBSCAN computation is O(N²) for quality metrics
- Larger sample sizes (50K) dominate runtime
- ScoreSG is typically 2-3× faster than HDBSCAN
- HAI computation is expensive; ARI is cheaper

### Optimization Tips

1. **Use multiple threads:**
   ```bash
   THREADS=4 ./benchmarking/quality/run_quality_comparison_to_50k.sh
   ```

2. **Run reduced set first (test/validation):**
   ```bash
   SAMPLES_CSV="50000" DIMENSIONS_CSV="20" DISTRIBUTIONS_CSV="gaussian" \
     ./benchmarking/quality/run_article_quality_to_50k.sh
   ```

3. **Split runs across machines:**
   - Machine 1: `DISTRIBUTIONS_CSV="gaussian,poisson,chi_square"`
   - Machine 2: `DISTRIBUTIONS_CSV="gamma,beta,von_mises"`
   - Machine 3: `DISTRIBUTIONS_CSV="gumbel,logistic,gaussian_sparse"`

4. **Use resume mode to avoid recomputation:**
   ```bash
   SKIP_EXISTING=1 THREADS=8 ./benchmarking/quality/run_quality_comparison_to_50k.sh
   ```

### System Monitoring During Run

```bash
# Monitor CPU, memory, and I/O
htop

# Monitor Python processes specifically
watch "ps aux | grep '[p]ython.*quality'"

# Monitor disk usage
watch "du -sh benchmarking/quality/results"

# Check remaining time from log
tail -20 benchmark.log | grep -E "Configuration|completed"
```

---

## Part 7: Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'core_sg'"

**Solution:**
```bash
cd /path/to/core-sg
pip install -e .
python -c "from core_sg import CoreSG; print('OK')"
```

#### "HDBSCAN methods failed"

**Possible causes:**
- Insufficient RAM for dataset size
- HDBSCAN installation issue
- Thread contention

**Solution:**
```bash
# Reduce sample sizes or dimensions
export SAMPLES_CSV="5000,10000"
export DIMENSIONS_CSV="2,10,20"

# Use single thread
THREADS=1 ./benchmarking/quality/run_quality_comparison_to_50k.sh
```

#### "n_samples exceeds 50,000 (exact HDBSCAN limit)"

**Solution:** The script enforces this limit. Edit SAMPLES_CSV:
```bash
export SAMPLES_CSV="5000,10000,20000,30000,40000,50000"  # ✓ Valid
export SAMPLES_CSV="5000,10000,100000"                   # ✗ Invalid
```

#### Out of Memory errors

**Solution:**
```bash
# Reduce problem size
export SAMPLES_CSV="5000,10000,20000"
export DIMENSIONS_CSV="2,10,20"
export DISTRIBUTIONS_CSV="gaussian"

# Use single thread
THREADS=1 ./benchmarking/quality/run_quality_comparison_to_50k.sh
```

#### Results appear incomplete

**Solution:** Resume with SKIP_EXISTING:
```bash
SKIP_EXISTING=1 THREADS=4 ./benchmarking/quality/run_quality_comparison_to_50k.sh
```

---

## Part 8: Validation and Result Checking

### Verify Successful Run

```bash
# All three output files should exist
ls -lh benchmarking/quality/results/*.csv
ls -l benchmarking/quality/results/*.json

# Check for errors (should be empty or minimal)
tail -50 benchmark.log | grep -i error

# Count completed configurations
wc -l benchmarking/quality/results/quality_comparison_by_k.csv
```

### Compare Against Reference

If you have reference results from another run:
```bash
# Diff summary results
diff reference_summary.csv benchmarking/quality/results/quality_comparison_summary.csv

# Check if scores are in expected range (ARI: -1 to 1, HAI: -1 to 1)
awk -F',' '$6 ~ /ari_vs_hdbscan_generic/ {print $0}' quality_comparison_by_k.csv | head
```

---

## Part 9: Advanced Usage

### Custom Configuration File

Create a file `my_config.sh`:

```bash
#!/bin/bash
# Custom quality benchmark configuration

export SAMPLES_CSV="10000,50000"
export DIMENSIONS_CSV="20,64"
export DISTRIBUTIONS_CSV="gaussian,gaussian_sparse"
export K_MAX=50
export THREADS=8
export SKIP_EXISTING=1
export LOG_LEVEL="INFO"

./benchmarking/quality/run_quality_comparison_to_50k.sh "$@"
```

Usage:
```bash
chmod +x my_config.sh
./my_config.sh
```

### Splitting Runs for Parallel Execution

**Machine 1 (gaussian family):**
```bash
DISTRIBUTIONS_CSV="gaussian,poisson,chi_square" \
  THREADS=4 ./benchmarking/quality/run_quality_comparison_to_50k.sh
```

**Machine 2 (continuous family):**
```bash
DISTRIBUTIONS_CSV="gamma,beta,gumbel,logistic" \
  THREADS=4 ./benchmarking/quality/run_quality_comparison_to_50k.sh
```

**Machine 3 (special):**
```bash
DISTRIBUTIONS_CSV="von_mises,gaussian_sparse" \
  THREADS=4 ./benchmarking/quality/run_quality_comparison_to_50k.sh
```

Then merge results:
```bash
# Results in quality_comparison_by_k.csv are automatically merged
# Just copy result directories or CSV files as needed
```

---

## Part 10: Summary Checklist

- [ ] Python 3.10+ installed
- [ ] Repository cloned and navigated
- [ ] Virtual environment created and activated
- [ ] CoreSG installed: `pip install -e .`
- [ ] All dependencies verified
- [ ] Test run successful: `SAMPLES_CSV="5000" DIMENSIONS_CSV="2" ... run_...sh`
- [ ] Full run initiated with proper thread settings
- [ ] Background job configured (if needed): `nohup ... benchmark.log &`
- [ ] Results directory monitored: `tail -f benchmark.log`
- [ ] Output files checked: `ls benchmarking/quality/results/*.csv`

---

## Contact and Support

For issues with:
- **CoreSG installation:** See CONTRIBUTING.md
- **HDBSCAN:** https://hdbscan.readthedocs.io
- **Benchmark configuration:** Check README.md in `benchmarking/quality/`

---

## References

- CoreSG Repository: This repository
- Quality Benchmark Documentation: `benchmarking/quality/README.md`
- Runtime Benchmark Reference: `benchmarking/run_time/scripts/run_article_runtime_to_50k.sh`
- Article: Check `paper/main.tex` for experimental details
