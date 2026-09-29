# Missing-Edges Connectivity Experiment

This experiment evaluates whether sparse supports built by ScoreSG remain
connected after anti-hub missing-edge reinforcement. It is intended to produce
evidence for the missing-edge/connectivity discussion in the paper.

For the `paper` preset, every synthetic distribution is evaluated in two
structures: an `iid` version and a `separated_clusters` version. The separated
version keeps the same distribution, sample size, dimension, and seed, but adds
well-separated cluster centers to make sparse inter-cluster regions explicit.

The script writes:

- `benchmarking/missing_edges/results/missing_edges_connectivity_by_k.csv`: one row for each dataset, variant, and
  `k` value from `k_max` down to `k_min`.
- `benchmarking/missing_edges/results/missing_edges_connectivity_summary.csv`: one row per dataset and support
  variant.
- `benchmarking/missing_edges/results/missing_edges_connectivity_failures.csv`: only configurations that were not
  connected for all evaluated `k` values.
- `benchmarking/missing_edges/results/missing_edges_connectivity_by_structure.csv`: connected counts aggregated by
  dataset structure and support variant.
- `benchmarking/unties/results/missing_edges_tie_diagnostics.csv`: one row per dataset with degree-only and
  score-based anti-hub tie diagnostics.
- `benchmarking/unties/results/missing_edges_tie_by_dimension.csv`: synthetic tie diagnostics aggregated by
  feature dimension.
- `benchmarking/missing_edges/results/*_connectivity_rate.png`: optional bar plots of the connected fraction.
- `benchmarking/unties/results/score_sg_tie_break_by_dimension.png`: optional line plot comparing degree-only
  tie selection with score-based tie-breaking as dimensionality changes.

Three support variants are reported:

- `approx_knn_only`: approximate kNN support before anti-hub reinforcement.
- `random_sqrt_clique`: approximate kNN support after adding a clique over
  `floor(sqrt(N))` uniformly sampled vertices.
- `score_sg_antihub`: approximate kNN support after the ScoreSG anti-hub clique.

Run a small smoke test:

```bash
.venv310/bin/python benchmarking/missing_edges/missing_edges_connectivity.py \
  --output-dir benchmarking/missing_edges/results_smoke \
  --tie-output-dir benchmarking/unties/results_smoke \
  --distributions gaussian,poisson \
  --sample-sizes 1000 \
  --dimensions 2 \
  --k-max 50 \
  --k-min 2 \
  --no-real
```

Run the paper-oriented grid one configuration at a time:

```bash
.venv310/bin/python benchmarking/missing_edges/run_large_grid_one_by_one.py \
  --output-dir benchmarking/missing_edges/results_large \
  --tie-output-dir benchmarking/unties/results_large \
  --k-max 50 \
  --k-min 2 \
  --threads 1 \
  --resume
```

This is the recommended command for the full grid because it evaluates each
distribution/sample-size/dimension combination in a separate process, limits
threaded numerical libraries, writes a `done.json` marker for completed batches,
and merges partial CSV files after every batch. If the run is interrupted, use
the same command with `--resume` to continue from the next unfinished batch.

Run the same grid in one process only on machines with enough memory:

```bash
.venv310/bin/python benchmarking/missing_edges/missing_edges_connectivity.py \
  --preset paper \
  --output-dir benchmarking/missing_edges/results \
  --tie-output-dir benchmarking/unties/results \
  --k-max 50 \
  --k-min 2
```

The `paper` preset evaluates synthetic sample sizes `1k`, `5k`, `10k`, `25k`,
`50k`, `100k`, `200k`, `500k`, and `1M`, with dimensions `2`, `10`, `32`,
`64`, and `128`.

Use `--no-separated-synthetic` to reproduce only the original iid synthetic
grid. Use `--separated-cluster-separation` and `--separated-cluster-scale` to
control the sparse-cluster geometry.

Run with a local real CSV dataset:

```bash
.venv310/bin/python benchmarking/missing_edges/missing_edges_connectivity.py \
  --preset paper \
  --real-csv beans=/path/to/Dry_Bean_Dataset.csv \
  --csv-drop-column beans:Class \
  --output-dir benchmarking/missing_edges/results \
  --tie-output-dir benchmarking/unties/results
```

Real datasets are normalized with median imputation and `StandardScaler` by
default. Built-in real datasets are `iris`, `wine`, `breast_cancer`, and
`digits`. Use `--no-normalize-real` only when the feature scales are already
comparable. Use `--no-plots` to skip matplotlib output and produce CSV files
only.
