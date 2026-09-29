# Quality Benchmark: ARI and HAI

This benchmark compares clustering quality against exact HDBSCAN with
`algorithm="generic"`, `approx_min_span_tree=False`, and
`match_reference_implementation=True`.

## Compared Methods

- `HDBSCAN`: exact reference, using the generic HDBSCAN implementation.
- `Optimized HDBSCAN`: HDBSCAN with `algorithm="best"`.
- `ScoreSG`: approximate ScoreSG with anti-hub reinforcement.
- `ScoreSG Random`: ScoreSG-style approximate support in which the
  `sqrt(N)` reinforced vertices are selected uniformly at random.

## Metrics

- `ari_vs_hdbscan_generic`: ARI between each method and exact HDBSCAN labels
  for the same dataset and `k`.
- `ari_vs_true`: ARI against generator or real-dataset labels when labels are
  available. This is empty for i.i.d. synthetic distribution diagnostics.
- `hai`: exact Hierarchy Agreement Index used by this script, computed from
  the single-linkage hierarchy. For every pair of points, the script finds the
  smallest single-linkage cluster in which the pair appears together in each
  hierarchy, normalizes that cluster size by `N`, computes the absolute
  difference between the two hierarchy levels, and averages this difference
  over all ordered pairs. The final score is `1 - mean_difference`.
- `mst_edge_jaccard`: auxiliary MST edge-set diagnostic. This is not HAI.

The `hai` definition is intentionally recorded in the output column
`hai_definition` as
`exact_single_linkage_pairwise_smallest_common_cluster_size_agreement`. The
implementation is exact and does not sample point pairs. It uses a
numba-accelerated all-pairs comparison over LCA queries in the two
single-linkage hierarchies when numba is available, with a pure-Python fallback
for runs with at most `50,000` samples and for tests.

## Outputs

The runner writes one folder per dataset configuration under:

```text
benchmarking/quality/results/_runs/
```

Merged outputs are updated after each completed configuration:

```text
benchmarking/quality/results/quality_comparison_by_k.csv
benchmarking/quality/results/quality_comparison_summary.csv
benchmarking/quality/results/quality_comparison_manifest.json
```

Use `--resume` to continue after interruption.

## Server Run

```bash
THREADS=1 PYTHON_BIN=.venv310/bin/python benchmarking/quality/run_quality_comparison.sh
```

Extra arguments can be appended to the shell script call, for example:

```bash
THREADS=4 benchmarking/quality/run_quality_comparison.sh --limit 10
```
