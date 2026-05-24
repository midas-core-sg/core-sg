# Core-SG and ScoreSG for Multi-k Hierarchical Extraction: Context, Scalability, and Comparative Performance Analysis

## Abstract

This report evaluates Core-SG and ScoreSG as reusable-structure approaches for repeated multi-`k` density-based hierarchical extraction. The analysis compares their cumulative runtime against HDBSCAN variants over synthetic datasets with fixed dimensionality, cluster structure, and random seed. For the common range from `5,000` to `50,000` samples, ScoreSG with optimized extraction is the fastest method across all tested dataset sizes, followed by optimized exact CoreSG. Empirical power-law fits show that ScoreSG grows approximately as `O(N^1.25)`, while HDBSCAN `best` grows around `O(N^1.60)` and HDBSCAN `generic` around `O(N^2.11)`. The results indicate that reusable graph construction substantially reduces the cumulative cost of multi-`k` workflows, especially as the number of samples increases. The report also discusses methodological limitations, practical deployment contexts, and potential market applications for exploratory clustering systems.

**Keywords:** Core-SG; ScoreSG; HDBSCAN; density-based clustering; scalability; minimum spanning tree; multi-`k` analysis.

## 1. Introduction

Density-based clustering methods are widely used when the structure of the data cannot be adequately represented by simple geometric assumptions, such as spherical clusters or linear separability. HDBSCAN has become a robust and widely adopted method in this setting because it produces density hierarchies, handles noise, and exposes interpretable artifacts such as condensed trees, single-linkage trees, and cluster persistence scores.

However, a practical limitation appears in exploratory workflows. Analysts often need to evaluate multiple minimum-neighborhood values, or multiple levels of local-density smoothing, before selecting a useful granularity. If the method is executed independently for each value of `k`, cumulative runtime grows quickly. This pattern is especially costly in applications where multi-scale exploration is part of the analytical process, including customer segmentation, anomaly investigation, embedding organization, scientific data analysis, and pattern mining in large transactional datasets.

Core-SG was proposed for this context. Its central premise is to build a reusable support structure for a maximum neighborhood value (`k_max`) and then extract hierarchical artifacts for smaller values of `k` without rebuilding the entire pipeline. Therefore, the relevant performance target is not only single-run execution time, but the cumulative cost of traversing a sequence of `k` values.

This report extends the previous benchmark reports by presenting the library in a broader academic style. In addition to performance results, it discusses technical context, potential applications, methodological limitations, and the observed differences between CoreSG, ScoreSG, and HDBSCAN variants.

## 2. Library Context

Core-SG implements a workflow compatible with the user experience of HDBSCAN, exposing artifacts such as labels, probabilities, cluster persistence, condensed trees, single-linkage trees, and minimum spanning trees. The main difference lies in the computational strategy: rather than rebuilding the structure for each `k`, the library aims to amortize an initial construction through reusable extraction.

Two algorithmic paths are relevant in the current library:

- **Exact CoreSG:** builds support structures with a stronger commitment to the exact formulation, making it appropriate when construction cost is acceptable and fidelity to the original graph is a priority.
- **ScoreSG:** uses an optimized approximate construction path, including approximate neighbor discovery and anti-hub reinforcement, to reduce construction and extraction costs at larger scales.

HDBSCAN is the natural baseline. In the experiments, it is evaluated through two main families:

- **HDBSCAN `best`:** allows HDBSCAN to select the most appropriate internal algorithmic path.
- **HDBSCAN `generic`:** forces execution through a generic path, which tends to be more expensive and less scalable.

The experiments also distinguish between reference-like and optimized configurations. In this report, `matchRef=true` is interpreted as closer to a reference implementation, whereas `matchRef=false` represents the optimized path observed in the benchmarks.

## 3. Experimental Methodology

The analyzed data come from CSV files in [`benchmarking/results`](./results). Two groups of experiments are considered:

- `results_*.csv`: exact CoreSG and HDBSCAN for `N` between `1,000` and `50,000`;
- `core_sg_cython_*.csv`: ScoreSG for `N` between `5,000` and `200,000`.

To ensure comparability across all methods, the joint analysis focuses on the common interval from `5,000` to `50,000` samples. The extended analysis up to `200,000` samples is presented separately and applies only to ScoreSG, since equivalent measurements for HDBSCAN and exact CoreSG are not available in that range.

All experiments use:

- `n_features = 20`;
- `centers = 10`;
- `seed = 42`;
- `k` values from `50` down to `2`, totaling `49` extractions;
- `30` repetitions per configuration.

For CoreSG and ScoreSG, cumulative runtime was reconstructed as:

```text
T_cumulative(N) = T_build(N) + sum_k T_extraction(N, k)
```

This formulation is necessary because construction is performed only once. Directly summing the total time recorded on each CSV row would count the construction phase multiple times and distort the interpretation of reusable methods. For HDBSCAN, cumulative runtime is the sum of the execution times for each value of `k`, since the method is executed independently for each configuration.

To characterize scalability, an empirical power-law model was fitted:

```text
T(N) = a * N^b
```

The exponent `b` is used as a descriptive indicator of observed growth over the measured interval. It should not be interpreted as a formal proof of asymptotic complexity, but as an experimental approximation useful for comparing growth regimes.

## 4. Results

### 4.1 Cumulative Runtime in the Common Interval

The following table reports cumulative runtime, in seconds, required to complete all `49` values of `k` in the common interval across methods.

| Method | 5,000 | 10,000 | 20,000 | 30,000 | 40,000 | 50,000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ScoreSG, optimized extraction | 12.69 | 28.44 | 65.10 | 111.28 | 166.37 | 224.36 |
| Exact CoreSG, optimized extraction | 14.29 | 36.11 | 100.21 | 191.72 | 381.33 | 525.45 |
| ScoreSG, reference extraction | 16.27 | 39.81 | 108.45 | 217.78 | 417.72 | 646.98 |
| Exact CoreSG, reference extraction | 17.36 | 47.11 | 141.25 | 296.81 | 597.55 | 971.22 |
| HDBSCAN `best`, optimized | 35.43 | 91.22 | 331.03 | 542.19 | 980.50 | 1,342.44 |
| HDBSCAN `best`, reference | 38.85 | 102.44 | 372.93 | 665.30 | 1,216.81 | 1,718.93 |
| HDBSCAN `generic`, optimized | 76.68 | 299.81 | 1,429.84 | 2,894.76 | 6,747.33 | 9,272.92 |
| HDBSCAN `generic`, reference | 80.07 | 311.31 | 1,323.41 | 3,015.08 | 7,181.05 | 9,580.93 |

The ranking is stable across the entire common interval: ScoreSG with optimized extraction is the fastest method for every value of `N`; it is followed by optimized exact CoreSG, then by the reference extraction variants of ScoreSG and CoreSG, and finally by HDBSCAN. This ordering suggests that structural reuse is more important for cumulative cost than smaller internal differences among HDBSCAN configurations.

At `N=50,000`, optimized ScoreSG completes the multi-`k` workflow in **224.36 s**. At the same point, optimized exact CoreSG takes **525.45 s**, optimized HDBSCAN `best` takes **1,342.44 s**, and optimized HDBSCAN `generic` takes **9,272.92 s**. Thus, at the largest common point, optimized ScoreSG is approximately **5.98x** faster than optimized HDBSCAN `best` and **41.33x** faster than optimized HDBSCAN `generic`.

### 4.2 Empirical Scalability

| Method | Empirical exponent b | R² |
| --- | ---: | ---: |
| ScoreSG, optimized extraction | 1.247 | 0.9988 |
| Exact CoreSG, optimized extraction | 1.575 | 0.9932 |
| HDBSCAN `best`, optimized | 1.601 | 0.9971 |
| HDBSCAN `best`, reference | 1.668 | 0.9975 |
| ScoreSG, reference extraction | 1.592 | 0.9902 |
| Exact CoreSG, reference extraction | 1.736 | 0.9913 |
| HDBSCAN `generic`, optimized | 2.113 | 0.9979 |
| HDBSCAN `generic`, reference | 2.114 | 0.9973 |

The exponents indicate three growth regimes. The first is represented by optimized ScoreSG, whose exponent near `1.25` indicates subquadratic and moderately superlinear growth. The second includes optimized exact CoreSG, HDBSCAN `best`, and reference ScoreSG, with exponents between `1.57` and `1.67`. The third is represented by HDBSCAN `generic`, with an exponent close to `2.11`, indicating nearly quadratic or worse behavior in the observed interval.

This distinction is methodologically important. In multi-`k` workloads, a method with acceptable single-run cost can become expensive when repeated dozens of times. HDBSCAN `best` remains much more competitive than HDBSCAN `generic`, but it still pays the cost of recomputation for each value of `k`. CoreSG and ScoreSG, in contrast, transform the sequence of executions into one initial construction followed by repeated extraction, changing the cumulative runtime regime.

### 4.3 Internal Differences in HDBSCAN

The HDBSCAN variants exhibit important differences. The `best` mode is systematically superior to the `generic` mode. Considering optimized configurations, the ratio between HDBSCAN `generic` and HDBSCAN `best` increases with `N`:

| N | HDBSCAN `generic` / HDBSCAN `best` |
| --- | ---: |
| 5,000 | 2.16x |
| 10,000 | 3.29x |
| 20,000 | 4.32x |
| 30,000 | 5.34x |
| 40,000 | 6.88x |
| 50,000 | 6.91x |

This pattern shows that the internal algorithm selected by HDBSCAN is not an operational detail. For larger datasets, the `generic` strategy strongly amplifies cumulative runtime. The `best` configuration substantially reduces this penalty by selecting more efficient paths, but it still remains slower than reusable-structure methods when the experiment requires multiple `k` values.

The results also show that `matchRef=true` tends to be slower than `matchRef=false` in HDBSCAN `best`. In HDBSCAN `generic`, there is a local inversion at `N=20,000`, where the reference variant is faster than the optimized variant. This exception does not alter the overall trend, since at larger values of `N` the optimized variant is again slightly faster and both remain much slower than the other methods.

### 4.4 Relative Gains against HDBSCAN `best`

Comparing the two main library methods against optimized HDBSCAN `best`, ScoreSG shows the larger advantage.

| N | Optimized exact CoreSG vs HDBSCAN `best` | Optimized ScoreSG vs HDBSCAN `best` |
| --- | ---: | ---: |
| 5,000 | 2.48x | 2.79x |
| 10,000 | 2.53x | 3.21x |
| 20,000 | 3.30x | 5.08x |
| 30,000 | 2.83x | 4.87x |
| 40,000 | 2.57x | 5.89x |
| 50,000 | 2.55x | 5.98x |

Exact CoreSG already provides substantial gains over HDBSCAN `best`, mainly because it avoids full recomputation. ScoreSG increases these gains, indicating that the approximate optimized strategy reduces both construction cost and recurring extraction cost. The relative advantage of ScoreSG becomes especially clear from `N=20,000` onward.

### 4.5 Extended Scale of ScoreSG

The new `core_sg_cython` experiments extend to `N=200,000`, but only for ScoreSG. The optimized configuration maintains regular growth:

| N | Build (s) | Cumulative extraction (s) | Cumulative total (s) |
| --- | ---: | ---: | ---: |
| 5,000 | 0.87 | 11.82 | 12.69 |
| 10,000 | 1.77 | 26.67 | 28.44 |
| 20,000 | 3.77 | 61.33 | 65.10 |
| 50,000 | 12.16 | 212.20 | 224.36 |
| 100,000 | 24.38 | 495.91 | 520.29 |
| 200,000 | 54.92 | 1,192.01 | 1,246.93 |

From `5,000` to `200,000` samples, optimized ScoreSG has an empirical exponent of `b = 1.257`, with `R² = 0.9994`. The reference extraction configuration, in contrast, has `b = 1.703`, with `R² = 0.9951`. Therefore, the extraction routine is decisive: the optimized configuration preserves moderate growth, while the reference version degrades rapidly as `N` increases.

Cost decomposition is also relevant. In the optimized configuration, construction accounts for only `6.8%` of total runtime at `N=5,000` and `4.4%` at `N=200,000`. This indicates that, for a sequence of `49` extractions, the main cost driver is cumulative extraction rather than initial construction. Consequently, future improvements in extraction are likely to produce larger practical gains than isolated optimizations in the build phase.

## 5. Discussion

The results support the central hypothesis of the library: in multi-`k` tasks, reusing internal structures changes the performance profile relative to independent execution. HDBSCAN remains a robust and broadly applicable method, especially for single analyses, but its cumulative cost grows quickly when the procedure is repeated for dozens of `k` values.

HDBSCAN `best` is a strong baseline and is substantially superior to HDBSCAN `generic`. Even so, this optimized baseline is outperformed by CoreSG and ScoreSG across all common values of `N`. This suggests that the main advantage is not merely a matter of implementation-level micro-optimization, but a change in computational model: build once and reuse.

The comparison between exact CoreSG and ScoreSG is also informative. Exact CoreSG provides a more conservative path, but its growth is steeper. ScoreSG introduces approximations and optimizations that improve empirical scalability. This choice involves a methodological trade-off: better runtime and larger-scale reach may depend more strongly on approximate neighbor quality, support-graph connectivity, and stability across different data distributions.

In terms of behavior with increasing `N`, the pattern is clear. Methods that recompute complete structures for each value of `k` accumulate cost rapidly. Methods that build reusable support reduce total growth, and optimized ScoreSG exhibits the most favorable regime among the evaluated alternatives.

## 6. Limitations

Despite the consistency of the results, several limitations should be made explicit.

First, the experiments use synthetic data with `20` features, `10` centers, and a fixed seed. This control is useful for comparison, but it does not capture the full variability of real data, such as extreme high dimensionality, highly heterogeneous densities, intertwined clusters, severe noise, or non-stationary distributions.

Second, the main comparison across all methods is limited to `N <= 50,000`. For `N=60,000` through `N=200,000`, only ScoreSG measurements are available. Therefore, relative behavior against HDBSCAN and exact CoreSG in this extended range should be inferred cautiously, even though the exponents observed up to `50,000` suggest that the gap would likely widen.

Third, this report measures runtime but does not evaluate memory usage, clustering quality, label stability, hierarchy preservation, or agreement between methods. This distinction is fundamental: an approximate method can be faster, but its appropriateness also depends on the fidelity of its results to the analytical objective.

Fourth, the traditional exact CoreSG path has a practical sample-size limitation because it relies on dense pairwise distance information. This limitation should not be interpreted as a limitation of the whole library design: ScoreSG is precisely the scalable approximate path intended to avoid the dense all-pairs construction and extend the repeated multi-`k` workflow to larger `N`.

Fifth, the results depend on the execution environment, implementation details, approximate-neighbor libraries, and hardware. Without a complete description of CPU, memory, operating system, and parallelism policy, absolute runtimes should be interpreted as local experimental evidence; relative patterns and growth exponents are more informative than exact seconds.

Finally, the analysis considers a workflow with `49` values of `k`. In real usage, the advantage of CoreSG and ScoreSG depends on the number of extractions. For a single isolated execution, the construction cost may not be amortized; for extensive exploration, reuse tends to become increasingly advantageous.

## 7. Potential Market Applications

The library is particularly promising in scenarios where clustering is exploratory, recurrent, and sensitive to granularity. Examples include:

- **Customer segmentation:** CRM and growth teams can evaluate different granularity levels without rerunning the entire pipeline for each hypothesis.
- **Anomaly detection and investigation:** fraud, security, and monitoring workflows can explore local groupings at multiple scales to distinguish rare patterns from operational noise.
- **Embedding organization:** semantic search, recommendation, and document analysis systems can inspect clusters in vector spaces.
- **Bioinformatics and scientific data analysis:** population, experiment, or measurement analysis can benefit from hierarchical inspection at different density levels.
- **Observability and telemetry:** large volumes of events, logs, and metrics can be grouped at different granularities to support diagnosis and pattern discovery.

The main market value is not replacing HDBSCAN in every setting, but reducing the cost of exploratory cycles where many `k` values must be tested on the same dataset. In production environments, this can reduce experimentation time, computational cost, and analytical latency.

## 8. Conclusion

The analysis shows that CoreSG and ScoreSG address a practical limitation of HDBSCAN-like workflows: the cost of recomputing structures for multiple values of `k`. In the common comparison interval, ScoreSG with optimized extraction is the most efficient method, followed by optimized exact CoreSG. HDBSCAN `best` remains an important baseline and is much faster than HDBSCAN `generic`, but both are penalized by repeated independent execution.

The most important result concerns scalability. While HDBSCAN `generic` has empirical growth near `O(N^2.11)` and HDBSCAN `best` grows around `O(N^1.60)`, optimized ScoreSG grows approximately as `O(N^1.25)` in the common interval and maintains similar behavior up to `N=200,000`. This makes the library especially suitable for multi-`k` analysis over medium and large datasets.

As future work, the results suggest two priorities: extending evaluation to real datasets and incorporating quality, memory, and stability metrics in addition to runtime. Even so, under the criterion of cumulative performance, the current evidence is consistent: reusable structural computation provides a clear advantage over independent recomputation, and ScoreSG is the most scalable path among the evaluated methods.
