# ScoreSG Performance Report

## Abstract

This report analyzes only the benchmark files identified by the `core_sg_cython` prefix, which correspond to the current ScoreSG implementation. The objective is to characterize how runtime scales as the number of samples (`N`) increases. The experiments cover synthetic datasets from `5,000` to `200,000` samples, with `20` features, `10` centers, seed `42`, `30` repetitions per configuration, and hierarchy extraction for all `k` values from `50` down to `2`.

The empirical evidence indicates that `ScoreSG_extractMatchRef_false` is the preferred configuration in this benchmark suite. Under this configuration, cumulative runtime increases from **12.69 s** at `N=5,000` to **1,246.93 s** at `N=200,000`. A simple log-log fit suggests an approximate growth rate of **O(N^1.26)** for cumulative runtime, with `R² = 0.9994`. In contrast, `ScoreSG_extractMatchRef_true` shows a steeper growth rate, close to **O(N^1.70)**, and becomes progressively less suitable for large datasets.

## 1. Context and Objective

ScoreSG is intended for scenarios in which a reusable support structure is built once and then used to generate multiple hierarchies or tree artifacts for different `k` values. Therefore, performance analysis must separate the initial construction cost from the recurring extraction cost.

The main question addressed in this report is: **how does the method scale as `N` increases, while dimensionality, number of centers, random seed, and the sequence of `k` values remain fixed?** The answer is obtained by decomposing total runtime into two components: construction time (`fit_seconds`) and the sum of extraction times (`extract_seconds`) across the `49` evaluated values of `k`.

## 2. Experimental Design

### 2.1 Data and Parameters

The analyzed files are located in [`benchmarking/results`](./results) and use the `core_sg_cython` prefix. The experimental suite contains:

- `n_samples` in `{5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000}`;
- `n_features = 20`;
- `centers = 10`;
- `seed = 42`;
- `k` values from `50` down to `2`, totaling `49` values;
- `30` repetitions per configuration.

### 2.2 Evaluated Configurations

The CSV files contain two configurations of the algorithm:

- `ScoreSG_extractMatchRef_false`;
- `ScoreSG_extractMatchRef_true`.

Although the file prefix is `core_sg_cython`, the `family` column records the method as `ScoreSG` and the `algorithm` column as `score_sg`. For consistency with the available implementation and benchmark naming, this report refers to the method simply as ScoreSG.

### 2.3 Cumulative Runtime Metric

The cumulative ScoreSG runtime was reconstructed as:

```text
T_cumulative(N) = T_fit(N) + sum_k T_extract(N, k)
```

This formulation avoids counting the construction phase repeatedly. The `fit_seconds` field represents a single construction step, whereas `extract_seconds` represents the extraction cost for each `k`. For cumulative standard deviations, variance propagation was used:

```text
std_cumulative = sqrt(std_fit^2 + sum_k std_extract(k)^2)
```

This estimate assumes approximate independence between aggregated measurements and is the most defensible estimate supported by the available CSV outputs.

## 3. Aggregated Results

### 3.1 Configuration `extractMatchRef=false`

This is the fastest configuration for every tested value of `N`.

| N | Build mean ± std (s) | Cumulative extraction ± std (s) | Cumulative total ± std (s) | Build share | Mean extraction per k (s) |
| --- | --- | --- | --- | --- | --- |
| 5,000 | 0.87 ± 0.04 | 11.82 ± 0.02 | 12.69 ± 0.04 | 6.8% | 0.24 |
| 10,000 | 1.77 ± 0.04 | 26.67 ± 0.02 | 28.44 ± 0.04 | 6.2% | 0.54 |
| 20,000 | 3.77 ± 0.04 | 61.33 ± 0.04 | 65.10 ± 0.06 | 5.8% | 1.25 |
| 30,000 | 6.21 ± 0.07 | 105.07 ± 0.08 | 111.28 ± 0.11 | 5.6% | 2.14 |
| 40,000 | 9.27 ± 0.23 | 157.09 ± 0.42 | 166.37 ± 0.48 | 5.6% | 3.21 |
| 50,000 | 12.16 ± 0.18 | 212.20 ± 0.69 | 224.36 ± 0.72 | 5.4% | 4.33 |
| 60,000 | 13.42 ± 0.11 | 257.56 ± 0.21 | 270.98 ± 0.23 | 5.0% | 5.26 |
| 70,000 | 16.14 ± 0.13 | 316.63 ± 0.17 | 332.77 ± 0.21 | 4.9% | 6.46 |
| 80,000 | 21.80 ± 4.09 | 394.24 ± 1.86 | 416.04 ± 4.49 | 5.2% | 8.05 |
| 90,000 | 21.64 ± 0.11 | 435.25 ± 0.42 | 456.89 ± 0.43 | 4.7% | 8.88 |
| 100,000 | 24.38 ± 0.13 | 495.91 ± 0.23 | 520.29 ± 0.26 | 4.7% | 10.12 |
| 200,000 | 54.92 ± 0.21 | 1,192.01 ± 0.68 | 1,246.93 ± 0.71 | 4.4% | 24.33 |

The cumulative runtime is monotonic and shows that most of the total cost is spent in the extractions summed across the `49` values of `k`. The relative contribution of the initial build decreases slightly as `N` grows, from `6.8%` at `N=5,000` to `4.4%` at `N=200,000`. This suggests that, under this implementation and protocol, recurring extraction is the main determinant of total cost when many `k` values are evaluated.

### 3.2 Configuration `extractMatchRef=true`

| N | Build mean ± std (s) | Cumulative extraction ± std (s) | Cumulative total ± std (s) | Build share | Mean extraction per k (s) |
| --- | --- | --- | --- | --- | --- |
| 5,000 | 0.95 ± 0.03 | 15.32 ± 0.04 | 16.27 ± 0.05 | 5.8% | 0.31 |
| 10,000 | 2.01 ± 0.05 | 37.80 ± 0.04 | 39.81 ± 0.06 | 5.0% | 0.77 |
| 20,000 | 4.69 ± 0.08 | 103.76 ± 0.64 | 108.45 ± 0.65 | 4.3% | 2.12 |
| 30,000 | 8.22 ± 0.08 | 209.56 ± 0.27 | 217.78 ± 0.29 | 3.8% | 4.28 |
| 40,000 | 13.79 ± 0.26 | 403.93 ± 1.28 | 417.72 ± 1.31 | 3.3% | 8.24 |
| 50,000 | 19.93 ± 0.32 | 627.05 ± 2.44 | 646.98 ± 2.46 | 3.1% | 12.80 |
| 60,000 | 25.49 ± 0.79 | 841.07 ± 4.58 | 866.56 ± 4.64 | 2.9% | 17.17 |
| 70,000 | 32.65 ± 0.98 | 1,107.91 ± 5.96 | 1,140.56 ± 6.04 | 2.9% | 22.61 |
| 80,000 | 42.13 ± 1.31 | 1,445.43 ± 5.99 | 1,487.55 ± 6.13 | 2.8% | 29.50 |
| 90,000 | 47.83 ± 1.25 | 1,722.01 ± 9.49 | 1,769.84 ± 9.57 | 2.7% | 35.14 |
| 100,000 | 57.70 ± 1.50 | 2,081.92 ± 11.94 | 2,139.62 ± 12.03 | 2.7% | 42.49 |
| 200,000 | 183.15 ± 5.47 | 7,409.22 ± 42.70 | 7,592.37 ± 43.05 | 2.4% | 151.21 |

This configuration is systematically slower than `extractMatchRef=false`, and the gap grows with `N`. At `N=5,000`, total runtime is `1.28x` higher; at `N=200,000`, it is `6.09x` higher. Most of this penalty is associated with extraction, whose cumulative cost is `6.22x` higher in the largest dataset.

## 4. Scalability Analysis

### 4.1 Empirical Power-Law Fit

To characterize the observed growth, a model of the following form was fitted:

```text
T(N) = a * N^b
```

The fit was performed in log-log scale, separating construction, extraction, and total cost.

| Configuration | Component | Exponent b | R² |
| --- | --- | --- | --- |
| `extractMatchRef=false` | Build | 1.14 | 0.9979 |
| `extractMatchRef=false` | Cumulative extraction | 1.26 | 0.9995 |
| `extractMatchRef=false` | Cumulative total | 1.26 | 0.9994 |
| `extractMatchRef=true` | Build | 1.44 | 0.9933 |
| `extractMatchRef=true` | Cumulative extraction | 1.71 | 0.9952 |
| `extractMatchRef=true` | Cumulative total | 1.70 | 0.9951 |

The most relevant result is that `extractMatchRef=false` exhibits subquadratic growth and remains only moderately superlinear. The exponent `1.26` for total runtime indicates that doubling `N` tends to increase cumulative cost by approximately `2.4x`, which is consistent with the transition from `N=100,000` to `N=200,000`: total runtime increases from `520.29 s` to `1,246.93 s`, or `2.40x`.

In contrast, `extractMatchRef=true` shows an exponent close to `1.70`, indicating stronger degradation at scale. For the same doubling from `N=100,000` to `N=200,000`, total runtime increases from `2,139.62 s` to `7,592.37 s`, or `3.55x`.

### 4.2 Incremental Growth of the Recommended Configuration

| Transition in N | Increase in N | Build increase | Extraction increase | Total increase |
| --- | --- | --- | --- | --- |
| 5,000 -> 10,000 | 2.00x | 2.04x | 2.26x | 2.24x |
| 10,000 -> 20,000 | 2.00x | 2.13x | 2.30x | 2.29x |
| 20,000 -> 30,000 | 1.50x | 1.65x | 1.71x | 1.71x |
| 30,000 -> 40,000 | 1.33x | 1.49x | 1.50x | 1.50x |
| 40,000 -> 50,000 | 1.25x | 1.31x | 1.35x | 1.35x |
| 50,000 -> 60,000 | 1.20x | 1.10x | 1.21x | 1.21x |
| 60,000 -> 70,000 | 1.17x | 1.20x | 1.23x | 1.23x |
| 70,000 -> 80,000 | 1.14x | 1.35x | 1.25x | 1.25x |
| 80,000 -> 90,000 | 1.12x | 0.99x | 1.10x | 1.10x |
| 90,000 -> 100,000 | 1.11x | 1.13x | 1.14x | 1.14x |
| 100,000 -> 200,000 | 2.00x | 2.25x | 2.40x | 2.40x |

The incremental evolution reinforces the log-log interpretation. In general, total runtime closely follows cumulative extraction time, as expected from the dominance of extraction in the final cost. There is a local fluctuation at `N=80,000`, where the build standard deviation increases (`±4.09 s`) and the mean build time is slightly higher than at `N=90,000`. This point suggests experimental or environmental variability, but it does not alter the overall scalability trend.

## 5. Behavior as a Function of k

For each dataset size, the first value (`k=50`) includes an almost negligible extraction after the initial construction. For the remaining `k` values, extraction times vary only moderately within a fixed `N`.

In the `extractMatchRef=false` configuration, for example:

- at `N=5,000`, post-build extractions range from `0.23 s` to `0.27 s`;
- at `N=50,000`, they range from `4.09 s` to `4.65 s`;
- at `N=100,000`, they range from `9.61 s` to `10.86 s`;
- at `N=200,000`, they range from `23.24 s` to `25.85 s`.

This behavior suggests that, once `N` is fixed, the specific value of `k` has a secondary impact relative to dataset size. The dominant factor is the volume of data over which extraction is performed, not the exact position within the descending sequence of `k` values.

## 6. Internal Configuration Comparison

| N | Total ratio (`true` / `false`) | Extraction ratio (`true` / `false`) | Build ratio (`true` / `false`) |
| --- | --- | --- | --- |
| 5,000 | 1.28x | 1.30x | 1.09x |
| 10,000 | 1.40x | 1.42x | 1.14x |
| 20,000 | 1.67x | 1.69x | 1.24x |
| 30,000 | 1.96x | 1.99x | 1.32x |
| 40,000 | 2.51x | 2.57x | 1.49x |
| 50,000 | 2.88x | 2.96x | 1.64x |
| 60,000 | 3.20x | 3.27x | 1.90x |
| 70,000 | 3.43x | 3.50x | 2.02x |
| 80,000 | 3.58x | 3.67x | 1.93x |
| 90,000 | 3.87x | 3.96x | 2.21x |
| 100,000 | 4.11x | 4.20x | 2.37x |
| 200,000 | 6.09x | 6.22x | 3.34x |

The difference between configurations is not constant: it increases with `N`. This indicates that `extractMatchRef=true` introduces an additional cost that is amplified at scale, primarily during extraction. Under the analyzed protocol, `extractMatchRef=false` should therefore be considered the recommended configuration for larger executions.

## 7. Discussion

The results show that ScoreSG has a favorable execution regime when used for multiple extractions over a previously built support structure. The `extractMatchRef=false` configuration maintains subquadratic growth across the evaluated range and preserves a regular runtime profile, with strong fit to a power-law model.

From a practical standpoint, cumulative runtime is governed mainly by repeated extraction rather than by initial construction. This is important because it indicates that future optimizations should prioritize the extraction routine when the objective is to reduce total runtime in multi-`k` scenarios. Construction still grows superlinearly, but it represents a relatively small fraction of cumulative cost when `49` values of `k` are processed.

The specific value of `k` also does not substantially alter extraction time within a fixed `N`. The relative stability of post-build extractions suggests that the algorithm behaves predictably across the hierarchy sequence, which is desirable for exploratory applications where multiple granularities are evaluated successively.

## 8. Conclusion

The analysis of the `core_sg_cython` files supports three main conclusions. First, `ScoreSG_extractMatchRef_false` is consistently superior to `extractMatchRef=true` across all evaluated dataset sizes. Second, the cumulative total cost of the recommended configuration grows approximately as `N^1.26`, suggesting favorable empirical scalability from `5,000` to `200,000` samples. Third, the main optimization opportunity lies in recurring extraction, since it dominates total runtime when the algorithm is used for multiple `k` values.

In summary, ScoreSG exhibits performance consistent with the goal of reusing an initial construction to support multiple hierarchical extractions. The method scales predictably, remains subquadratic in the recommended configuration, and shows stronger runtime robustness when `extractMatchRef=false` is used.
