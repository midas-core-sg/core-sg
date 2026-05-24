# Core-SG

[![PyPI version](https://img.shields.io/pypi/v/core-sg.svg)](https://pypi.org/project/core-sg/)
[![Python versions](https://img.shields.io/pypi/pyversions/core-sg.svg)](https://pypi.org/project/core-sg/)
[![Tests](https://img.shields.io/github/actions/workflow/status/midas-core-sg/core-sg/test.yml?branch=develop&label=tests)](https://github.com/midas-core-sg/core-sg/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

Core-SG - Core Support Graph for efficient computation of multiple MSTs and HDBSCAN-style hierarchy outputs over varying values of `k`. The recommended public workflow is the scikit-learn-style `CoreSGClusterer`, which builds reusable graph support at `k_max` and then extracts hierarchy artifacts for smaller values of `k <= k_max` without rebuilding the full structure each time.

In practice this means that `CoreSGClusterer` lets you fit once at `k_max` and reuse the result straight away for many smaller `k` values with little or no extra setup, while keeping familiar HDBSCAN-like outputs such as `labels_`, `probabilities_`, `cluster_persistence_`, `condensed_tree_`, `single_linkage_tree_`, and `minimum_spanning_tree_`.

Core-SG is ideal for exploratory multi-`k` density-based analysis; it is a practical approach for workflows where you want to compare smoothing levels on the same dataset and inspect graph-level artifacts, not only final labels.

Based on the papers:


>Antonio Cavalcante Araujo Neto, Murilo Coelho Naldi, Ricardo J. G. B. Campello, and Jorg Sander. CORE-SG: Efficient Computation of Multiple MSTs for Density-Based Methods. In: 2022 IEEE 38th International Conference on Data Engineering (ICDE), IEEE, pp. 951-964. 2022.



>Leland McInnes and John Healy. Accelerated Hierarchical Density Based Clustering. In: 2017 IEEE International Conference on Data Mining Workshops (ICDMW), IEEE, pp. 33-42. 2017.



>R. Campello, D. Moulavi, and J. Sander. Density-Based Clustering Based on Hierarchical Density Estimates. In: Advances in Knowledge Discovery and Data Mining, Springer, pp. 160-172. 2013.


Documentation and project overview are available at https://midas-core-sg.github.io/core-sg/. Notebooks comparing Core-SG to HDBSCAN and illustrating the intended multi-`k` workflow are available in [`notebooks/`](notebooks/).

## What Core-SG is for

Core-SG is designed for workflows where you need to compare multiple `k` values on the same dataset and you care about graph-level artifacts, not only final labels. Most users should start with `CoreSGClusterer`; the lower-level reusable object acts behind the estimator and remains available for advanced internal workflows.

In practice, Core-SG helps you:

- reuse support across repeated `k` evaluations
- extract minimum spanning trees for different `k`
- keep an HDBSCAN-like workflow (`labels_`, `probabilities_`, `cluster_persistence_`)
- inspect tree artifacts (`condensed_tree_`, `single_linkage_tree_`, `minimum_spanning_tree_`)

## Installing

Install from PyPI:

```bash
pip install core-sg
```

Install for local development:

```bash
pip install -e .
```

Install with development tooling:

```bash
pip install -e ".[dev]"
```

Dependencies:

- `numpy>=1.24,<3`
- `pandas>=2.0`
- `scikit-learn>=1.3`
- `hdbscan>=0.8.39`
- `pynndescent>=0.5.13`

The package metadata, runtime dependencies, and optional extras are defined in `pyproject.toml`.

## How to use Core-SG

Use `CoreSGClusterer` for the public user-facing workflow:

```python
from sklearn.datasets import make_blobs
from core_sg import CoreSGClusterer

X, _ = make_blobs(
    n_samples=1000,
    n_features=10,
    centers=10,
    random_state=42,
)

clusterer = CoreSGClusterer(k_max=15)
clusterer.fit(X, k=10)

labels = clusterer.labels_
```

In this API, `k_max` is a constructor parameter because it defines the reusable
support graph capacity and is visible through `get_params()` / `set_params()`.
The first `fit(X, y=None, *, k=...)` builds the internal reusable Core-SG object
once. Later calls to `fit(...)` reuse `core_sg_` and only extract the hierarchy
for the requested `k`. The decision to build or extract is based on whether
`core_sg_` already exists, not on whether `k == k_max`.

The `k` argument defines the specific clustering extraction exposed by
`labels_` and the other fitted attributes. If `k=None`, `fit` uses the fitted
`k_max_`.

```python
clusterer.fit(X, k=10)
labels_10 = clusterer.labels_

clusterer.fit(X, k=8)  # reuses the same core_sg_ object
labels_8 = clusterer.labels_
```

`CoreSGClusterer` exposes fitted artifacts for the selected `k`:

- `labels_`
- `probabilities_`
- `cluster_persistence_`
- `condensed_tree_`
- `single_linkage_tree_`
- `minimum_spanning_tree_`
- `k_`
- `k_max_`
- `core_sg_`

`fit_predict(X, y=None, *, k=...)` is also available and returns `labels_`.
`predict(...)` is intentionally not implemented yet because Core-SG does not
currently define assignment semantics for unseen samples.

More details are available in [`doc/estimators.md`](doc/estimators.md).

### Quick workflow

The most common usage pattern is:

1. choose a largest neighborhood value `k_max`
2. create `CoreSGClusterer(k_max=...)`
3. call `fit(X, k=...)` for each `k` value that you want to compare
4. inspect `labels_`, `probabilities_`, persistence values, and tree objects

### Primary example

```python
from sklearn.datasets import make_blobs
from core_sg import CoreSGClusterer

# Example dataset used only to illustrate the workflow.
X, _ = make_blobs(
    n_samples=1000,
    n_features=10,
    centers=10,
    random_state=42,
)

# Build reusable support once and expose outputs for k=10.
clusterer = CoreSGClusterer(k_max=15, metric="euclidean", p=2)
clusterer.fit(X, k=10)

# Read the HDBSCAN-style outputs exposed on the estimator.
labels = clusterer.labels_
probabilities = clusterer.probabilities_
cluster_persistence = clusterer.cluster_persistence_

# Later calls reuse the same internal core_sg_ object.
clusterer.fit(X, k=8)
labels_8 = clusterer.labels_
```

In this example, `k_max=15` is the largest neighborhood size used during the
initial fit, while `k=10` and `k=8` are extracted afterward from the same
internal fitted support graph.

### Key parameters

- `k_max`: largest neighborhood size configured on `CoreSGClusterer`; this is the reference value that defines what smaller `k` values can later be extracted
- `k`: neighborhood size passed to `fit(X, k=...)`; it must satisfy `2 <= k <= k_max`
- `metric`: distance metric used to build the support graph
- `p`: metric power parameter for distance families such as Minkowski
- `algorithm`: choose `"core-sg"` for the default exact workflow or `"score-sg"` for the approximate anti-hub reinforced variant
- `no_noise`: when `True`, applies an optional post-processing step so final labels do not remain at `-1`
- `noise_label_strategy`: selects the post-processing strategy used when `no_noise=True`

### Algorithms

`algorithm="core-sg"` is the default exact path. It builds dense pairwise
distance information internally, constructs reusable support at `k_max`, and
is the recommended first choice when the exact graph construction cost is
acceptable. Because this path materializes dense pairwise distance information,
it has a practical `n_samples` limitation as datasets grow.

`algorithm="score-sg"` is the approximate path. It uses PyNNDescent for
approximate neighbor discovery, selects anti-hubs by directed in-degree, and
uses `random_state` plus `approx_knn_kwargs` to control reproducibility and
approximate-neighbor behavior. It can be useful when dense all-pairs distance
construction is too expensive, but the resulting support graph may be
disconnected for some data and parameter settings. In practice, this is the
scalable path to try when the traditional exact CoreSG construction becomes
limited by sample size.

### Using the approximate variant

To enable the approximate anti-hub reinforced variant, set
`algorithm="score-sg"`:

```python
clusterer = CoreSGClusterer(
    k_max=15,
    metric="euclidean",
    p=2,
    algorithm="score-sg",
    random_state=42,
)
clusterer.fit(X, k=10)
```

### Extracting only an MST

The estimator exposes the current HDBSCAN-style MST wrapper after `fit(...)`:

```python
clusterer.fit(X, k=10)
mst = clusterer.minimum_spanning_tree_
```

### Extracting hierarchy outputs

Call `fit(X, k=...)` when you want clustering outputs and tree artifacts
similar to HDBSCAN:

```python
clusterer.fit(X, k=10)

labels = clusterer.labels_
probabilities = clusterer.probabilities_
cluster_persistence = clusterer.cluster_persistence_
```

After hierarchy extraction, the current instance also exposes:

- `condensed_tree_`
- `single_linkage_tree_`
- `minimum_spanning_tree_`

### Reassigning noise labels

If you prefer a full assignment with no final `-1` labels, keep the
post-processing step enabled with `no_noise=True` (the default). The current
strategy, `noise_label_strategy="mst_label_propagation"`, updates only
`labels_` after hierarchy extraction and leaves the remaining hierarchy
artifacts unchanged.

```python
clusterer = CoreSGClusterer(
    k_max=15,
    metric="euclidean",
    p=2,
    no_noise=True,
    noise_label_strategy="mst_label_propagation",
)

clusterer.fit(X, k=10)
labels = clusterer.labels_
```

This is useful when you want a final label assignment for every point, while
still preserving the original extracted hierarchy objects.

This post-processing flow is inspired by the density-connectivity
label-propagation view discussed in:

- Gertrudes, J. C., Zimek, A., Sander, J., and Campello, R. J. G. B.  
  *A unified view of density-based methods for semi-supervised clustering and classification*.  
  Data Mining and Knowledge Discovery, 33, 1894-1952 (2019).  
  DOI: `10.1007/s10618-019-00651-1`

### Inspecting tree objects

If you need direct access to the current extracted hierarchy objects:

```python
condensed_tree = clusterer.condensed_tree_
single_linkage_tree = clusterer.single_linkage_tree_
minimum_spanning_tree = clusterer.minimum_spanning_tree_
```

### Accessing artifacts stored at `k_max`

The estimator keeps the internal reusable object at `clusterer.core_sg_` for
advanced inspection. Most users can stay on the estimator attributes above.

```python
fitted = clusterer.core_sg_.get_fitted_hdbscan_objects(wrapped=True)
```

Returned keys:

- `labels_`
- `probabilities_`
- `cluster_persistence_`
- `condensed_tree_`
- `single_linkage_tree_`
- `minimum_spanning_tree_`

Direct cached wrappers at fit time:

- `condensed_tree_k_max_`
- `single_linkage_tree_k_max_`
- `minimum_spanning_tree_k_max_`

## Performance (multi-k workflows)

Core-SG is optimized for repeated `k` analysis, not necessarily for a single one-off run.

In `notebooks/01-HDBSCAN_comparision.ipynb`, for a synthetic setup (`n=5000`, `d=2`, `centers=10`) with repeated evaluations from `k=30` down to `k=10`, cumulative runtime was:

- Core-SG: `9.76 s`
- HDBSCAN: `32.44 s`

This notebook demonstrates the intended tradeoff: higher upfront cost at `k_max`, lower cumulative cost when reusing across multiple smaller `k` values.

For larger sample sizes, prefer the approximate `algorithm="score-sg"` path.
The exact `algorithm="core-sg"` path still relies on dense pairwise distance
information and can become constrained by `n_samples`. The current ScoreSG
benchmarks show substantially better cumulative runtime in repeated multi-`k`
workloads: at `n=50000`, ScoreSG completes the tested `49`-value workflow in
`224.36 s`, compared with `525.45 s` for optimized exact CoreSG and
`1342.44 s` for optimized HDBSCAN `best`.

## Known limitations

- Core-SG provides strongest gains in repeated multi-`k` usage
- exact `algorithm="core-sg"` can be limited by `n_samples` because it uses
  dense pairwise distance information
- use `algorithm="score-sg"` when that dense exact construction becomes too
  costly and an approximate sparse-neighbor workflow is acceptable
- for single `k` workflows, plain HDBSCAN may be simpler
- current hierarchy pipeline still depends on HDBSCAN ecosystem components

## Python version

Core-SG supports Python `>=3.10`.

## Help and support

- Documentation and project overview: https://midas-core-sg.github.io/core-sg/
- Issues: https://github.com/midas-core-sg/core-sg/issues

## Contributing

Contributions are welcome. Please follow the contribution workflow in
[`CONTRIBUTING.md`](CONTRIBUTING.md), including the local test commands and
development checks documented there.

## Acknowledgment

Core-SG is structurally inspired by and technically based on the `hdbscan` ecosystem.

- HDBSCAN repository: https://github.com/scikit-learn-contrib/hdbscan
- HDBSCAN documentation: https://hdbscan.readthedocs.io/en/latest/

Core-SG also interoperates with internal `hdbscan` APIs to reconstruct and
expose HDBSCAN-style hierarchy artifacts. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
for third-party attribution and the reproduced upstream BSD-3-Clause notice.

## Citing

If Core-SG contributes to your research, publication, or technical results,
please cite the following paper:

Antonio Cavalcante Araujo Neto, Murilo Coelho Naldi, Ricardo J. G. B.
Campello, and Jorg Sander. *CORE-SG: Efficient Computation of Multiple MSTs
for Density-Based Methods*. In: 2022 IEEE 38th International Conference on
Data Engineering (ICDE), pp. 951-964, IEEE, 2022.

BibTeX:

```bibtex
@inproceedings{neto2022core_sg,
  author = {Neto, Antonio Cavalcante Araujo and Naldi, Murilo Coelho and Campello, Ricardo J. G. B. and Sander, Jorg},
  title = {{CORE-SG}: Efficient Computation of Multiple MSTs for Density-Based Methods},
  booktitle = {2022 IEEE 38th International Conference on Data Engineering (ICDE)},
  pages = {951--964},
  year = {2022},
  publisher = {IEEE},
  doi = {10.1109/ICDE53745.2022.00076},
  url = {https://doi.org/10.1109/ICDE53745.2022.00076}
}
```

## License

Core-SG is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.

This repository also includes third-party attribution and license information
for `hdbscan` in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## References


### HDBSCAN
- Repository: https://github.com/scikit-learn-contrib/hdbscan
- Documentation: https://hdbscan.readthedocs.io/en/latest/
- Basic usage: https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html
- Advanced usage: https://hdbscan.readthedocs.io/en/latest/advanced_hdbscan.html
- API reference: https://hdbscan.readthedocs.io/en/latest/api.html
