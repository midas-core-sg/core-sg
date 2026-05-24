# Scikit-learn-style estimator API

Core-SG exposes two public workflows:

- `CoreSG`: the native reusable multi-`k` API.
- `CoreSGClusterer`: a scikit-learn-style estimator wrapper for a single fitted
  extraction `k`.

Use `CoreSG` when you want the native graph API directly. Use
`CoreSGClusterer` when you want estimator-style parameter introspection,
cloning, `fit(...)`, and `fit_predict(...)` behavior while still preserving the
Core-SG build-once, extract-many workflow.

## Basic estimator workflow

```python
from core_sg import CoreSGClusterer

clusterer = CoreSGClusterer(k_max=15)
clusterer.fit(X, k=10)

labels = clusterer.labels_
```

`k_max` is a constructor parameter because it defines the reusable support graph
capacity. It is visible through `get_params()` and can be changed with
`set_params(...)`.

On the first `fit(...)`, `CoreSGClusterer` builds the native `CoreSG` support
once by calling `CoreSG.fit(X, k_max=k_max)`. On later `fit(...)` calls, the
existing `core_sg_` object is reused and only
`core_sg_.extract_hierarchy_from_core_sg(k)` is executed. The decision is based
on whether `core_sg_` already exists, not on whether `k == k_max`.

`k` is passed to `fit(...)` because it defines the concrete clustering
extraction for the current fitted result. If `k=None`, `fit(...)` uses the
fitted `k_max_`.

```python
clusterer = CoreSGClusterer(k_max=15)
clusterer.fit(X)  # equivalent to clusterer.fit(X, k=15)
clusterer.fit(X, k=10)  # reuses core_sg_ and extracts k=10 only
clusterer.fit(X, k=8)  # reuses core_sg_ again and extracts k=8 only
```

## Constructor parameters

`CoreSGClusterer` exposes explicit constructor parameters:

```python
CoreSGClusterer(
    k_max,
    metric="euclidean",
    p=2,
    algorithm="core-sg",
    no_noise=True,
    noise_label_strategy="mst_label_propagation",
    random_state=None,
    approx_knn_kwargs=None,
    verbose=0,
    progress_callback=None,
    cluster_selection_method="eom",
    allow_single_cluster=False,
    match_reference_implementation=False,
    cluster_selection_epsilon=0.0,
    cluster_selection_persistence=0.0,
    max_cluster_size=0,
    cluster_selection_epsilon_max=float("inf"),
)
```

The wrapper stores constructor parameters unchanged. Mutable dictionaries such
as `approx_knn_kwargs` are copied only when the internal `CoreSG` instance is
built.

## Fitted attributes

After `fit(X, k=...)`, these attributes correspond to the fitted `k`:

- `labels_`
- `probabilities_`
- `cluster_persistence_`
- `condensed_tree_`
- `single_linkage_tree_`
- `minimum_spanning_tree_`

The fitted neighborhood values are also stored:

- `k_`: the `k` used for the fitted extraction
- `k_max_`: the support graph capacity used for the native `CoreSG` fit

The fitted native object is exposed as:

```python
clusterer.core_sg_
```

Advanced users can use it for native multi-`k` workflows:

```python
clusterer.core_sg_.extract_hierarchy_from_core_sg(k=8)
labels_at_8 = clusterer.core_sg_.labels_
```

Most users can call `clusterer.fit(X, k=...)` repeatedly to update the public
fitted attributes for different values of `k` without rebuilding the support
graph.

## `fit_predict(...)`

`fit_predict(...)` fits the estimator and returns `labels_`:

```python
labels = clusterer.fit_predict(X, k=10)
```

This is equivalent to:

```python
clusterer.fit(X, k=10)
labels = clusterer.labels_
```

## Prediction for unseen samples

`CoreSGClusterer` intentionally does not implement `predict(...)` yet. Core-SG
is currently a fit/extract clustering workflow, and assigning unseen samples to
existing clusters needs a separate, well-defined approximate prediction
strategy.

## Input support

The sklearn-facing wrapper validates dense feature matrices with scikit-learn
validation utilities. Sparse input and `metric="precomputed"` are not supported
by the wrapper in this first estimator-oriented API. Use the native `CoreSG`
workflow for lower-level experimentation.
