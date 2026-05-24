# Score-SG Implementation Notes

This document explains how `algorithm="score-sg"` is currently implemented in the codebase, what the execution flow is, which parts are shared with classical `core-sg`, and which design decisions were intentionally changed to keep the implementation aligned with the current goals of the project.

ScoreSG is also the intended scalable alternative when the traditional exact `algorithm="core-sg"` path becomes limited by `n_samples`. The exact path relies on dense pairwise distance information, while ScoreSG avoids the explicit all-pairs construction by working from an approximate sparse-neighbor graph.

Its purpose is simple:

- make the implemented behavior easy to audit
- clarify what the code is actually doing today
- help validate whether the implementation matches the intended `score-sg` semantics

## 1. Where `score-sg` fits in the project

`CoreSG` now supports two internal graph-construction flows:

- `algorithm="core-sg"`
- `algorithm="score-sg"`

The public entry point remains the same class:

```python
from core_sg import CoreSG

core = CoreSG(algorithm="score-sg", random_state=42)
core.fit(X, k_max=10)
```

The key architectural choice is that the **class itself performs the algorithm dispatch**, rather than placing algorithm selection inside a generic shared builder.

This is closer to the style used by HDBSCAN, where:

- one main estimator coordinates the workflow
- the algorithm parameter selects a concrete internal path
- shared post-processing is reused when possible

In this repository, the dispatch happens in:

- `CoreSG.fit(...)`
- `CoreSG._build_by_algorithm(...)`

inside [core_sg.py](/home/gab04/Desktop/core-sg/core_sg/core_sg.py).

## 2. High-level flow of `score-sg`

When `CoreSG.fit(X, k_max)` is called with `algorithm="score-sg"`, the flow is:

1. `CoreSG.fit(...)` calls `CoreSG._build_by_algorithm(...)`
2. the class routes to `build_score_sg_from_data(...)`
3. `score-sg` constructs a support graph using approximate neighbors
4. the class stores the resulting support structures
5. the class tries to extract the `k_max` MST from the support graph itself
6. if MST extraction succeeds, the usual `_tree_to_labels(...)` path is reused
7. if the support graph is disconnected, `fit(...)` raises an explicit error

So the important difference from the previous version is:

- `score-sg` does **not** call `hdbscan.HDBSCAN(...)`
- `score-sg` does **not** build an auxiliary MST to force connectivity
- `score-sg` must stand on the support graph it actually constructed

## 3. Main implementation files

The main files involved are:

- [core_sg.py](/home/gab04/Desktop/core-sg/core_sg/core_sg.py)
- [score_sg.py](/home/gab04/Desktop/core-sg/core_sg/score_sg.py)

Responsibilities are split as follows.

### 3.1 `core_sg/core_sg.py`

This file remains the central coordinator.

For `score-sg`, it is responsible for:

- validating the public configuration
- dispatching to the correct construction path
- storing fitted artifacts
- attempting MST extraction at `k_max`
- reusing `_tree_to_labels(...)` once an MST exists
- surfacing a clear error when the support graph is disconnected

### 3.2 `core_sg/score_sg.py`

This file contains the algorithm-specific construction logic for `score-sg`.

It is responsible for:

- building the approximate `k_max` nearest-neighbor graph with `PyNNDescent`
- deriving the approximate core-distance list from approximate neighbor distances
- computing anti-hub scores by directed in-degree
- selecting anti-hubs with tie handling
- building the anti-hub clique
- assembling the final support graph and `metric_edges`
- checking support-graph connectivity through a helper used by the class

## 4. Approximate kNN construction

The approximate kNN graph is built in:

- `build_approximate_knn_graph(...)`

inside [score_sg.py](/home/gab04/Desktop/core-sg/core_sg/score_sg.py).

It uses:

- `pynndescent.NNDescent`

Important implementation details:

- the metric is normalized through `_resolve_metric(...)`
- `"arccos"` is mapped to `"cosine"`
- `"minkowski"` forwards `p` through `metric_kwds`
- `n_neighbors` is forced to be at least `k_max + 1` when possible so we can safely discard self-neighbors and still keep `k_max` useful neighbors

After receiving the neighbor graph, the code normalizes it with:

- `_normalize_neighbor_graph(...)`

That helper:

- removes self-neighbors
- keeps only the first `k_max` valid neighbors
- sorts each row by approximate distance using stable ordering

The result is:

- `neighbor_indices` with shape `(n_samples, k_max)`
- `neighbor_distances` with shape `(n_samples, k_max)`

## 5. Core-distance handling in `score-sg`

This is one of the most important differences from classical `core-sg`.

For `score-sg`, the implementation does **not** compute exact pairwise distances and does **not** derive exact core-distances.

Instead, it builds:

- `core_k_list`

from the approximate neighbor distances returned by `PyNNDescent`.

This happens in:

- `build_approximate_core_k_list(...)`

The behavior is:

- column `0` is set to `0.0`
- columns `1:` are filled from the sorted approximate neighbor distances

So for `score-sg`, the core-distance information used later in reweighting is approximate by construction.

This is intentional in the current implementation.

## 6. Anti-hub definition

Anti-hubs are defined through **in-degree in the directed approximate kNN graph**.

This happens in:

- `compute_in_degrees(...)`

The logic is:

- each point appears in the outgoing neighbor lists of other points
- the number of occurrences of a point in those lists is its directed in-degree
- lower in-degree means a stronger anti-hub candidate

This matches the intended interpretation of anti-hubs as points that are rarely chosen as neighbors by others.

## 7. Anti-hub selection logic

Selection happens in:

- `select_score_sg_anti_hubs(...)`

The implemented procedure is:

1. compute `m = max(1, floor(sqrt(n_samples)))`
2. sort points by:
   - lower in-degree
   - increasing index as a stable secondary ordering
3. determine the cutoff degree at position `m`
4. keep every point strictly below that degree
5. handle the tied boundary group with a second criterion
6. if needed, break the remaining tie randomly with `random_state`

### 7.1 Secondary tie-break

For tied candidates, the code computes:

- the sum of the in-degrees of the candidate's outgoing neighbors

Lower sum is preferred.

This is implemented with:

```python
tie_scores = in_degrees[neighbor_indices[tied]].sum(axis=1)
```

### 7.2 Final tie-break

If a tie still remains inside the same secondary score bucket, selection is randomized through:

- `sklearn.utils.check_random_state(random_state)`

This keeps the final selection reproducible when a seed is provided.

## 8. Edge distances used by `score-sg`

The implementation intentionally distinguishes between:

- approximate neighbor discovery
- exact edge-weight recovery for support edges

### 8.1 kNN support edges

The approximate kNN structure tells us **which pairs** belong to the support graph.

But the actual distances stored in `metric_edges` for those support edges are recomputed exactly from `X` using:

- `_compute_exact_edge_distances(...)`

This uses `sklearn.metrics.pairwise.paired_distances`.

So in the current implementation:

- neighbor membership is approximate
- support-edge stored distances are exact for the selected pairs

### 8.2 Clique edges among selected anti-hubs

After anti-hub selection, the chosen points are fully connected in:

- `build_selected_clique(...)`

The clique edges are also assigned exact distances computed from the raw data and the configured metric.

## 9. Final support graph assembly

The final `score-sg` support graph is assembled in:

- `build_score_sg_from_data(...)`

It is the union of:

- approximate kNN support edges
- clique edges among the selected anti-hubs

The helper returns:

- `core_sg`
- `metric_edges`
- `core_k_list`
- `tree_to_labels_data`

Important note:

- for `score-sg`, `tree_to_labels_data` is the raw feature matrix `X`
- for classical `core-sg`, the equivalent artifact is still the dense distance matrix `D`

This is why the class stores:

- `self._tree_to_labels_data_`

instead of assuming `_D` always exists.

## 10. Why `_D` is `None` for `score-sg`

The class now stores two related but different concepts:

- `self.distance_matrix_`
- `self._tree_to_labels_data_`

Behavior:

- for `core-sg`, `_D` is the exact dense pairwise distance matrix
- for `score-sg`, `_D` is set to `None`
- for `score-sg`, `_tree_to_labels_data` stores `X`

This was introduced so that:

- `score-sg` does not require dense all-pairs distance computation
- `_tree_to_labels(...)` can still run using the raw data matrix, just like HDBSCAN usually does for feature-space inputs

## 11. `k_max` artifacts for `score-sg`

Another important difference from the classical path is how fit-time artifacts at `k_max` are obtained.

For `core-sg`, the fit-time artifacts come from the HDBSCAN reference fit.

For `score-sg`, the fit-time artifacts are derived from the support graph itself:

1. the class checks whether the support graph is connected
2. if connected, it extracts an MST from the support graph at `k_max`
3. it converts that MST to a single linkage tree
4. it calls `_tree_to_labels(...)`
5. it stores the resulting labels, probabilities, persistence, condensed tree, and MST as the `k_max` artifacts

This logic lives in:

- `CoreSG._extract_score_sg_k_max_outputs(...)`

This means that, for `score-sg`, the cached `k_max` outputs are **not** imported from HDBSCAN and are instead reconstructed from the graph that `score-sg` itself built.

## 12. Connectivity check and failure mode

The current implementation explicitly accepts that `score-sg` may fail to produce a connected support graph.

Connectivity is checked through:

- `is_graph_connected(...)`

If the support graph is disconnected, `fit(...)` raises:

- `"score-sg support graph is disconnected for k_max, so an MST cannot be extracted from the constructed support graph."`

This is intentional.

The code does **not**:

- build an auxiliary MST
- add reference-MST edges to force connectivity
- silently patch the support graph

Instead, it surfaces the limitation clearly so the user knows the support graph produced by that run is not sufficient to support MST extraction.

The same idea is respected for later extractions:

- if `extract_mst_from_core_sg(k)` fails because the graph is disconnected, the class raises a `score-sg`-specific explanatory error

## 13. Shared downstream logic

Even though the graph construction changes, most of the downstream flow is still shared.

The following pieces remain reused:

- `reweight_core_sg_mutual_reachability(...)`
- `mst_from_core_sg(...)`
- `extract_mst_from_core_sg(...)`
- `extract_hierarchy_from_core_sg(...)`
- noise-label post-processing

So the design is:

- different graph-construction logic
- shared reweighting and hierarchy pipeline once a valid support graph exists

## 14. Public parameters relevant to `score-sg`

The public constructor now accepts:

```python
CoreSG(
    algorithm="score-sg",
    random_state=42,
    approx_knn_kwargs=None,
)
```

### 14.1 `algorithm`

Must be one of:

- `"core-sg"`
- `"score-sg"`

### 14.2 `random_state`

Used only in `score-sg` where random tie-breaking is necessary.

### 14.3 `approx_knn_kwargs`

Optional dictionary forwarded to `PyNNDescent`.

This allows later tuning of parameters such as:

- `n_trees`
- `n_iters`
- `max_candidates`

subject to the constraints of the underlying `pynndescent` API.

## 15. What `score-sg` currently does not do

The current implementation intentionally does **not** do the following:

- compute a dense exact pairwise matrix `D`
- compute exact core-distances
- run `hdbscan.HDBSCAN(...)` as part of the `score-sg` fit path
- inject auxiliary MST edges to force graph connectivity
- guarantee that the support graph will be connected

These are not accidental omissions; they are part of the current design direction.

## 16. Test coverage added for `score-sg`

The current test suite includes dedicated checks for:

- approximate core-distance construction from approximate neighbors
- directed in-degree counting
- anti-hub selection
- tie-breaking behavior
- reproducible seeded random tie-breaking
- clique creation among selected anti-hubs
- connectivity detection
- class-level dispatch for `score-sg`
- explicit fit failure on disconnected support graphs
- end-to-end `score-sg` fit and hierarchy extraction on a connected dataset

The main files covering this are:

- [tests/unit/test_score_sg.py](/home/gab04/Desktop/core-sg/tests/unit/test_score_sg.py)
- [tests/unit/test_core_sg_fit.py](/home/gab04/Desktop/core-sg/tests/unit/test_core_sg_fit.py)
- [tests/integration/test_reference_equivalence.py](/home/gab04/Desktop/core-sg/tests/integration/test_reference_equivalence.py)

## 17. Practical validation checklist

If you want to validate whether the implementation matches the intended design, the most important questions are:

1. Should `score-sg` use approximate core-distances from the approximate kNN graph?
   Current implementation: yes.
2. Should `score-sg` avoid calling `hdbscan.HDBSCAN(...)`?
   Current implementation: yes.
3. Should `score-sg` avoid computing dense exact `D`?
   Current implementation: yes.
4. Should `score-sg` fail explicitly when the support graph is disconnected?
   Current implementation: yes.
5. Should the estimator class itself coordinate algorithm dispatch?
   Current implementation: yes.

If any of those answers should be different, then the implementation should be revised further.

## 18. Summary

The implemented `score-sg` is currently best understood as:

- an approximate graph-construction variant of `CoreSG`
- coordinated by the main `CoreSG` class
- using `PyNNDescent` to build the initial support
- deriving core-distance information from approximate neighbors
- enriching the support graph with anti-hub clique edges
- reusing the shared MST and hierarchy machinery only when the resulting support graph is connected

The most important caveat is:

- `score-sg` may legitimately fail on disconnected support graphs, and the current code treats that as an explicit user-visible outcome rather than silently repairing the graph

## 19. Why the current flow avoids explicit quadratic behavior

One of the main goals of `score-sg` is to avoid the explicit quadratic cost of building a dense pairwise distance matrix over all pairs of points.

The most important statement is:

- the current `score-sg` implementation does **not** build dense `D`
- the current `score-sg` implementation does **not** run an all-pairs exact kNN extraction

That is the main reason the flow avoids the classical `O(N^2)` step present in dense pairwise pipelines.

That said, the correct technical description is:

- the flow is designed to avoid the explicit quadratic stage
- most stages are linear or near-linear in the number of support edges
- some stages still involve sorting or ANN internals, so the flow should not be described as “strictly linear in every step”

### 19.1 Practical notation

To reason about the current implementation, define:

- `N`: number of samples
- `k_max`: neighborhood size used to build the support
- `m = floor(sqrt(N))`: number of selected anti-hubs

In the intended operating regime:

- `k_max` is relatively small compared to `N`
- `m` grows as `sqrt(N)`

Under that regime, the support graph stays sparse.

## 20. Step-by-step complexity intuition

This section explains the implemented behavior stage by stage.

### 20.1 Approximate kNN graph construction

Implemented in:

- `build_approximate_knn_graph(...)`

This stage delegates neighbor discovery to `PyNNDescent`.

Why this avoids explicit quadratic behavior:

- the code does not compare every point against every other point
- the code does not materialize a dense `N x N` matrix
- the ANN backend works directly on the raw feature matrix

What to keep in mind:

- this should be viewed as an approximate sparse-neighbor construction stage
- it is typically much better than dense all-pairs construction in practice
- but the code should not claim a strict formal linear guarantee for the ANN backend itself

So the right interpretation is:

- this stage avoids explicit `O(N^2)` dense distance construction
- it is meant to behave in a scalable sparse-graph regime

### 20.2 Normalizing the returned neighbor graph

Implemented in:

- `_normalize_neighbor_graph(...)`

This helper processes the `neighbor_graph` returned by `PyNNDescent`.

What it does for each point:

1. removes the self-neighbor if present
2. keeps only `k_max` useful neighbors
3. sorts only that local neighbor list

Why this is not quadratic:

- the code iterates over `N` rows
- each row handles only about `k_max + 1` entries
- it never scans all `N` points inside each row

So the cost scales with:

- `N * k_max`

plus a small local sort over each row.

That means:

- sparse local work per point
- not all-pairs work over the dataset

### 20.3 Building the approximate `core_k_list`

Implemented in:

- `build_approximate_core_k_list(...)`

What it does:

- writes one row of `core_k_list` per point
- fills the row directly from the approximate neighbor distances

Why this is not quadratic:

- each point contributes only its own local neighbor distances
- the code does not compare that point against all other points

So the work grows with:

- `N * k_max`

not with `N^2`.

### 20.4 Computing in-degree

Implemented in:

- `compute_in_degrees(...)`

What it does:

- flattens the directed approximate kNN structure
- counts how often each point appears in other points’ neighbor lists

Why this is not quadratic:

- the total number of directed neighbor references is about `N * k_max`
- the code counts those references once

So again the work is proportional to the sparse graph size, not to all pairs.

### 20.5 Selecting anti-hubs

Implemented in:

- `select_score_sg_anti_hubs(...)`

This stage has two parts:

1. global ranking by in-degree
2. local tie handling through neighbor-degree sums

Why it still avoids quadratic behavior:

- the code does not compare every point with every other point
- tie handling only looks at the outgoing neighbor lists of tied candidates
- those lists are bounded by `k_max`

What is slightly different here:

- the initial ranking requires sorting the `N` candidate points

So this stage is better described as:

- sparse plus sorting
- not dense all-pairs computation

In other words:

- it is not quadratic in the dataset size
- but it is not “pure linear scan only” either

### 20.6 Recomputing exact distances only for support edges

Implemented in:

- `_compute_exact_edge_distances(...)`

This is a very important design point.

The current implementation does compute exact distances, but only for:

- the approximate kNN edges actually selected into the support
- the clique edges among selected anti-hubs

It does **not** compute exact distances for all point pairs.

Why this avoids quadratic behavior:

- the code computes exact distances only for the sparse edge list
- the number of kNN support edges is about `N * k_max`

So instead of:

- all-pairs exact distance recovery

the implementation performs:

- exact distance recovery only on support edges

This is one of the central reasons the current flow stays sparse.

### 20.7 Building the anti-hub clique

Implemented in:

- `build_selected_clique(...)`

At first glance, building a clique may sound dangerous because a clique is quadratic in the number of selected points.

That is true with respect to the selected subset:

- a clique over `m` points has about `m^2 / 2` edges

But the implementation uses:

- `m = floor(sqrt(N))`

So the clique size becomes:

- `O(m^2) = O(N)`

Why this matters:

- the clique is quadratic in `m`
- but because `m` itself grows like `sqrt(N)`, the total number of clique edges grows only linearly in `N`

This is exactly why the `sqrt(N)` design is important.

Without that restriction, the clique could easily become the first explicitly quadratic stage.

### 20.8 Assembling the final support graph

Implemented in:

- `build_score_sg_from_data(...)`

The final support graph is built from:

- approximate kNN support edges
- anti-hub clique edges

The total number of support edges is therefore approximately:

- `N * k_max`
- plus `O(N)` from the clique

So if `k_max` is treated as small relative to `N`, the support graph remains sparse.

This is very different from a dense graph over all points, which would require:

- `O(N^2)` edges

### 20.9 Connectivity check

Implemented in:

- `is_graph_connected(...)`

This stage performs a graph traversal over:

- the vertices
- the sparse support edges

Why this is not quadratic:

- the traversal touches each node and edge a bounded number of times
- it works on the sparse support graph, not on all pairs

So the complexity follows the graph size, not the dense dataset size.

### 20.10 Reweighting and MST extraction

Implemented through:

- `reweight_core_sg_mutual_reachability(...)`
- `mst_from_core_sg(...)`

These stages operate on the sparse support graph that was already built.

Why they avoid explicit quadratic behavior:

- they do not start from a dense `N x N` graph
- they work only on the support edges already selected by the previous stages

This means the downstream cost follows:

- the number of support edges

not:

- the number of all possible pairs

## 21. Simple worked intuition

Suppose:

- `N = 1,000,000`
- `k_max = 20`
- `m = floor(sqrt(N)) = 1000`

Then:

- approximate kNN references are about `20,000,000`
- clique edges are about `1000 * 999 / 2`, which is about `499,500`

So the total support remains on the order of:

- tens of millions of sparse edges

That is large, but still far from:

- `1,000,000^2 = 1,000,000,000,000`

possible dense pairwise relations.

This is the key intuition behind the current implementation:

- work on a sparse graph of selected edges
- never build the full dense relation structure

## 22. What would reintroduce quadratic behavior

The current flow avoids the explicit dense quadratic stage, but some bad choices could still destroy that property.

Examples:

- building dense exact `D`
- computing exact distances for all pairs instead of only support edges
- letting `k_max` grow proportionally with `N`
- selecting too many anti-hubs instead of using `sqrt(N)`
- building a dense repair graph when the support is disconnected

The current implementation avoids all of those.

## 23. Correct wording for the current implementation

The safest way to describe the implemented complexity is:

- `score-sg` avoids the explicit quadratic all-pairs distance construction
- it operates on a sparse support graph whose size is driven mainly by `N * k_max` plus a linear-size anti-hub clique
- several stages are linear in the sparse graph size
- some stages include sorting or ANN internals, so the flow is better described as sparse, scalable, and near-linear in practice rather than “strictly linear at every single step”

That is the most faithful description of what the code is doing today.
