# HDBSCAN Adapter Notes

This document explains how Core-SG currently isolates its direct dependency on
HDBSCAN internals, why that separation exists, and what changed in the package
structure.

The goal of this adaptation is not to remove the HDBSCAN dependency. The goal
is to keep HDBSCAN-specific behavior behind a small internal adapter so the
main `CoreSG` implementation can depend on Core-SG-owned functions instead of
importing HDBSCAN private APIs directly.

## 1. Why this adapter exists

Core-SG interoperates with the HDBSCAN ecosystem for several important parts of
the current workflow:

- building the reference HDBSCAN result at `k_max`
- reading the reference minimum spanning tree
- converting MST arrays into HDBSCAN-style single linkage trees
- converting hierarchy trees into labels, probabilities, persistence values,
  and condensed tree artifacts
- exposing HDBSCAN-compatible wrapper objects for tree inspection

Some of this behavior currently depends on HDBSCAN private APIs. Those APIs are
useful for preserving HDBSCAN-style behavior, but they are not guaranteed to be
stable by HDBSCAN as public contracts.

For that reason, Core-SG now isolates the direct usage of those private APIs in
one internal module:

```text
core_sg/hdbscan_adapter.py
```

This keeps the risk localized. If HDBSCAN changes an internal function or object
shape in the future, the first place to adapt should be the adapter module, not
the main `CoreSG` class.

## 2. Previous structure

Before the adapter separation, `core_sg/core_sg.py` imported HDBSCAN directly,
including private modules:

```python
import hdbscan
from hdbscan._hdbscan_linkage import label
from hdbscan.hdbscan_ import _tree_to_labels
from hdbscan.plots import CondensedTree, MinimumSpanningTree, SingleLinkageTree
```

That meant the main `CoreSG` implementation was responsible for both:

1. the Core-SG workflow and estimator-like state
2. the details of how HDBSCAN exposes internal hierarchy functionality

This made `core_sg/core_sg.py` harder to evolve toward a cleaner architecture,
especially for future sklearn-friendly work.

## 3. Current structure

The direct HDBSCAN imports now live in `core_sg/hdbscan_adapter.py`.

The main `core_sg/core_sg.py` module imports Core-SG-owned adapter functions:

```python
from .hdbscan_adapter import (
    mst_to_single_linkage_tree,
    reference_mst_original_distance,
    tree_to_labels as hdbscan_tree_to_labels,
    wrap_condensed_tree,
    wrap_minimum_spanning_tree,
    wrap_single_linkage_tree,
)
```

This creates the intended dependency direction:

```text
CoreSG -> core_sg.hdbscan_adapter -> HDBSCAN internals
```

Instead of:

```text
CoreSG -> HDBSCAN internals
```

The public `CoreSG` API remains unchanged.

## 4. Adapter responsibilities

The adapter currently owns the following responsibilities.

### 4.1 Reference MST construction

`reference_mst_original_distance(...)` builds the HDBSCAN reference object used
at `k_max` and returns its MST as a NumPy array.

It preserves the previous HDBSCAN configuration:

```python
hdbscan.HDBSCAN(
    min_cluster_size=k_max,
    min_samples=k_max,
    metric="precomputed",
    algorithm="generic",
    approx_min_span_tree=False,
    gen_min_span_tree=True,
    match_reference_implementation=True,
)
```

This keeps the reference behavior used by Core-SG unchanged.

### 4.2 MST to single linkage conversion

`mst_to_single_linkage_tree(...)` wraps the HDBSCAN linkage conversion function.

`CoreSG` no longer calls HDBSCAN's linkage function directly. It calls the
adapter function instead.

### 4.3 Tree-to-label conversion

`tree_to_labels(...)` wraps the HDBSCAN tree post-processing path.

The main `CoreSG` class still decides which keyword arguments are allowed
through `_get_tree_to_labels_kwargs()`. The adapter receives those already
filtered arguments and forwards them to HDBSCAN.

This preserves the existing parameter filtering behavior.

### 4.4 HDBSCAN tree wrappers

The adapter also owns wrapper creation for:

- condensed trees
- single linkage trees
- minimum spanning trees

The `CoreSG` properties still expose the same HDBSCAN-style objects as before,
but construction now happens through adapter functions.

## 5. What did not change

This refactor is intentionally mechanical.

The following public behavior did not change:

- `from core_sg import CoreSG`
- `CoreSG(...)`
- `fit(X, k_max, _round_distances=False)`
- `extract_mst_from_core_sg(k, toDF=False)`
- `extract_hierarchy_from_core_sg(k, c=5)`
- `get_fitted_hdbscan_objects(wrapped=True)`
- `labels_`
- `probabilities_`
- `cluster_persistence_`
- `condensed_tree_`
- `single_linkage_tree_`
- `minimum_spanning_tree_`
- `*_k_max` artifacts
- `algorithm="core-sg"`
- `algorithm="score-sg"`

The adapter separation should not require users or notebooks to change how they
use Core-SG.

## 6. What changed internally

The internal changes are:

- `core_sg/core_sg.py` no longer imports HDBSCAN private APIs directly.
- `core_sg/hdbscan_adapter.py` centralizes direct HDBSCAN interaction.
- the local `tree_to_labels(...)` helper in `core_sg/core_sg.py` remains as a
  compatibility-preserving delegator to the adapter.
- tests that previously patched the internal `label` symbol now patch
  `mst_to_single_linkage_tree`.
- adapter-specific tests validate the HDBSCAN boundary.

This keeps existing tests and behavior stable while making the dependency
boundary explicit.

## 7. Testing and validation

The adapter boundary is covered by:

- focused adapter unit tests in `tests/unit/test_hdbscan_adapter.py`
- existing `CoreSG` initialization, fit, hierarchy, and artifact tests
- integration tests using real HDBSCAN
- validation tests comparing Core-SG behavior with HDBSCAN reference behavior

The test suite validates both algorithm paths:

- `algorithm="core-sg"`
- `algorithm="score-sg"`

A boundary regression test also checks that `core_sg/core_sg.py` does not
reintroduce direct HDBSCAN private imports.

## 8. Remaining architectural risk

This adapter does not eliminate the risk of using HDBSCAN private APIs.

It only localizes that risk.

The project still depends on HDBSCAN internals for HDBSCAN-style hierarchy
outputs. If those internals change in a future HDBSCAN release, Core-SG may
still need compatibility work. The intended benefit is that such work should be
concentrated in `core_sg/hdbscan_adapter.py`.

## 9. Future improvements

Future iterations can build on this separation by:

- introducing structured internal artifact objects instead of long tuples
- adding clearer contracts around HDBSCAN-derived hierarchy artifacts
- documenting supported HDBSCAN versions more explicitly
- evaluating which HDBSCAN-dependent pieces can eventually be replaced by
  Core-SG-owned implementations
- creating a future sklearn-style adapter without exposing HDBSCAN internals
  through the estimator-facing code

The adapter is therefore a first step toward a cleaner architecture, not the
final endpoint of the HDBSCAN integration strategy.
