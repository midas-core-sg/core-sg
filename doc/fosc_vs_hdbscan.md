# FOSC Timing Differences Between Core-SG and Native HDBSCAN

This document explains why the hierarchy post-processing stage (`FOSC` in the local timing logs) can appear slower when executed through the Core-SG workflow than when using the native HDBSCAN pipeline directly.

The goal is not to claim that one implementation is universally better than the other, but to clarify that the two execution paths are not identical and therefore should not be expected to have identical timing behavior.

## 1. What is being compared

There are two different workflows involved in the comparison:

### 1.1 Native HDBSCAN workflow

In the native HDBSCAN path, the library controls the full clustering pipeline end to end:

- neighbor search
- core distance computation
- minimum spanning tree construction
- single linkage conversion
- hierarchy post-processing

When HDBSCAN receives raw data directly, it may also choose optimized internal execution paths depending on the metric, data representation, and algorithm selection.

### 1.2 Core-SG workflow

In the Core-SG path, the workflow is split:

1. build the reusable Core-SG support
2. extract a mutual reachability MST for a chosen `k`
3. convert that MST to a single linkage tree
4. call HDBSCAN internals for hierarchy post-processing

In other words, Core-SG does not call the exact same integrated internal path that native HDBSCAN uses from raw input to final hierarchy output.

## 2. What the code is doing today

The current Core-SG implementation uses a reference-oriented HDBSCAN configuration during the initial fit:

- `algorithm="generic"`
- `approx_min_span_tree=False`
- `match_reference_implementation=True`

This can be seen in the reference fit helper used during Core-SG construction.

Later, when a hierarchy is reconstructed for a given `k`, the flow is:

1. extract the MST from Core-SG
2. call `hdbscan._hdbscan_linkage.label(...)`
3. call `hdbscan.hdbscan_._tree_to_labels(...)`

That means the hierarchy stage is executed after a custom MST reconstruction path rather than through the full native HDBSCAN pipeline.

## 3. Why FOSC can look slower in Core-SG

There are several concrete reasons for the timing difference.

### 3.1 Core-SG is using the generic/reference-oriented path

The current configuration explicitly favors reference-compatible behavior:

- `algorithm="generic"` tends to be more general but less optimized
- `match_reference_implementation=True` prioritizes behavioral fidelity
- `approx_min_span_tree=False` disables the approximation shortcut

This is already enough to make the hierarchy stage less likely to match the fastest native HDBSCAN execution mode.

### 3.2 The pipeline is no longer monolithic

In native HDBSCAN, the implementation controls all stages as one internal pipeline.

In Core-SG, the MST is reconstructed externally and then passed into the HDBSCAN post-processing internals afterward.

That changes:

- where memory is allocated
- which intermediate arrays are materialized
- how data flows between stages
- what assumptions the downstream code can exploit

Even if the final hierarchy logic is the same in spirit, the execution path is not the same.

### 3.3 The MST stage became much faster, so FOSC is now more visible

After the Cython acceleration, the `reweight` and especially the `kruskal` portions became much faster.

That changes the performance profile of the whole workflow:

- before the optimization, MST extraction dominated the timing
- after the optimization, the hierarchy stage becomes a larger fraction of the remaining runtime

This can create the impression that `FOSC` got slower, when in practice it may simply have become the next visible bottleneck.

### 3.4 Approximate HDBSCAN is solving a different performance problem

When HDBSCAN is run with approximation enabled, the observed runtime can be much lower.

However, that comparison should be interpreted carefully:

- the approximate MST path is not equivalent to the exact path
- the resulting hierarchy and cluster structure may differ
- timing gains there do not imply that Core-SG is doing unnecessary work

So if approximate HDBSCAN looks much faster, that is expected. It is not a like-for-like comparison with the exact, reference-compatible Core-SG route.

## 4. Why this does not contradict the value of Core-SG

Core-SG is designed to amortize work across multiple values of `k`.

Its main value proposition is:

- build the support once
- reuse it for multiple MSTs and hierarchies

That means the right comparison is often not:

- one Core-SG hierarchy extraction versus one isolated HDBSCAN call

but rather:

- many `k` evaluations with Core-SG reuse
- many separate HDBSCAN runs from scratch

Even if the hierarchy post-processing stage is not faster by itself, the overall multi-`k` workflow may still be substantially more efficient.

## 5. Likely explanation for the current measurements

Based on the current code, the most likely explanation is the combination of:

1. Core-SG using the exact/reference-compatible HDBSCAN path
2. Core-SG reconstructing the MST externally before calling hierarchy post-processing
3. Cython removing most of the previous MST bottleneck, which makes FOSC stand out more clearly

So the current evidence points more toward:

- a difference in execution path
- a difference in optimization strategy
- a change in bottleneck visibility

and less toward:

- a clear bug in the Core-SG hierarchy integration

## 6. What would be worth investigating next

This topic is still worth investigating because there may be room for improvement.

Reasonable follow-up directions include:

- benchmarking `extract_hierarchy_from_core_sg(...)` against native HDBSCAN under matched settings only
- isolating the runtime of `label(...)` versus `_tree_to_labels(...)`
- checking whether additional HDBSCAN parameters should be exposed for a fast hierarchy mode
- evaluating whether a non-reference mode should be supported for users who prefer speed over strict equivalence
- measuring whether the dense/precomputed distance handoff is contributing avoidable overhead

## 7. Practical conclusion

At the moment, the observed difference is explainable from the codebase:

- Core-SG is not using the same execution path as the fastest native HDBSCAN modes
- Core-SG intentionally favors exact/reference-compatible behavior
- after the Cython optimization, the hierarchy stage is now the most visible remaining cost

This is therefore a valid optimization target, but the current timing difference is not surprising given the implementation choices.
