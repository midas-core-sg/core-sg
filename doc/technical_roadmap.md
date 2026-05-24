# Technical Roadmap

## Why this project exists

When practitioners compare several values of `k` in density-based clustering workflows, the expensive part is often not the final labels themselves, but the repeated reconstruction of the graph and MST support behind them.

Core-SG addresses that problem with a simple idea:

1. build the graph support once at `k_max`
2. reuse that support for smaller `k`
3. recover MSTs and hierarchy objects from the same fitted structure

This makes Core-SG especially useful when your workflow is **comparative**, **iterative**, or **graph-centric**.

## Value proposition

Core-SG helps you:

- **Reuse work across multiple `k` values** instead of rebuilding the full support for every experiment
- **Extract MSTs directly from Core-SG** for different `k` values
- **Keep an HDBSCAN-like interface** for labels, probabilities, and hierarchy objects
- **Inspect graph artifacts explicitly** when you need more than just final cluster labels
- **Prototype faster** in research and benchmarking workflows where `k` is part of the analysis

## Quick comparison

In the repository notebook `notebooks/01-HDBSCAN_comparision.ipynb`, a synthetic dataset with `n=5000`, `d=2`, `centers=10`, and repeated evaluations from `k=30` down to `k=10` shows the intended Core-SG tradeoff:

- **Initial build at `k=30`**: Core-SG takes longer because it builds the reusable support
- **Subsequent extractions for smaller `k`**: Core-SG reuses that support instead of repeating full HDBSCAN fits
- **Cumulative runtime across all tested `k` values**:
  - Core-SG: **9.76 s**
  - HDBSCAN: **32.44 s**

That notebook therefore shows a cumulative speedup of about **3.3×** for a multi-`k` workflow.

> Practical takeaway: if you only need one clustering result for one value of `k`, plain HDBSCAN may be enough. If you need to inspect or compare many `k` values, Core-SG becomes much more attractive.

## When to use Core-SG

Core-SG is a good fit when you want to:

- compare many values of `k` on the same dataset
- reuse graph support across repeated experiments
- extract MSTs for analysis, debugging, or downstream processing
- keep access to HDBSCAN-style objects such as:
  - `labels_`
  - `probabilities_`
  - `cluster_persistence_`
  - `condensed_tree_`
  - `single_linkage_tree_`
  - `minimum_spanning_tree_`
- build tooling around the graph/MST stage rather than around labels only
- run research notebooks, benchmarks, or internal evaluations where repeated recomputation is costly

## When not to use Core-SG

Core-SG is **not** the best choice when:

- you only need a **single** HDBSCAN run for one `k`
- you do not need explicit access to graph or MST artifacts
- your workflow does not compare multiple `k` values
- you want the most standard, widely documented clustering path with the smallest conceptual surface area
- you are looking for a drop-in replacement for every HDBSCAN use case

In those cases, using plain `hdbscan` directly may be simpler.

## Limitations and realistic expectations

Core-SG is intentionally focused. It should be understood as a **companion project**, not a full replacement for HDBSCAN.

Current limitations include:

- the main value comes from **multi-`k` reuse**, not necessarily from a single fit
- the exact `algorithm="core-sg"` path has a practical `n_samples` limitation because it relies on dense pairwise distance information
- `algorithm="score-sg"` is the scalable approximate path intended to relieve that dense construction bottleneck when sample size becomes the limiting factor
- the project still relies on `hdbscan` for important parts of the hierarchy post-processing pipeline
- users still need to understand the role of `k` in the workflow to interpret results correctly
- performance gains depend on the workload pattern; they are strongest when the same fitted support is reused many times

## Relationship with HDBSCAN

Core-SG is structurally inspired by and technically based on the `hdbscan` ecosystem.

In the current implementation, it relies on `hdbscan` for:

- reference behavior at `k_max`
- hierarchy post-processing
- single linkage tree conversion
- tree and MST wrapper objects

So the right mental model is:

**HDBSCAN** provides the familiar clustering ecosystem.  
**Core-SG** adds a reusable graph/MST layer for workflows that need repeated extraction across multiple `k` values.

If you already know HDBSCAN, the Core-SG interface should feel natural.

This document clarifies the current technical scope of **Core-SG**, the project dependencies that are critical to its current operation, the main known limitations, and the intended future direction of the library.

Its purpose is to help contributors, users, and stakeholders understand what is already stable in the project, what still depends on external components, and where the project is expected to evolve next.

## 1. Project maturity at a glance

Core-SG is currently best understood as a **specialized companion library for HDBSCAN-style workflows**, not as a full replacement for the broader HDBSCAN ecosystem.

At the current stage, the project is focused on:

- building a reusable graph support at `k_max`
- extracting MSTs for smaller values of `k`
- enabling repeated multi-`k` analysis without rebuilding the entire support from scratch each time
- exposing HDBSCAN-style clustering artifacts through a familiar workflow

This means the project already provides a clear practical value for graph-centric and comparative clustering workflows, while still depending on external components for part of the hierarchy and post-processing pipeline.

## 2. Current stable scope

The current stable scope of Core-SG is centered on the following capabilities:

### 2.1 Reusable support construction
Core-SG builds a reusable support structure using a reference value `k_max`, allowing subsequent extractions for smaller `k` values without repeating the full pipeline.

The exact construction path uses dense pairwise distance information and is therefore bounded in practice by `n_samples`. The ScoreSG path extends the same reuse idea with approximate sparse-neighbor construction, making it the preferred path when the exact dense construction no longer fits the target scale.

### 2.2 MST extraction for smaller `k`
Once fitted, Core-SG can extract minimum spanning tree information for smaller values of `k`, which is especially useful in benchmarking, diagnostics, and repeated comparative analysis.

### 2.3 HDBSCAN-style hierarchy workflow
Core-SG integrates with an HDBSCAN-style workflow and exposes familiar clustering outputs and tree-based artifacts, including objects such as:

- labels
- probabilities
- cluster persistence information
- condensed tree objects
- single linkage tree objects
- minimum spanning tree objects

### 2.4 Comparative multi-`k` workflows
The project is currently strongest in workflows where the same dataset is analyzed for multiple values of `k`, and repeated recomputation would otherwise become expensive.

## 3. What is explicitly in scope today

The following are in scope for the current project direction:

- efficient reuse across repeated `k` evaluations on the same dataset
- extraction of graph and MST artifacts for analysis
- HDBSCAN-style object exposure after hierarchy extraction
- support for research, experimentation, benchmarking, and internal graph-based clustering workflows
- a Python package experience centered on graph construction and MST reuse

## 4. What is not the primary scope today

The following should **not** be considered the primary scope of the project at its current stage:

- replacing every use case covered by standard `hdbscan`
- becoming a general-purpose clustering framework for unrelated algorithms
- optimizing only for single-run, single-`k` workflows
- removing all dependence on external clustering ecosystem components immediately
- providing a fully independent hierarchy/post-processing stack at the current maturity level

This distinction is important because the main value of Core-SG comes from **reuse and repeated extraction**, not from being the most minimal option for a one-off clustering run.

## 5. Critical dependencies

Core-SG currently depends on a small but important set of external libraries.

### 5.1 Runtime and packaging dependencies
The repository uses `pyproject.toml` as the main packaging configuration and source of truth for runtime dependencies. The current package metadata indicates dependency on libraries such as:

- `numpy`
- `hdbscan`
- `scikit-learn`
- `pandas`
- Python `>=3.10`

These libraries are fundamental to the current package behavior and development workflow.

### 5.2 HDBSCAN dependency
`hdbscan` is the most strategically important dependency in the current architecture.

At this stage, Core-SG still relies on HDBSCAN-related components for parts of the clustering and hierarchy ecosystem, including behavior and wrappers associated with tree and post-processing outputs.

This means:

- Core-SG should currently be seen as **built around the HDBSCAN ecosystem**
- compatibility and evolution should be considered with that dependency in mind
- users should expect conceptual continuity with HDBSCAN rather than a fully detached implementation

### 5.3 Scientific Python stack
The project also depends on the standard scientific Python stack for data representation, numeric operations, experimentation, and examples.

## 6. Current limitations

The following limitations should be considered part of the current state of the project.

### 6.1 Best payoff appears in repeated workflows
Core-SG provides its strongest value when the same fitted support is reused for several values of `k`.

If a user only needs a single run for one `k`, the additional structure introduced by Core-SG may not provide a significant practical advantage.

### 6.2 Dependence on external hierarchy components
Part of the hierarchy extraction and object wrapping still depends on the external HDBSCAN ecosystem.

This is acceptable for the current maturity stage, but it also means the project is not yet fully independent in all stages of the pipeline.

### 6.3 Performance gains are workload-dependent
Any performance benefit should be understood as **scenario-dependent**, especially in comparative multi-`k` workflows.

The project should not be described as universally faster in all clustering settings.

### 6.4 Exact CoreSG has a sample-size limitation
The traditional exact `algorithm="core-sg"` path builds dense pairwise distance information. This is useful for reference-style construction, but it creates a practical `n_samples` limitation in runtime and memory.

The current mitigation is `algorithm="score-sg"`, which avoids dense all-pairs construction by using approximate sparse-neighbor support. ScoreSG should be treated as the scalable option for larger repeated multi-`k` workflows, subject to the usual approximate-neighbor caveats around connectivity, reproducibility, and output validation.

### 6.5 Narrower scope than a full clustering framework
Core-SG is intentionally focused on graph support reuse and MST-centered workflows. Users looking for a broad, general clustering toolkit may find the scope narrower than expected.

## 7. Recommended usage today

Core-SG is currently most appropriate when:

- multiple values of `k` must be evaluated on the same dataset
- MST artifacts are needed for inspection, diagnostics, or downstream use
- users want to keep an HDBSCAN-like workflow while avoiding repeated full recomputation
- experiments, notebooks, or internal tools need graph-level artifacts rather than labels only
- exact dense construction is feasible, or ScoreSG is acceptable as the scalable approximate path for larger `n_samples`

## 8. When stakeholders should be cautious

Stakeholders should set more conservative expectations when:

- the use case involves only a single clustering fit
- graph or MST artifacts are not needed
- the team expects a fully standalone replacement for HDBSCAN today
- the evaluation is based on one isolated runtime measurement rather than repeated reuse scenarios

## 9. Future direction

The intended future direction of Core-SG can be summarized as follows.

### 9.1 Clarify and harden the reusable-core architecture
The project should continue strengthening its identity as a reusable graph-support layer for repeated clustering analysis.

This includes clearer contracts around what is stored at `k_max`, what is reusable, and what guarantees are provided for smaller `k` extractions.

### 9.2 Reduce architectural ambiguity
Future iterations should make it easier to distinguish:

- what is owned directly by Core-SG
- what is delegated to HDBSCAN or other upstream components
- what is an implementation detail versus a public API contract

### 9.3 Improve documentation for technical adoption
A future goal is to continue improving project documentation so users can more easily understand:

- supported workflows
- expected tradeoffs
- integration patterns
- maturity boundaries
- benchmark interpretation

### 9.4 Expand maturity of the independent pipeline
Over time, one strategic direction may be to reduce reliance on external post-processing or wrapper components where that meaningfully improves maintainability, control, and project identity.

This should be treated as a medium- to long-term direction rather than a short-term guarantee.

### 9.5 Improve benchmarking discipline
Future iterations should continue to document the conditions under which Core-SG performs well, especially in repeated multi-`k` scenarios, so performance claims remain transparent and reproducible.

## 10. Roadmap view: now vs next

### Current stage
Today, Core-SG is a focused library for:

- reusable support construction at `k_max`
- repeated extraction for smaller `k`
- MST- and graph-aware clustering workflows
- integration with HDBSCAN-style hierarchy artifacts

### Next stage
The next maturity steps are expected to focus on:

- clearer documentation of architecture boundaries
- reduced ambiguity around dependency responsibilities
- stronger communication of tradeoffs and supported workflows
- gradual strengthening of Core-SG as a more explicit technical layer in the clustering stack

## 11. Summary for stakeholders

In practical terms, Core-SG is already useful when the problem is:

> “We need to analyze the same dataset across multiple values of `k`, reuse the graph support, and keep access to MST and HDBSCAN-style artifacts.”

At the same time, the project is still evolving in how it separates its own responsibilities from the broader HDBSCAN ecosystem.

That makes the current maturity level best described as:

- **useful and technically meaningful today**
- **focused rather than general-purpose**
- **dependent on upstream ecosystem components in important parts**
- **moving toward clearer architectural boundaries in future iterations**
