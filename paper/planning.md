# Pre-Manuscript Planning for ScoreSG

## 1. Paper Type

Type A -- New method or architecture. The paper presents ScoreSG, an approximate graph-construction path for the Core-SG Python library. The method changes the construction of the reusable support graph by combining approximate k-nearest-neighbor discovery with anti-hub reinforcement.

## 2. Research Question

Can an anti-hub reinforced approximate support graph reduce cumulative runtime for repeated multi-k HDBSCAN-style hierarchy extraction while avoiding the explicit dense all-pairs construction used by exact Core-SG?

## 3. Hypotheses

H1: In a repeated multi-k workflow, ScoreSG has lower cumulative runtime than exact CoreSG and HDBSCAN baselines on the available synthetic benchmark.

H2: Anti-hub reinforcement provides a sparse way to address missing-edge and connectivity risks without constructing the complete mutual-reachability graph.

H3: Optimized extraction is required for ScoreSG to preserve favorable scaling at larger sample sizes.

## 4. Literature Gap

HDBSCAN provides density-based hierarchical clustering and robust treatment of noise, but repeated exploration over many values of the density smoothing parameter requires repeated execution. Core-SG addresses repeated multi-k extraction by reusing support structures, but the exact path still depends on dense pairwise information. Approximate kNN construction reduces neighbor-discovery cost, but sparse kNN graphs can omit MST edges and can be disconnected. The current gap is therefore a sparse construction that is practical for repeated multi-k extraction while making its missing-edge and connectivity assumptions explicit.

## 5. Proposed Contributions

1. We contribute ScoreSG, a sparse approximate support-graph construction for Core-SG, demonstrated by its implementation in the Python `core-sg` package, in contrast with the exact Core-SG path that materializes dense pairwise information.
2. We contribute an anti-hub reinforcement procedure based on directed in-degree and a sqrt(N)-sized clique, demonstrated by the method formulation and implementation, in contrast with random or neighborhood-expansion heuristics that do not target low in-degree vertices.
3. We contribute a runtime evaluation for repeated multi-k hierarchy extraction over synthetic `make_blobs` datasets, in contrast with single-k HDBSCAN execution.
4. We contribute a transparent limitation profile for ScoreSG, demonstrated by explicit disconnected-graph failure modes and by marking unavailable quality, memory, and external-dataset evidence as open information.

## 6. Detailed Structure

1. Introduction
2. Related Work
3. Problem Formulation and Preliminaries
4. ScoreSG Method
5. Experimental Setup
6. Main Results
7. Ablation and Diagnostic Analysis
8. Robustness, Error Analysis, and Limitations
9. Ethics, Reproducibility, and Broader Impact
10. Conclusion
11. Appendices A--H

## 7. Required Experiments

Already available:

- Runtime comparison over synthetic datasets with N in {5000, 10000, 20000, 30000, 40000, 50000}.
- ScoreSG-only scaling up to N=200000.
- Comparison among ScoreSG optimized extraction, exact CoreSG optimized extraction, HDBSCAN best, and HDBSCAN generic.
- Reference-vs-optimized extraction comparisons from the benchmarked method variants.

Still needed:

- Clustering quality comparison against exact Core-SG and HDBSCAN.
- MST edge recall or hierarchy agreement.
- Connectivity failure rate over multiple datasets, dimensions, densities, and seeds.
- Anti-hub ablation without clique, random sqrt(N) clique, larger/smaller anti-hub sets, and no exact distance recomputation for selected edges.
- Memory profiling.
- Real-dataset evaluation.
- Statistical tests or confidence intervals from raw per-run traces.

## 8. Missing Information

- Dataset license for synthetic benchmark: generated data, but script/data licensing should be explicitly stated.
- Hardware, operating system, CPU model, memory, and parallelism settings.
- Full library versions at benchmark time.
- Raw per-repetition traces for cumulative confidence intervals.
- Approximate-neighbor hyperparameters beyond defaults.
- Clustering quality metrics.
- Memory usage.
- External datasets and real-world validation.
- Formal theoretical guarantee for anti-hub reinforcement.
- Author list and affiliations.
- Code/data release URL or artifact DOI.

## 9. Scientific Risks

- Runtime improvements do not imply clustering quality preservation.
- Synthetic `make_blobs` data may favor neighborhood-based methods and underrepresent difficult density structures.
- A single dataset seed does not support broad generalization.
- Anti-hub reinforcement may introduce long edges or alter the hierarchy in sparse or manifold-shaped data.
- The support graph can be disconnected; ScoreSG explicitly fails in this case.
- Cumulative standard deviations reconstructed from per-k summaries are not a substitute for raw-run confidence intervals.

## 10. Figures and Tables

Figures:

1. Overview of the Core-SG and ScoreSG problem setting.
2. ScoreSG pipeline: approximate kNN, in-degree scoring, anti-hub selection, clique reinforcement, MST/hierarchy extraction.
3. Cumulative runtime comparison on the common benchmark interval.
4. ScoreSG extended scaling to N=200000.

Tables:

1. Dataset and benchmark characterization.
2. Model and algorithm configuration.
3. Main cumulative runtime results.
4. Ablation plan and unavailable ablation evidence.
5. Robustness and missing evaluation matrix.
