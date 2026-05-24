Benchmark Methodology
=====================

Core-SG performance should be interpreted as a build-once, extract-many
workflow. The initial build can be more expensive than a single HDBSCAN run,
but repeated extraction for multiple ``k`` values can amortize that cost.

The exact ``algorithm="core-sg"`` path and the approximate
``algorithm="score-sg"`` path should be interpreted separately. Exact CoreSG
still has a dense pairwise construction step, which creates a practical
``n_samples`` limit. ScoreSG is evaluated as the scalable approximate path that
avoids that dense all-pairs construction and is therefore the relevant method
when sample size becomes the limiting factor.

The benchmark material in ``benchmarking/`` evaluates repeated multi-``k``
workloads over synthetic datasets. The central comparison is cumulative time
for all requested ``k`` values, not only the first result.

The report assets are copied into the documentation by
``docs/scripts/generate_benchmark_figures.py``.
