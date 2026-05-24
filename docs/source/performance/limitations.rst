Limitations
===========

Exact CoreSG and sample-size limits
-----------------------------------

The traditional exact ``algorithm="core-sg"`` path has a practical
``n_samples`` limitation because it builds and stores dense pairwise distance
information internally. This can become the dominant cost as the number of
samples grows, both in runtime and in memory pressure. The exact path remains
useful when reference-style graph construction is required and the dataset size
is still compatible with dense pairwise computation, but it should not be
presented as the scalable option for every large-``N`` workload.

ScoreSG is the intended scalable alternative for this limitation. By using an
approximate sparse-neighbor construction, ScoreSG avoids materializing the full
all-pairs distance matrix and extends the repeated multi-``k`` workflow to
larger sample sizes. In the current benchmark suite, optimized ScoreSG scales
to ``N=200,000`` with an empirical cumulative-runtime exponent near ``1.26``.
This does not make ScoreSG a universal guarantee for all datasets, but it does
directly address the exact CoreSG sample-size bottleneck observed in dense
construction.

General benchmark limits
------------------------

Performance results depend on:

* sample size;
* feature dimension;
* metric choice;
* selected ``k_max`` and target ``k`` values;
* HDBSCAN version and private API behavior;
* whether Cython extensions are available;
* PyNNDescent warm-up and approximate neighbor settings.

The benchmark pages should not be read as universal guarantees. They document
the intended repeated multi-``k`` interpretation and provide reproducible
scripts and figures for the repository's benchmark data.

For large sample counts, prefer reading the exact CoreSG and ScoreSG results
together: exact CoreSG documents the reusable dense construction baseline,
while ScoreSG documents the scalable approximate path that is intended to
relieve the dense ``n_samples`` limitation.
