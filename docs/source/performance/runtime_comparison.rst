Runtime Comparison
==================

ScoreSG approximate path
------------------------

ScoreSG should be included in the primary runtime comparison because it changes
the practical scalability limit of the traditional exact CoreSG path. Exact
CoreSG reuses work across many ``k`` values, but its construction still relies
on dense pairwise distance information. This makes it increasingly constrained
as ``n_samples`` grows. ScoreSG addresses that limitation by using an
approximate sparse-neighbor construction, avoiding the dense all-pairs stage
that dominates the exact path at larger scale.

.. image:: ../_static/images/benchmarks/score_sg_common_runtime.png
   :alt: ScoreSG, exact CoreSG, and HDBSCAN cumulative runtime in the common benchmark interval

In the common benchmark interval from ``N=5,000`` to ``N=50,000``, ScoreSG with
optimized extraction is faster than optimized exact CoreSG and both HDBSCAN
baselines. At ``N=50,000``, ScoreSG completes the full repeated multi-``k``
workflow in ``224.36 s``, compared with ``525.45 s`` for optimized exact
CoreSG, ``1,342.44 s`` for optimized HDBSCAN ``best``, and ``9,272.92 s`` for
optimized HDBSCAN ``generic``.

This corresponds to approximately ``2.34x`` speedup over optimized exact
CoreSG, ``5.98x`` speedup over optimized HDBSCAN ``best``, and ``41.33x``
speedup over optimized HDBSCAN ``generic`` at the largest directly comparable
sample size. The result is not merely a constant-factor improvement: ScoreSG
also shows a lower empirical growth rate in the benchmark, which is why it is
the preferred option when the exact dense construction becomes constrained by
sample size.

For the extended ScoreSG-only scaling results, see :doc:`score_sg_results`.

Exact CoreSG and HDBSCAN baseline figures
-----------------------------------------

.. image:: ../_static/images/benchmarks/cumulative_runtime_by_samples.png
   :alt: Cumulative runtime by sample size

.. image:: ../_static/images/benchmarks/per_k_runtime_grid.png
   :alt: Per-k runtime comparison

The cumulative view is the primary performance view for Core-SG because it
charges the build step once and then sums repeated extraction work.

The per-``k`` view is useful for seeing the initial build cost and the much
smaller extraction costs that follow.
