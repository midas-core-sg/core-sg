Interpreting Benchmark Results
==============================

.. image:: ../_static/images/benchmarks/speedup_heatmap.png
   :alt: Speedup heatmap

Interpret Core-SG results in terms of total analysis cost.

If a workflow needs one clustering result, the Core-SG build cost may not be
worth paying. If the workflow needs many ``k`` values over the same ``X``, the
one-time build can be amortized by extraction reuse.

Exact CoreSG and ScoreSG answer different scaling questions. Exact CoreSG
shows the benefit of reuse when dense pairwise construction is still feasible.
ScoreSG should be used to evaluate the scalable approximate path once
``n_samples`` makes dense construction too expensive. In the current benchmark
data, ScoreSG provides both lower cumulative runtime and a lower empirical
growth rate than exact CoreSG over the directly comparable sample sizes.

How to read the ScoreSG results
-------------------------------

ScoreSG should be interpreted as the approximate scalable path, not merely as
another implementation variant. The traditional exact CoreSG path demonstrates
the build-once, extract-many idea, but it still pays for dense pairwise distance
information. ScoreSG changes that construction regime by using an approximate
sparse-neighbor graph.

The main ScoreSG findings are:

* In the common interval from ``N=5,000`` to ``N=50,000``, optimized ScoreSG is
  the fastest method at every measured sample size.
* At ``N=50,000``, optimized ScoreSG takes ``224.36 s`` for all ``49`` values
  of ``k``. Optimized exact CoreSG takes ``525.45 s``, HDBSCAN ``best`` takes
  ``1,342.44 s``, and HDBSCAN ``generic`` takes ``9,272.92 s``.
* The empirical cumulative-runtime exponent for optimized ScoreSG is
  approximately ``1.247`` in the common interval, compared with ``1.575`` for
  optimized exact CoreSG, ``1.601`` for optimized HDBSCAN ``best``, and
  ``2.113`` for optimized HDBSCAN ``generic``.
* In the ScoreSG-only extended benchmark, optimized ScoreSG reaches
  ``N=200,000`` with ``1,246.93 s`` cumulative runtime and an empirical
  exponent near ``1.26``.

The most important interpretation is that ScoreSG addresses the practical
``n_samples`` limitation of exact CoreSG. If exact CoreSG is already feasible
and exact construction is preferred, it remains a useful baseline. If the
dataset is large enough that dense pairwise construction becomes the
bottleneck, ScoreSG is the method that should be considered first.

Extraction optimization also matters. At ``N=200,000``, the reference ScoreSG
extraction path is approximately ``6.09x`` slower than optimized ScoreSG.
Therefore, the observed speedup should be attributed to the combination of
approximate sparse graph construction and optimized repeated extraction.

Practical decision rule
-----------------------

Use the results as follows:

* Use HDBSCAN directly when a single clustering run is enough and hierarchy
  reuse is not required.
* Use exact CoreSG when repeated ``k`` extraction is needed and dense pairwise
  construction is still acceptable for the target ``n_samples``.
* Use ScoreSG when repeated multi-``k`` analysis is needed and sample size makes
  exact dense construction too expensive.

PyNNDescent and Score-SG results may show warm-up effects. Treat first-run
timings carefully and prefer repeated measurements when reporting results.
