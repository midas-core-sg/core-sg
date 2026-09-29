Benchmarking Workflow
=====================

Benchmark sources live in ``benchmarking/``.

Useful files:

``benchmarking/run_time/scripts/script.py``
   Main repeated multi-``k`` benchmark.

``benchmarking/run_time/scripts/score_sg_script.py``
   Score-SG focused experiments.

``benchmarking/run_time/results/``
   Runtime CSV result files.

``benchmarking/run_time/report_assets/``
   Generated runtime plots used by the documentation.

``benchmarking/missing_edges/``
   Missing-edge connectivity experiment and results.

``benchmarking/unties/``
   Anti-hub tie-breaking diagnostics and plots.

Documentation figures are refreshed with:

.. code-block:: bash

   python docs/scripts/generate_benchmark_figures.py

Always describe Core-SG benchmark claims as build-once, extract-many claims.
Do not present the cumulative advantage as a single-run guarantee.
