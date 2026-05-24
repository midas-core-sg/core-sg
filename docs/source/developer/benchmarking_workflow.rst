Benchmarking Workflow
=====================

Benchmark sources live in ``benchmarking/``.

Useful files:

``benchmarking/script.py``
   Main repeated multi-``k`` benchmark.

``benchmarking/score_sg_script.py``
   Score-SG focused experiments.

``benchmarking/results/``
   CSV result files.

``benchmarking/report_assets/``
   Generated plots used by the documentation.

Documentation figures are refreshed with:

.. code-block:: bash

   python docs/scripts/generate_benchmark_figures.py

Always describe Core-SG benchmark claims as build-once, extract-many claims.
Do not present the cumulative advantage as a single-run guarantee.
