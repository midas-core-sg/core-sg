Performance
===========

The performance section distinguishes between the exact CoreSG path and the
approximate ScoreSG path. Exact CoreSG demonstrates reusable multi-``k``
extraction, but its dense pairwise construction creates a practical
``n_samples`` limit. ScoreSG is the scalable alternative for larger sample
counts and is therefore included as a first-class performance result.

.. toctree::
   :maxdepth: 1

   methodology
   repeated_multi_k
   runtime_comparison
   score_sg_results
   scaling
   interpreting_results
   limitations
