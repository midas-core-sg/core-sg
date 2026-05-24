MST Extraction
==============

After reweighting, Core-SG extracts a minimum spanning tree with Kruskal's
algorithm.

The MST is the graph-level artifact used to reconstruct the hierarchy. Through
the estimator, the current wrapped object is exposed as
``minimum_spanning_tree_`` after ``fit(X, k=...)``.

If the support graph is disconnected, MST extraction fails. This is most
visible in ``algorithm="score-sg"``, where the approximate support graph is not
silently repaired.
