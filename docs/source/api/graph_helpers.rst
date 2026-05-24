Graph Helpers
=============

``core_sg.edges``
   Builds vectorized kNN edge arrays and merges MST edges into metric edges.

``core_sg.knn``
   Extracts nearest-neighbor arrays from a precomputed distance matrix.

``core_sg.reweight``
   Sorts Core-SG edge arrays and applies mutual-reachability reweighting.

``core_sg.mst_kruskal``
   Provides Kruskal MST extraction with a Cython backend and Python fallback.

Key helper functions include ``build_knng_vectors``,
``add_mst_edges_to_metric_edges``, ``knn_from_precomputed``,
``reweight_core_sg_mutual_reachability``, ``sort_core_sg``, and
``kruskal_mst``.
