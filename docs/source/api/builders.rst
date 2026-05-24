Builder Functions
=================

.. py:function:: build_core_sg_from_data(X, k_max, *, metric="euclidean", p=2, pairwise_dtype=float, _round_distances=False)

   Build the reusable Core-SG support structures from input data.

.. py:function:: mst_from_core_sg(core_sg, metric_edges, core_k_list, n_nodes, k, *, verbose=0, progress_callback=None)

   Reweight the support graph for ``k`` and extract an MST.

.. py:function:: core_sg_mutual_reachability_distance(core_sg, metric_edges, core_k_list, n_nodes, k_max, k)

   Return the target-``k`` mutual-reachability weighted support graph.

These functions are documented for contributors and advanced users. Most users
should prefer :class:`core_sg.CoreSGClusterer`.
