Expected Outputs
================

``CoreSGClusterer`` exposes current artifacts for the most recent fitted
``k``. Its internal ``core_sg_`` object keeps lower-level fit-time artifacts
for advanced inspection.

After CoreSGClusterer.fit(...)
------------------------------

After ``CoreSGClusterer.fit(X, k=...)``:

``labels_``
   Cluster labels for the most recent extracted ``k``.

``probabilities_``
   HDBSCAN-style membership strengths.

``cluster_persistence_``
   Persistence values for selected clusters.

``condensed_tree_``
   Wrapped condensed tree for the current extraction.

``single_linkage_tree_``
   Wrapped single linkage tree for the current extraction.

``minimum_spanning_tree_``
   Wrapped minimum spanning tree for the current extraction.

``k_``
   The current extracted ``k``.

``k_max_``
   The validated reusable capacity.

``core_sg_``
   Internal reusable Core-SG object. Most users do not need to call it
   directly.

Advanced internal artifacts
---------------------------

The internal ``core_sg_`` object stores:

``support_graph_``
   Reusable Core-SG support graph.

``metric_edges_``
   Original metric distances for support edges.

``core_distances_``
   Matrix of core-distance candidates. ``get_core_distance(k)`` selects column
   ``k - 1``.

``distance_matrix_``
   Dense pairwise distance matrix for ``algorithm="core-sg"`` only.

``anti_hubs_``
   Selected anti-hub indices for ``algorithm="score-sg"`` only.

``labels_k_max_``
   Labels saved for the reference fit at ``k_max``.

``probabilities_k_max_``
   Probabilities saved for the reference fit at ``k_max``.

``cluster_persistence_k_max_``
   Cluster persistence saved for the reference fit at ``k_max``.

``condensed_tree_k_max_``
   Wrapped condensed tree saved at ``k_max``.

``single_linkage_tree_k_max_``
   Wrapped single linkage tree saved at ``k_max``.

``minimum_spanning_tree_k_max_``
   Wrapped minimum spanning tree saved at ``k_max``.
