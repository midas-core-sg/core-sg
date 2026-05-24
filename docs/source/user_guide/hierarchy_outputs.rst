Working With Hierarchy Outputs
==============================

``CoreSGClusterer.fit(X, k=...)`` computes the current extraction and exposes
HDBSCAN-style outputs on the estimator.

The method updates:

* ``labels_``;
* ``probabilities_``;
* ``cluster_persistence_``;
* ``condensed_tree_``;
* ``single_linkage_tree_``;
* ``minimum_spanning_tree_``.

Current Artifacts vs k_max Artifacts
------------------------------------

Current artifacts refer to the most recent extracted ``k``. Fit-time artifacts
ending in ``_k_max_`` refer to the original reference value used during
``fit(...)``.

Internally, ``fit(X, k=...)`` maps onto hierarchy extraction in the reusable
Core-SG object. The estimator copies the current outputs onto itself after each
fit call.

Noise Handling
--------------

When ``no_noise=True``, Core-SG may post-process ``labels_`` with the selected
``noise_label_strategy``. This affects only ``labels_``. The hierarchy
artifacts remain tied to the extracted hierarchy.
