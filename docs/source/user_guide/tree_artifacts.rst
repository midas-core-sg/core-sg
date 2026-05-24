Tree Artifacts
==============

Core-SG exposes HDBSCAN-style tree wrappers for inspection:

``minimum_spanning_tree_``
   MST wrapper for the current extracted hierarchy.

``single_linkage_tree_``
   Single linkage tree wrapper.

``condensed_tree_``
   Condensed cluster tree wrapper.

Most HDBSCAN tree wrappers support pandas conversion:

.. code-block:: python

   condensed = core.condensed_tree_.to_pandas()
   single_linkage = core.single_linkage_tree_.to_pandas()
   mst = core.minimum_spanning_tree_.to_pandas()

Fit-time versions are available as ``condensed_tree_k_max_``,
``single_linkage_tree_k_max_``, and ``minimum_spanning_tree_k_max_``.
