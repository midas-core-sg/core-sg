HDBSCAN-Compatible Parameters
=============================

Core-SG forwards a supported subset of HDBSCAN-style tree selection parameters
to the hierarchy conversion path.

Common parameters exposed by ``CoreSGClusterer`` include:

* ``cluster_selection_method``;
* ``allow_single_cluster``;
* ``match_reference_implementation``;
* ``cluster_selection_epsilon``;
* ``cluster_selection_persistence``;
* ``max_cluster_size``;
* ``cluster_selection_epsilon_max``.

Compatibility Note
------------------

Core-SG depends on HDBSCAN private APIs for hierarchy reconstruction. The exact
accepted private signature may vary by HDBSCAN version.

Core-SG filters unsupported ``_tree_to_labels(...)`` keyword arguments at the
adapter boundary. This avoids version-specific errors such as:

.. code-block:: text

   TypeError: _tree_to_labels() got an unexpected keyword argument 'cluster_selection_persistence'

The adapter is the compatibility boundary. User-facing code should configure
Core-SG parameters normally and let the adapter filter private kwargs.
