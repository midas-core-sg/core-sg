Troubleshooting
===============

When should I use Core-SG instead of HDBSCAN?
---------------------------------------------

Use Core-SG when you need repeated ``k`` analysis on the same dataset or want
to inspect MST and hierarchy artifacts. Use plain HDBSCAN when you only need
one clustering run and a simpler API is more important than reuse.

When should I use CoreSGClusterer?
----------------------------------

Use ``CoreSGClusterer`` for ordinary workflows. It provides the
scikit-learn-style ``fit(...)`` and ``fit_predict(...)`` interface and manages
the lower-level reusable Core-SG object internally.

What is k_max?
--------------

``k_max`` is the largest neighborhood value used to build reusable support. It
must be large enough for the largest later extraction.

What is k?
----------

``k`` is the current extraction value. It must satisfy ``2 <= k <= k_max``.

Does CoreSGClusterer.fit(...) rebuild the graph every time?
-----------------------------------------------------------

No. The first ``fit(...)`` call creates ``core_sg_`` and builds the reusable
support graph for ``k_max``. Later ``fit(...)`` calls reuse ``core_sg_`` and
only extract the hierarchy for the requested ``k``. The decision is based on
whether ``core_sg_`` exists, not on whether ``k == k_max``.

How do I rebuild with another dataset or another k_max?
-------------------------------------------------------

Create a new ``CoreSGClusterer`` instance. A dedicated reset/refit lifecycle
may be added in the future.

Why does ``labels_`` change after extracting a different k?
---------------------------------------------------------------

``labels_`` always reflects the most recent extracted hierarchy. Save a copy if
you need to keep labels for multiple ``k`` values.

What is the difference between ``labels_`` and ``labels_k_max_``?
-----------------------------------------------------------------

``labels_k_max_`` is saved from the fit-time reference value. ``labels_`` is
the current extraction output.

What is ``core_sg_``?
---------------------

``core_sg_`` is the fitted internal reusable Core-SG object owned by
``CoreSGClusterer``. Most users do not need to call it directly.

Why do my results differ from HDBSCAN?
--------------------------------------

Core-SG reconstructs hierarchy outputs from its reusable support graph and
adapter boundary. Differences can also come from HDBSCAN version, tree
selection parameters, metric choices, and optional noise reassignment.

Why is Score-SG disconnected for my data?
-----------------------------------------

Score-SG builds an approximate sparse support graph. Some data and parameter
settings may not produce a connected graph. In that case, MST extraction is not
possible and Core-SG raises a clear error.

What artifacts are available after fit?
---------------------------------------

After ``CoreSGClusterer.fit(...)``, the estimator exposes ``labels_``,
``probabilities_``, ``cluster_persistence_``, ``condensed_tree_``,
``single_linkage_tree_``, ``minimum_spanning_tree_``, ``k_``, ``k_max_``, and
``core_sg_``.

What artifacts are available after hierarchy extraction?
--------------------------------------------------------

After ``fit(X, k=...)``, use ``labels_``, ``probabilities_``,
``cluster_persistence_``, ``condensed_tree_``, ``single_linkage_tree_``, and
``minimum_spanning_tree_``.

How should I interpret ``minimum_spanning_tree_``?
--------------------------------------------------

It is the HDBSCAN-style MST wrapper for the current extracted ``k``. Advanced
users can access ``clusterer.core_sg_`` when they need lower-level MST arrays
or DataFrames.

Why is performance different from the benchmark?
------------------------------------------------

Performance depends on data geometry, sample size, metric, Cython extension
availability, HDBSCAN version, PyNNDescent settings, and whether you measure a
single run or a repeated multi-``k`` workflow.

Why does PyNNDescent have warm-up behavior?
-------------------------------------------

PyNNDescent may pay setup and compilation-like costs on early runs. Prefer
repeated measurements when benchmarking Score-SG or approximate neighbor
workflows.

What does it mean that Core-SG uses HDBSCAN internals?
------------------------------------------------------

Core-SG uses private HDBSCAN functions to produce familiar hierarchy outputs.
Those calls are isolated in ``core_sg/hdbscan_adapter.py``.

Why did I see ``_tree_to_labels()`` got an unexpected keyword argument?
-----------------------------------------------------------------------

Problem:

.. code-block:: text

   TypeError: _tree_to_labels() got an unexpected keyword argument 'cluster_selection_persistence'

This happens when the installed HDBSCAN version exposes a private
``_tree_to_labels(...)`` function whose signature does not accept a parameter
Core-SG knows about. Core-SG filters forwarded keyword arguments at the
``hdbscan_adapter`` boundary according to the installed signature.

Upgrade Core-SG to a version that includes adapter-side filtering. If
developing locally, rerun:

.. code-block:: bash

   pytest tests/unit/test_hdbscan_adapter.py
