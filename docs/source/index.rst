Core-SG
=======

Core-SG is a graph-based library for repeated HDBSCAN-style clustering over
multiple neighborhood values. The recommended public workflow is
:class:`core_sg.CoreSGClusterer`, a scikit-learn-style estimator that builds a
reusable support graph at ``k_max`` and extracts hierarchy artifacts for any
valid ``k <= k_max`` without rebuilding the full structure each time.

The project is aimed at exploratory density-based analysis where comparing
several smoothing levels on the same dataset is part of the workflow. A single
plain HDBSCAN run may be simpler when only one ``k`` value is needed.

For larger datasets, use the algorithm choice deliberately. The exact
``algorithm="core-sg"`` path uses dense pairwise distance information and can
be limited by ``n_samples``. The approximate ``algorithm="score-sg"`` path is
the scalable alternative for larger repeated multi-``k`` workloads.

Key Features
------------

* use a scikit-learn-style estimator with ``fit(...)`` and ``fit_predict(...)``;
* build once at ``k_max`` and extract many smaller ``k`` values;
* inspect MSTs, condensed trees, single linkage trees, labels, probabilities,
  and persistence values;
* run the classical exact ``algorithm="core-sg"`` path or the approximate
  ``algorithm="score-sg"`` path;
* keep HDBSCAN-specific internals behind a documented adapter boundary.

Recommended Estimator API
-------------------------

:class:`core_sg.CoreSGClusterer` is the user-facing API. It provides
constructor parameters, ``fit(...)``, ``fit_predict(...)``, ``get_params``,
``set_params``, and clone compatibility. The lower-level reusable object is
created behind the estimator and exposed as ``core_sg_`` for advanced
inspection.

Quick Example
-------------

.. code-block:: python

   from sklearn.datasets import make_blobs
   from core_sg import CoreSGClusterer

   X, _ = make_blobs(n_samples=1000, centers=5, random_state=42)

   clusterer = CoreSGClusterer(k_max=15)
   clusterer.fit(X, k=10)

   labels = clusterer.labels_

.. note::

   Calling ``clusterer.fit(X, k=8)`` later reuses the same internal ``core_sg_``
   object and extracts the hierarchy for ``k=8``. The reuse decision is based
   on whether ``core_sg_`` exists, not on whether ``k == k_max``.

When To Use Core-SG
-------------------

Core-SG is useful when you need a sklearn-style estimator that can compare many
``k`` values on the same data, inspect graph-level artifacts, or reproduce
HDBSCAN-style hierarchy outputs from a reusable support graph.

When Not To Use Core-SG
-----------------------

Plain HDBSCAN may be a better first choice when you only need one clustering
result, do not need MST or hierarchy inspection, or want the smallest possible
API surface.

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   getting_started/index
   user_guide/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Reference Guides

   parameter_guide/index
   theory/index
   performance/index
   api/index
   references

.. toctree::
   :maxdepth: 2
   :caption: Support And Development

   faq/index
   developer/index
