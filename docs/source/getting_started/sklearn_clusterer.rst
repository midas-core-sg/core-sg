Scikit-Learn-Style Clusterer
============================

``CoreSGClusterer`` is the recommended Core-SG interface. It provides
constructor parameters, ``fit(...)``, ``fit_predict(...)``, ``get_params()``,
``set_params()``, and compatibility with ``sklearn.base.clone(...)``.

.. code-block:: python

   from sklearn.datasets import make_blobs
   from core_sg import CoreSGClusterer

   X, _ = make_blobs(
       n_samples=1000,
       n_features=10,
       centers=5,
       random_state=42,
   )

   clusterer = CoreSGClusterer(k_max=15)
   clusterer.fit(X, k=10)

   labels_10 = clusterer.labels_

   clusterer.fit(X, k=8)
   labels_8 = clusterer.labels_

Lifecycle
---------

``k_max`` is a constructor parameter because it defines the reusable support
graph capacity.

``k`` is a keyword-only ``fit(...)`` parameter because it defines the current
extraction exposed through ``labels_`` and the other fitted attributes.

On the first ``fit(...)`` call, ``CoreSGClusterer`` creates the internal
``core_sg_`` object and builds the reusable support graph for ``k_max``.

On later ``fit(...)`` calls, it reuses the existing ``core_sg_`` object and
extracts the hierarchy for the requested ``k``.

The reuse decision is based on whether ``core_sg_`` exists, not on whether
``k == k_max``.

Current Limits
--------------

``CoreSGClusterer`` creates ``core_sg_`` only once. To rebuild with a different
dataset or a different ``k_max``, create a new ``CoreSGClusterer`` instance. A
dedicated reset/refit lifecycle API may be added later.

``predict(...)`` is intentionally not implemented. Core-SG is currently a
fit/extract clustering workflow and does not define assignment semantics for
unseen samples.
