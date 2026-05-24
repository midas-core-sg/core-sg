First Example With CoreSGClusterer
==================================

The recommended user-facing API is ``CoreSGClusterer``. It follows a
scikit-learn-style workflow:

1. configure reusable capacity with ``k_max`` in the constructor;
2. call ``fit(X, k=...)`` for the current extraction.

.. code-block:: python

   from sklearn.datasets import make_blobs
   from core_sg import CoreSGClusterer

   X, _ = make_blobs(
       n_samples=1000,
       n_features=10,
       centers=5,
       random_state=42,
   )

   clusterer = CoreSGClusterer(k_max=15, metric="euclidean", p=2)
   clusterer.fit(X, k=10)

   labels = clusterer.labels_

What Each Step Does
-------------------

``CoreSGClusterer(k_max=15, metric="euclidean", p=2)`` configures the
estimator and the reusable support graph capacity.

``clusterer.fit(X, k=10)`` builds the internal reusable Core-SG object on the
first call and exposes labels and hierarchy artifacts for ``k=10``.

Later calls such as ``clusterer.fit(X, k=8)`` reuse ``clusterer.core_sg_`` and
update ``labels_``, ``probabilities_``, ``cluster_persistence_``, and tree
artifacts for the new ``k``.
