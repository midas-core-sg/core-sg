Basic Estimator Usage
=====================

Create a ``CoreSGClusterer`` instance, fit it for a target ``k``, and read the
current HDBSCAN-style outputs.

.. code-block:: python

   from core_sg import CoreSGClusterer

   clusterer = CoreSGClusterer(k_max=30, metric="euclidean", p=2)
   clusterer.fit(X, k=15)

   labels = clusterer.labels_
   probabilities = clusterer.probabilities_

The first ``fit(...)`` call builds the internal reusable graph support. Later
``fit(X, k=...)`` calls reuse that support and update the current outputs in
place.

The value passed to ``k`` must satisfy ``2 <= k <= k_max``.
