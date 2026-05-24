Synthetic Walkthrough
=====================

.. code-block:: python

   from sklearn.datasets import make_blobs
   from core_sg import CoreSGClusterer

   X, _ = make_blobs(n_samples=1000, n_features=10, centers=5, random_state=42)

   clusterer = CoreSGClusterer(k_max=20, metric="euclidean", p=2)
   clusterer.fit(X, k=10)

   print(clusterer.labels_)

This example builds one reusable internal support graph and extracts one
hierarchy through the estimator.
