CoreSGClusterer Example
=======================

.. code-block:: python

   from sklearn.base import clone
   from core_sg import CoreSGClusterer

   clusterer = CoreSGClusterer(k_max=20)
   clusterer.fit(X, k=15)

   labels_15 = clusterer.labels_

   clusterer.fit(X, k=10)
   labels_10 = clusterer.labels_

   fresh = clone(clusterer)

The cloned estimator has the same constructor parameters but no fitted
``core_sg_`` object.
