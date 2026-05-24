Repeated k Extraction
=====================

.. code-block:: python

   from core_sg import CoreSGClusterer

   clusterer = CoreSGClusterer(k_max=30)

   results = {}
   for k in [30, 25, 20, 15, 10]:
       clusterer.fit(X, k=k)
       results[k] = clusterer.labels_.copy()

Use this pattern to compare cluster stability across smoothing levels.
