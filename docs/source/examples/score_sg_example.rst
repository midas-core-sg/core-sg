Score-SG Example
================

.. code-block:: python

   from core_sg import CoreSGClusterer

   clusterer = CoreSGClusterer(
       k_max=15,
       algorithm="score-sg",
       metric="euclidean",
       random_state=42,
   )
   clusterer.fit(X, k=10)

   anti_hubs = clusterer.core_sg_.anti_hubs_

If the Score-SG support graph is disconnected, Core-SG raises an explicit
error because MST extraction is not possible.
