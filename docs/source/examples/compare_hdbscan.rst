Compare With HDBSCAN
====================

Core-SG is most meaningful when compared against repeated HDBSCAN runs for
several ``k`` values.

.. code-block:: python

   from hdbscan import HDBSCAN
   from core_sg import CoreSGClusterer

   clusterer = CoreSGClusterer(k_max=30)
   clusterer.fit(X, k=10)

   hdb = HDBSCAN(min_cluster_size=10, min_samples=10)
   hdb.fit(X)

   core_labels = clusterer.labels_
   hdbscan_labels = hdb.labels_

Small differences can occur because Core-SG reconstructs hierarchy artifacts
through its support graph and HDBSCAN adapter boundary.
