Repeated Multi-k Workloads
==========================

The intended workload is:

.. code-block:: python

   clusterer = CoreSGClusterer(k_max=50)

   for k in range(50, 1, -1):
       clusterer.fit(X, k=k)

For ``CoreSGClusterer``, repeated calls to ``fit(X, k=...)`` after the first
call reuse ``core_sg_`` and should be interpreted as extraction cost, not a
full rebuild.

This is the workload where Core-SG is expected to differ most from running
HDBSCAN independently for each ``k``.
