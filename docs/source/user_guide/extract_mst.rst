Extracting MSTs
===============

The estimator exposes the current HDBSCAN-style MST wrapper after
``fit(X, k=...)``.

.. code-block:: python

   clusterer.fit(X, k=10)
   mst = clusterer.minimum_spanning_tree_

For advanced inspection, the internal object can also return a simple
DataFrame:

.. code-block:: python

   mst_df = clusterer.core_sg_.extract_mst_from_core_sg(k=10, toDF=True)

The DataFrame has three columns:

``to``
   First endpoint.

``from``
   Second endpoint.

``weight``
   Mutual-reachability edge weight used in the extracted MST.

With ``toDF=True``, the internal Core-SG object returns typed columns named
``to``, ``from``, and ``weight``.
