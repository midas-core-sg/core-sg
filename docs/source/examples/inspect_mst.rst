Inspect An MST
==============

.. code-block:: python

   from core_sg import CoreSGClusterer

   clusterer = CoreSGClusterer(k_max=15)
   clusterer.fit(X, k=10)

   mst_df = clusterer.core_sg_.extract_mst_from_core_sg(k=10, toDF=True)
   print(mst_df.head())

The returned DataFrame contains ``to``, ``from``, and ``weight`` columns.
