Noise Handling
==============

Core-SG can optionally post-process labels assigned to ``-1`` after hierarchy
extraction.

Parameters:

``no_noise``
   When ``True``, enables label reassignment after extraction.

``noise_label_strategy``
   Selects the reassignment strategy. The current implemented strategy is
   ``"mst_label_propagation"``.

``c``
   Advanced internal extraction parameter. It controls the top-``c`` path
   signature used by the current MST label propagation strategy.

Important Behavior
------------------

Noise reassignment updates only ``labels_``. The extracted hierarchy artifacts
remain tied to the hierarchy produced before label reassignment.

Disable this behavior when you want raw HDBSCAN-style ``-1`` labels:

.. code-block:: python

   clusterer = CoreSGClusterer(k_max=30, no_noise=False)
