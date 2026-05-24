Verbose Output And Progress Callbacks
=====================================

``verbose`` prints timed progress messages for major stages when greater than
zero:

.. code-block:: python

   clusterer = CoreSGClusterer(k_max=30, verbose=1)

``progress_callback`` receives structured timing events:

.. code-block:: python

   def callback(event, elapsed, info):
       print(event, elapsed, info)

   clusterer = CoreSGClusterer(k_max=30, progress_callback=callback)

The callback signature is:

.. code-block:: python

   progress_callback(event: str, elapsed: float, info: dict)

Events may include build, reweighting, Kruskal MST extraction, and hierarchy
extraction stages.
