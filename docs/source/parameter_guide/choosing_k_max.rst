Choosing k_max
==============

``k_max`` defines the reusable support graph capacity. It must be large enough
for the largest ``k`` that will be extracted later.

.. code-block:: python

   clusterer = CoreSGClusterer(k_max=30)

Practical Guidance
------------------

Choose ``k_max`` as the largest neighborhood value in your planned analysis.
For example, if you want to compare ``k`` values ``30``, ``20``, and ``10``,
fit with ``k_max=30``.

A larger ``k_max`` can increase build cost because the reusable support graph
must cover more neighborhood information. A smaller ``k_max`` limits what can
be extracted later.

Repeated multi-``k`` workflows are where Core-SG is most useful. If you only
need one value, plain HDBSCAN may be simpler.
