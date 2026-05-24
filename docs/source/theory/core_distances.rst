Core Distances
==============

For each point and each usable neighborhood level, Core-SG stores core-distance
candidates in ``core_distances_``.

``get_core_distance(k)`` selects column ``k - 1``:

.. code-block:: python

   core_k = core.get_core_distance(k=10)

Core distances are part of the mutual-reachability formula and therefore
directly affect MST edge weights and hierarchy structure.

In ``algorithm="score-sg"``, these values come from approximate neighbor
distances returned by PyNNDescent.
