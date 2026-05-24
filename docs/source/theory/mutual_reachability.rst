Mutual Reachability
===================

Mutual reachability combines the original metric distance between two points
with their core distances.

For an edge ``(u, v)`` at target ``k``, the weight is:

.. math::

   max(core_k(u), core_k(v), d(u, v))

This weight is what Core-SG uses to extract the target MST. Changing ``k``
changes ``core_k`` and therefore changes the reweighted support graph.
