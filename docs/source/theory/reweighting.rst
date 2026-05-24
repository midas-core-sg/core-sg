Reweighting
===========

Reweighting is the step that turns reusable support into a target-``k`` graph.

``get_core_sg_mutual_reachability_distance(k)`` returns the support graph with
mutual-reachability weights for a valid ``k``:

.. code-block:: python

   weighted = core.get_core_sg_mutual_reachability_distance(k=10)

The implementation reuses ``support_graph_``, ``metric_edges_``, and
``core_distances_``. It does not rebuild the original neighbor graph for each
target ``k``.
