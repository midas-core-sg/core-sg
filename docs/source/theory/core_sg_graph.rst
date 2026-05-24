Core-SG Graph
=============

The support graph stores enough edge information to support later MST
extraction for valid target ``k`` values. In the exact path, Core-SG builds a
k-nearest-neighbor graph from the dense pairwise distance matrix and augments
the metric edge set with reference MST edges from the ``k_max`` fit.

The graph is stored as arrays with edge endpoints and weights. ``metric_edges_``
keeps original metric distances, while ``support_graph_`` is reweighted later
for a target ``k``.

This separation lets Core-SG reuse the same structural support while changing
the mutual-reachability weights for each extraction.
