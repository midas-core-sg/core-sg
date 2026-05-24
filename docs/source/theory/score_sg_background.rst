Score-SG Background
===================

``algorithm="score-sg"`` is the approximate Core-SG construction path. It is
intended for experiments where dense all-pairs distance construction is too
expensive or where the anti-hub reinforced support graph is part of the method
being evaluated.

This makes Score-SG the scalable option when the traditional exact CoreSG path
is limited by ``n_samples``. Exact CoreSG remains valuable for reference-style
construction, but its dense pairwise matrix becomes increasingly restrictive as
sample size grows.

.. raw:: html

   <div class="docs-diagram">
     <div class="docs-diagram-title">Score-SG background flow</div>
     <div class="diagram-column">
       <div class="diagram-step accent">Input data X</div>
       <div class="diagram-arrow down">↓</div>
       <div class="diagram-step secondary">PyNNDescent builds an approximate kNN graph</div>
       <div class="diagram-arrow down">↓</div>
       <div class="diagram-step secondary">Approximate neighbor distances provide core-distance candidates</div>
       <div class="diagram-arrow down">↓</div>
       <div class="diagram-step warning">Directed in-degree identifies anti-hub candidates</div>
       <div class="diagram-arrow down">↓</div>
       <div class="diagram-step warning">Exact-distance clique edges reinforce selected anti-hubs</div>
       <div class="diagram-arrow down">↓</div>
       <div class="diagram-step">The support graph is reweighted for the target k</div>
       <div class="diagram-arrow down">↓</div>
       <div class="diagram-step">MST extraction and HDBSCAN-style hierarchy generation expose estimator outputs</div>
     </div>
   </div>

What Changes Compared With The Default Path
-------------------------------------------

The default ``algorithm="core-sg"`` path builds exact pairwise distance
information before constructing reusable support. Score-SG avoids that dense
all-pairs stage. Instead, it uses PyNNDescent to choose approximate neighbors
and then computes exact distances only for selected support edges.

This means Score-SG changes graph construction, not the public estimator
workflow. Users still call:

.. code-block:: python

   clusterer = CoreSGClusterer(k_max=30, algorithm="score-sg")
   clusterer.fit(X, k=20)

Approximate Neighbor Graph
--------------------------

PyNNDescent proposes the neighbor graph. Those neighbors define which local
edges are eligible for the support graph. Because this step is approximate,
two choices matter more than in the default path:

* ``random_state`` for reproducible tie handling;
* ``approx_knn_kwargs`` for PyNNDescent tuning.

Anti-Hub Reinforcement
----------------------

Score-SG counts how often each point appears in directed neighbor lists.
Points with low directed in-degree are anti-hub candidates: they are rarely
chosen as neighbors by other points. The implementation selects anti-hubs,
connects them with exact-distance clique edges, and merges those edges into
the support graph.

Why this matters: the anti-hub edges are intended to strengthen sparse support
in regions that approximate neighbor search may underrepresent.

Shared Downstream Pipeline
--------------------------

Once the Score-SG support graph exists, the downstream steps are shared with
the default Core-SG path:

* reweight support edges for the requested ``k``;
* extract an MST;
* convert the MST into HDBSCAN-style hierarchy artifacts;
* expose estimator attributes such as ``labels_`` and
  ``minimum_spanning_tree_``.

The current implementation:

* avoids materializing a dense all-pairs distance matrix;
* uses approximate neighbors for graph membership;
* computes exact distances for selected support edges;
* selects anti-hubs using directed in-degree;
* can fail explicitly when the resulting support graph is disconnected.

The important caveat is connectivity. If the approximate support graph is
disconnected, Core-SG does not silently repair it. MST extraction is impossible
on a disconnected graph, so the estimator raises a clear error.
