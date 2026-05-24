Score-SG
========

Use ``algorithm="score-sg"`` to select the approximate anti-hub reinforced
graph-construction path.

Score-SG is the scalable path to use when the traditional exact
``algorithm="core-sg"`` construction becomes limited by ``n_samples``. The
exact path builds dense pairwise distance information; Score-SG avoids that
dense all-pairs stage by using an approximate sparse-neighbor graph.

.. code-block:: python

   from core_sg import CoreSGClusterer

   clusterer = CoreSGClusterer(
       k_max=15,
       algorithm="score-sg",
       metric="euclidean",
       random_state=42,
       approx_knn_kwargs={"n_trees": 8},
   )
   clusterer.fit(X, k=10)

.. raw:: html

   <div class="docs-diagram">
     <div class="docs-diagram-title">Score-SG construction path</div>
     <div class="diagram-flow">
       <div class="diagram-step accent">X</div>
       <div class="diagram-arrow">→</div>
       <div class="diagram-step secondary">PyNNDescent</div>
       <div class="diagram-arrow">→</div>
       <div class="diagram-step secondary">approximate kNN graph</div>
       <div class="diagram-arrow">→</div>
       <div class="diagram-step warning">anti-hub support</div>
       <div class="diagram-arrow">→</div>
       <div class="diagram-step">MST and hierarchy</div>
       <div class="diagram-arrow">→</div>
       <div class="diagram-step">estimator outputs</div>
     </div>
   </div>

Score-SG uses PyNNDescent to build an approximate nearest-neighbor graph,
derives approximate core-distance lists, selects anti-hubs by directed
in-degree, adds support edges among selected anti-hubs, and then reuses the
same MST and hierarchy machinery as the classical path.

In the current benchmark pages, Score-SG is also the best-performing method in
the common repeated multi-``k`` comparison against exact CoreSG and HDBSCAN.
See :doc:`../performance/score_sg_results` for the measured speedups and
extended scaling results.

Important Behaviors
-------------------

``random_state`` controls random tie-breaking during anti-hub selection.

``approx_knn_kwargs`` is forwarded to PyNNDescent.

``anti_hubs_`` is available only when ``algorithm="score-sg"``.

Score-SG may fail explicitly if the constructed support graph is disconnected,
because MST extraction requires a connected support graph.
