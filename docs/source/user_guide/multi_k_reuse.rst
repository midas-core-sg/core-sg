Reusing One Estimator Across Multiple k Values
==============================================

The estimator workflow is designed around reuse:

.. code-block:: python

   clusterer = CoreSGClusterer(k_max=30)

   for k in [25, 20, 15, 10]:
       clusterer.fit(X, k=k)
       labels = clusterer.labels_

.. raw:: html

   <div class="docs-diagram">
     <div class="docs-diagram-title">Build once, extract several k values</div>
     <div class="diagram-column">
       <div class="diagram-step accent">CoreSGClusterer(k_max=30)</div>
       <div class="diagram-arrow down">↓</div>
       <div class="diagram-step secondary">fit(X, k=25): build core_sg_ and expose labels_ for k=25</div>
       <div class="diagram-arrow down">↓</div>
       <div class="diagram-step secondary">fit(X, k=20): reuse core_sg_ and expose labels_ for k=20</div>
       <div class="diagram-arrow down">↓</div>
       <div class="diagram-step secondary">fit(X, k=15): reuse core_sg_ and expose labels_ for k=15</div>
       <div class="diagram-arrow down">↓</div>
       <div class="diagram-step secondary">fit(X, k=10): reuse core_sg_ and expose labels_ for k=10</div>
     </div>
     <div class="diagram-note">Only the first fit builds the reusable support graph. Later fits update the estimator outputs for the requested k.</div>
   </div>

Interpretation
--------------

``k_max`` defines the largest neighborhood value available from this estimator.
The first ``fit(...)`` builds the internal support graph. Every later ``k``
extraction reuses the same ``core_sg_`` object and recomputes the current MST
and hierarchy outputs for that target value.

The attributes ``labels_``, ``probabilities_``, ``cluster_persistence_``, and
the current tree artifacts always refer to the most recent extraction.

Use ``clusterer.core_sg_`` only for advanced inspection of fit-time artifacts.
