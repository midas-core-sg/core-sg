CoreSGClusterer Workflow
========================

``CoreSGClusterer`` gives the same reuse behavior through an estimator-shaped
API:

.. code-block:: python

   from core_sg import CoreSGClusterer

   clusterer = CoreSGClusterer(k_max=30)

   for k in [25, 20, 15, 10]:
       clusterer.fit(X, k=k)
       labels = clusterer.labels_

.. raw:: html

   <div class="docs-diagram">
     <div class="docs-diagram-title">CoreSGClusterer lifecycle</div>
     <div class="diagram-column">
       <div class="diagram-step accent">User calls fit(X, k=25)</div>
       <div class="diagram-arrow down">↓</div>
       <div class="diagram-step secondary">Estimator creates core_sg_ and builds support at k_max=30</div>
       <div class="diagram-arrow down">↓</div>
       <div class="diagram-step">Estimator exposes attributes for k=25</div>
       <div class="diagram-arrow down">↓</div>
       <div class="diagram-step accent">User calls fit(X, k=20)</div>
       <div class="diagram-arrow down">↓</div>
       <div class="diagram-step secondary">Estimator reuses the existing core_sg_ object</div>
       <div class="diagram-arrow down">↓</div>
       <div class="diagram-step">Estimator exposes updated attributes for k=20</div>
     </div>
     <div class="diagram-note">The rebuild decision is based on whether core_sg_ exists, not on whether k equals k_max.</div>
   </div>

The first call builds ``core_sg_``. Later calls reuse the same object and only
extract a new hierarchy.

``labels_`` updates after each call and corresponds to the most recent ``k``.
The internal reusable object remains available through ``clusterer.core_sg_``
or ``clusterer.get_fitted_core_sg()`` for advanced inspection.

``fit_predict(...)`` is available:

.. code-block:: python

   labels = clusterer.fit_predict(X, k=10)

``predict(...)`` is not implemented because unseen-sample assignment semantics
are not part of the current Core-SG API.
