Estimator API
=============

.. autoclass:: core_sg.CoreSGClusterer
   :members:
   :exclude-members: set_fit_request
   :member-order: bysource

Required Behavior Notes
-----------------------

``CoreSGClusterer`` does not implement ``predict(...)``. Core-SG is currently
fit/extract oriented and does not define assignment semantics for unseen
samples.

The first ``fit(...)`` call creates ``core_sg_`` and builds reusable support
for ``k_max``. Later calls reuse ``core_sg_`` and extract the hierarchy for the
requested ``k``. The decision is based on whether ``core_sg_`` exists, not on
whether ``k == k_max``.

The estimator is compatible with ``get_params()``, ``set_params()``, and
``sklearn.base.clone(...)`` through scikit-learn's ``BaseEstimator``.
