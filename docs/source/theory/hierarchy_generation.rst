Hierarchy Generation
====================

Core-SG converts the extracted MST into a single linkage tree and then uses the
HDBSCAN tree post-processing path to produce:

* labels;
* probabilities;
* cluster persistence;
* condensed tree artifacts;
* single linkage tree artifacts;
* minimum spanning tree artifacts.

The estimator method ``CoreSGClusterer.fit(X, k=...)`` updates the current
attributes in place. Internally, the estimator uses the reusable Core-SG object
after the first fit.
