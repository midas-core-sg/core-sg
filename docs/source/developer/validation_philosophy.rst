Validation Philosophy
=====================

Core-SG validation focuses on public behavior:

* reusable support graph construction;
* MST extraction for valid ``k`` values;
* hierarchy artifact generation;
* HDBSCAN-style wrapper exposure;
* clear failures for invalid parameters and disconnected graphs.

The estimator wrapper is tested for sklearn-style parameter introspection and
cloning, but full sklearn estimator checks are not currently required.

Reference-equivalence tests are valuable, but they must be interpreted with the
HDBSCAN adapter boundary in mind because Core-SG depends on private HDBSCAN
interfaces for some hierarchy behavior.
