HDBSCAN Integration
===================

Core-SG uses HDBSCAN-compatible internals to expose familiar hierarchy outputs.
The direct dependency on HDBSCAN private APIs is isolated in
``core_sg/hdbscan_adapter.py``.

The adapter owns:

* reference MST construction at ``k_max`` for the exact path;
* MST-to-single-linkage conversion;
* tree-to-label conversion;
* HDBSCAN-style wrapper construction for tree artifacts;
* private keyword filtering based on the installed HDBSCAN signature.

Because these HDBSCAN functions are private, compatibility risk is localized at
the adapter boundary.
