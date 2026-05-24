Cython Integration
==================

Core-SG declares optional Cython extension modules for performance-sensitive
Kruskal MST extraction and reweighting.

The Python modules include fallback implementations. If the compiled extension
is unavailable, Core-SG emits a runtime warning and uses the Python fallback.

Packaging metadata for the extensions lives in ``pyproject.toml``. Contributors
touching Cython code should validate both source behavior and built wheel
behavior where possible.
