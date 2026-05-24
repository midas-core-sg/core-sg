Testing Strategy
================

Run the unit suite for fast feedback:

.. code-block:: bash

   pytest tests/unit -q

Run broader checks before release:

.. code-block:: bash

   pytest tests -v -ra
   ruff check core_sg tests benchmarking
   sphinx-build -W -b html docs/source docs/build/html

Test Categories
---------------

Unit tests isolate behavior and use controlled doubles where useful, including
fake HDBSCAN modules for adapter behavior.

Integration tests exercise real package boundaries and real dependency
behavior.

Validation tests compare Core-SG behavior against reference expectations across
multiple ``k`` values.

Required Estimator Note
-----------------------

``tests/unit/test_estimators.py`` verifies that ``CoreSGClusterer`` creates
``CoreSG`` only once and reuses it on subsequent ``fit(...)`` calls.
