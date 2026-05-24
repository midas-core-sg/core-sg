Installation
============

Core-SG supports Python ``>=3.10``. Runtime dependencies are declared in
``pyproject.toml`` and include NumPy, pandas, scikit-learn, HDBSCAN, and
PyNNDescent.

Install From PyPI
-----------------

.. code-block:: bash

   pip install core-sg

Local Development Install
-------------------------

From the repository root:

.. code-block:: bash

   pip install -e .

Install test and contributor tooling:

.. code-block:: bash

   pip install -e ".[dev]"

Install documentation tooling:

.. code-block:: bash

   pip install -e ".[docs]"

Build the documentation locally:

.. code-block:: bash

   python docs/scripts/generate_diagrams.py
   python docs/scripts/generate_benchmark_figures.py
   sphinx-build -b html docs/source docs/build/html

Use the strict command before opening documentation pull requests:

.. code-block:: bash

   sphinx-build -W -b html docs/source docs/build/html

Dependency Notes
----------------

The ``docs`` extra includes Sphinx, the Read the Docs theme, autodoc helpers,
MyST, nbsphinx, IPython, matplotlib, pandas, and scikit-learn. It intentionally
keeps notebook execution disabled during ordinary documentation builds so the
site remains fast and deterministic.
