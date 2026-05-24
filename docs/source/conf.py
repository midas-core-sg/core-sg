from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../.."))

project = "Core-SG"
author = "Midas Core-SG Team"
copyright = f"{datetime.now().year}, {author}"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

for optional_extension in [
    "sphinx_autodoc_typehints",
    "myst_parser",
    "nbsphinx",
]:
    try:
        __import__(optional_extension)
    except ImportError:
        continue
    extensions.append(optional_extension)

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "api/generated/*"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_mock_imports = []
for dependency in [
    "numpy",
    "pandas",
    "sklearn",
    "hdbscan",
    "pynndescent",
]:
    try:
        __import__(dependency)
    except Exception:
        autodoc_mock_imports.append(dependency)
napoleon_google_docstring = False
napoleon_numpy_docstring = True


def _skip_sklearn_metadata_request_methods(
    app,
    what,
    name,
    obj,
    skip,
    options,
):
    """Do not render scikit-learn metadata-routing request helpers.

    Newer scikit-learn versions dynamically add methods such as
    ``set_fit_request``. Their upstream docstrings contain cross-references
    that are valid in scikit-learn's documentation, but unresolved in this
    project. They are inherited compatibility helpers rather than Core-SG API
    surface, so hiding them keeps strict documentation builds stable across
    scikit-learn and Sphinx releases.
    """
    del app, what, obj, options
    if name.startswith("set_") and name.endswith("_request"):
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", _skip_sklearn_metadata_request_methods)


try:
    __import__("sphinx_rtd_theme")
except ImportError:
    html_theme = "alabaster"
else:
    html_theme = "sphinx_rtd_theme"
html_title = "Core-SG documentation"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

nbsphinx_execute = "never"
