Release Checklist
=================

Before release:

* confirm ``pyproject.toml`` metadata and version;
* run ``pytest tests -v -ra``;
* run ``ruff check core_sg tests benchmarking``;
* build docs with ``sphinx-build -W -b html docs/source docs/build/html``;
* regenerate documentation figures;
* build distributions with ``python -m build``;
* validate distributions with ``python -m twine check dist/*``;
* confirm README links point to the published documentation;
* confirm GitHub Pages is configured to use GitHub Actions.
