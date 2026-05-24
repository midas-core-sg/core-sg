# GitHub Pages Publication And Maintenance

This repository publishes the documentation site from `docs/source` with
Sphinx and GitHub Actions.

## Local build

Install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

Generate figures and build the site:

```bash
python docs/scripts/generate_diagrams.py
python docs/scripts/generate_benchmark_figures.py
sphinx-build -b html docs/source docs/build/html
```

Use the strict build before merging documentation changes:

```bash
sphinx-build -W -b html docs/source docs/build/html
```

## GitHub Pages deployment

The workflow is `.github/workflows/docs.yml`.

It runs on pushes to `main` and `develop`, pull requests, and manual
`workflow_dispatch` runs. The `build-docs` job installs `.[docs]`, generates
figures, builds Sphinx with warnings treated as errors, and uploads the HTML
artifact. The `deploy` job publishes only when the ref is `refs/heads/main`.

Repository settings must use:

```text
Settings -> Pages -> Source: GitHub Actions
```

## Maintenance checklist

- Keep `docs/source/index.rst` and section toctrees in sync with new pages.
- Run `python docs/scripts/generate_diagrams.py` after changing theory diagrams.
- Run `python docs/scripts/generate_benchmark_figures.py` after changing benchmark assets or CSVs.
- Keep API pages aligned with `core_sg/__init__.py`, `core_sg/core_sg.py`, and `core_sg/estimators.py`.
- Document lifecycle changes to `CoreSGClusterer` in Getting Started, User Guide, API, and FAQ.
- Keep HDBSCAN private API notes in `theory/hdbscan_integration.rst`, `parameter_guide/hdbscan_parameters.rst`, FAQ, and developer docs aligned.
- Prefer `sphinx-build -W` for release and PR validation.

## Optional link check

External links can be checked manually:

```bash
sphinx-build -b linkcheck docs/source docs/build/linkcheck
```

Avoid blocking every pull request on external link availability unless the
project decides to make that a release requirement.
