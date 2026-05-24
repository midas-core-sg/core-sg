# Contributing to core-sg

Thank you for your interest in contributing to `core-sg`.

This guide is the repository-specific reference for local setup, development
workflow, testing, and release practices.

## Local contributor quickstart

If you want the shortest working path from a fresh clone to running the local
tests, use the sequence below:

```bash
git clone git@github.com:<your-user>/core-sg.git
cd core-sg
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
python -c "from core_sg import CoreSG; print(CoreSG.__name__)"
pytest tests -v -ra
```

This is the canonical local contributor flow for:

- creating an isolated environment
- installing the package in editable mode
- installing the development dependencies required for local testing
- validating that the package imports correctly
- running the test suite locally

## Getting started

The recommended contributor path is:

1. fork the repository
2. clone your fork locally
3. create and activate a virtual environment
4. install `core-sg` in editable mode with development dependencies
5. run the local test and quality checks
6. open a pull request against `develop`

## Prerequisites

Before you begin, make sure you have:

- Python `>=3.10`
- Git
- `pip`
- a working C/C++ build toolchain appropriate for your platform

Using a virtual environment is strongly recommended for all local development.

## Local development setup

Start from a fresh clone of your fork:

```bash
git clone git@github.com:<your-user>/core-sg.git
cd core-sg
git remote add upstream git@github.com:midas-core-sg/core-sg.git
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Upgrade the basic packaging tools inside the virtual environment:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install the project in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

This is the recommended contributor install path because it provides:

- the package itself in editable mode
- test tooling such as `pytest` and `pytest-cov`
- linting and formatting via `ruff`
- build and package validation tools such as `build` and `twine`

If you only need a local editable runtime install without contributor tooling,
you can use:

```bash
pip install -e .
```

## Validate the installation

After installation, confirm that the package imports correctly:

```bash
python -c "from core_sg import CoreSG; print(CoreSG.__name__)"
```

If this import works, the local editable install is in place.

## Run tests and checks locally

After `pip install -e ".[dev]"`, the standard local test command is:

```bash
pytest tests -v -ra
```

If you want the exact contributor flow as a single sequence, run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
pytest tests -v -ra
```

Run the standard test suite:

```bash
pytest tests -v -ra
```

Run the test suite with coverage:

```bash
pytest tests -v -ra --cov=core_sg --cov-report=term-missing
```

Run lint checks:

```bash
ruff check core_sg tests
```

Run formatting checks:

```bash
ruff format --check core_sg tests
```

Build the package locally:

```bash
python -m build
```

Validate the built distribution metadata:

```bash
python -m twine check dist/*
```

Recommended order before opening a pull request:

1. `pytest tests -v -ra`
2. `ruff check core_sg tests`
3. `ruff format --check core_sg tests`
4. `pytest tests -v -ra --cov=core_sg --cov-report=term-missing`
5. `python -m build`
6. `python -m twine check dist/*`

For most contributions, the first three checks are the minimum expected local
validation steps.

## Troubleshooting local setup

If the contributor setup does not work from a fresh environment, check the
following first:

- confirm that the virtual environment is activated
- confirm that you ran `pip install -e ".[dev]"` from the repository root
- confirm that your Python version is `3.10` or newer
- if build dependencies fail during installation, verify that your local system
  has the compiler toolchain needed to build Python extension modules

If you still cannot complete the setup, please open an issue and include:

- your operating system
- your Python version
- the exact install command you ran
- the full error output

## Development workflow

All new work should follow the branch structure below.

### Main branches

#### `main`

Stable branch of the project.

This branch should reflect only released or release-ready versions of the
library.

Direct development on `main` is not recommended.

#### `develop`

Integration branch for the next version.

Everything planned for the next release should go through `develop` first.

This is also the branch from which pre-release and final release tags are
created.

### Feature branches

All new work should be developed in isolated feature branches created from
`develop`.

Branch naming convention:

- `feat/...` for new features
- `fix/...` for bug fixes
- `chore/...` for maintenance tasks
- `docs/...` for documentation updates

Examples:

- `feat/class-core-sg`
- `feat/test-suite`
- `fix/mst-ordering`
- `docs/readme-update`

## Standard contribution flow

### 1. Sync with `develop`

```bash
git checkout develop
git pull upstream develop
```

If you do not have an `upstream` remote configured, use `origin` instead.

### 2. Create a feature branch

```bash
git checkout -b feat/my-feature
```

### 3. Make your changes locally

Keep changes focused and avoid unrelated edits in the same pull request.

Example commit flow:

```bash
git add .
git commit -m "feat: add hierarchy extraction to CoreSG"
```

### 4. Run local validation

Before opening a pull request, run the relevant tests and checks for your
change.

### 5. Open a pull request to `develop`

All feature branches should be merged into `develop` through a pull request.

Expected flow:

- push your branch
- open a pull request targeting `develop`
- wait for CI to run
- make sure tests, lint, formatting, and any relevant packaging checks pass
- request review if needed
- merge only after approval and passing checks

In short:

`feature branch -> develop`

## Pull request expectations

Contributions should aim to keep the project:

- clear
- reproducible
- testable
- consistent with the existing codebase

When possible:

- add tests for new behavior
- avoid unrelated changes in the same pull request
- keep pull requests focused
- document public API changes

## What should go into `develop`

The `develop` branch should contain the integrated state of the next release.

This includes:

- new features
- bug fixes
- tests
- packaging adjustments
- documentation
- CI workflow changes

Do not use `develop` for direct unreviewed work unless absolutely necessary.

## Pre-release workflow

Pre-releases are used to validate the package on TestPyPI before publishing a
final version to PyPI.

### When to create a pre-release

Create a pre-release when `develop` is mature enough to test:

- package build
- metadata
- installation
- importability
- minimal runtime behavior

### Versioning for pre-releases

Use versions such as:

- `0.1.0rc1`
- `0.1.0rc2`

### Preparing a pre-release

Update the package version in `pyproject.toml` on `develop`, commit the change,
and push it.

```bash
git checkout develop
git pull upstream develop
```

Then create and push the tag:

```bash
git tag -a v0.1.0rc1 -m "core-sg v0.1.0rc1"
git push origin v0.1.0rc1
```

### What happens next

Once the tag is pushed:

- the unified release workflow validates that the tagged commit belongs to
  `develop`
- the package is built once and the validated artifact is reused for publication
- the package is published automatically to TestPyPI
- a smoke test runs after publication against the TestPyPI package

If the tagged commit is not contained in `develop`, the release workflow is
skipped.

### If something needs to be fixed

If the pre-release needs corrections:

1. go back to `develop`
2. apply the fixes
3. bump the version to the next pre-release, for example `0.1.0rc2`
4. commit
5. create a new tag
6. publish again to TestPyPI

Do not reuse the same pre-release version after making changes.

## Final release workflow

A final release should only happen after a pre-release has been validated.

### Final version format

Examples:

- `0.1.0`
- `0.1.1`
- `0.2.0`

### Preparing the final release

Update the version in `pyproject.toml` on `develop`, commit, and push:

```bash
git checkout develop
git pull upstream develop
git add .
git commit -m "chore(release): prepare v0.1.0"
git push origin develop
```

Create and push the final tag:

```bash
git tag -a v0.1.0 -m "core-sg v0.1.0"
git push origin v0.1.0
```

### What happens next

Once the final tag is pushed:

- the unified release workflow validates that the tagged commit belongs to
  `develop`
- the package is built once and the validated artifact is reused for publication
- the package is published automatically to PyPI
- a smoke test runs after publication against the published PyPI package

If the tagged commit is not contained in `develop`, the release workflow is
skipped.

## Synchronizing `develop` and `main`

After a final release is published, `main` should receive the stable release
state.

Recommended flow:

`develop -> main`

### Procedure

1. open a pull request from `develop` to `main`
2. review and merge it after checks pass

This ensures:

- `main` reflects the stable released state
- `develop` remains the integration branch for the next version

## Versioning policy

The project should follow semantic versioning.

Format:

`MAJOR.MINOR.PATCH`

Examples:

- `0.1.0`
- `0.1.1`
- `0.2.0`
- `1.0.0`

### Pre-releases

Use release candidates:

- `0.1.0rc1`
- `0.1.0rc2`

### General rules

- PATCH for backward-compatible fixes
- MINOR for backward-compatible new functionality
- MAJOR for breaking changes

As long as the API is still evolving, it is natural to remain in the `0.x.y`
range.

## What to avoid

Please avoid:

- developing directly on `main`
- creating release tags from feature branches
- publishing from feature branches
- reusing the same release version after modifying the code
- mixing unrelated features in a single pull request

## Recommended project flow

The standard project flow should be:

- `feat/* -> develop -> vX.Y.ZrcN -> TestPyPI -> vX.Y.Z -> PyPI -> main`

This keeps development, integration, release validation, and stable publication
clearly separated.

## HDBSCAN acknowledgment

`core-sg` is structurally inspired by and technically dependent on the
`hdbscan` library.

If your contribution touches areas that reuse or adapt ideas or implementation
paths from `hdbscan`, please preserve that relationship clearly in
documentation and attribution where appropriate.

## Questions

If you are unsure about the correct contribution path, prefer opening your work
against `develop` and keeping the change focused and reviewable.
