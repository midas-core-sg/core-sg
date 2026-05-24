# Core-SG Test Suite

This directory contains the `pytest` test suite for `core-sg`.

## Structure

```text
tests/
  conftest.py
  helpers.py
  validate.py
  integration/
    test_reference_equivalence.py
  unit/
    test_core_sg_fit.py
    test_core_sg_hierarchy.py
    test_core_sg_initialization.py
    test_graph_helpers.py
  validation/
    conftest.py
    test_class_contains_reference_mst.py
    test_class_extracted_mst_matches_reference.py
    test_class_hierarchy_artifacts.py
    test_class_matches_reference_weights.py
    test_core_sg_contains_reference_mst.py
    test_core_sg_matches_reference_weights.py
    test_extracted_mst_matches_reference.py
```

## Categories

- `unit/`: fast tests for isolated behavior using fixtures and fake `hdbscan` modules.
- `integration/`: tests that exercise the package boundary and real `hdbscan` integration.
- `validation/`: heavier algorithm-validation tests that compare Core-SG outputs with HDBSCAN reference behavior across multiple `k` values.
- `helpers.py`: shared test builders and comparison utilities.
- `validate.py`: validation helpers reused by the heavier equivalence tests.

## Naming Conventions

- Files follow `test_<subject>_<behavior>.py`.
- Test classes use `Test...` names as logical groupings.
- Test functions use descriptive `test_<behavior>_<expected_result>` naming.

## How to Run

Run the full suite:

```bash
pytest tests -q
```

Run only unit tests:

```bash
pytest tests/unit -q
```

Run only integration tests:

```bash
pytest tests/integration -q
```

Run only validation tests:

```bash
pytest tests/validation -q
```

Run tests by marker:

```bash
pytest -m unit -q
pytest -m integration -q
pytest -m "validation and not slow" -q
```

Run in verbose mode:

```bash
pytest tests -v -ra
```


