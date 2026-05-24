from __future__ import annotations

import pytest

from core_sg.core_sg import CoreSG
from tests.validate import validate_mst_from_core_sg

pytestmark = [pytest.mark.validation, pytest.mark.slow]


@pytest.fixture(scope="module")
def extracted_class_core_sg(validation_dataset):
    X = validation_dataset["X"]
    core_sg = CoreSG(
        metric="euclidean", p=2, verbose=1, match_reference_implementation=True
    )
    core_sg._fit_for_tests(X, 30)
    return core_sg


class TestClassExtractedMSTMatchesReference:
    def test_class_extracted_mst_matches_reference_for_each_k(
        self, extracted_class_core_sg, validation_dataset, reference_mst_builder
    ):
        n = validation_dataset["n"]

        for k_iter in range(30, 2, -2):
            mst_core = extracted_class_core_sg.extract_mst_from_core_sg(k_iter)
            mst_hdb = reference_mst_builder(
                extracted_class_core_sg.distance_matrix_, k_iter
            )
            result = validate_mst_from_core_sg(mst_core, mst_hdb, n, k_iter)

            if not result.ok:
                print(result)
                raise ValueError(
                    f"A MST para k = {k_iter} nao eh igual à extraída via HDBSCAN"
                )
