from __future__ import annotations

import pytest

from core_sg.core_sg import CoreSG
from tests.validate import validate_weights_in_core_sg

pytestmark = [pytest.mark.validation, pytest.mark.slow]


@pytest.fixture(scope="module")
def weighted_class_core_sg(validation_dataset):
    X = validation_dataset["X"]
    core_sg = CoreSG(
        metric="euclidean", p=2, verbose=1, match_reference_implementation=True
    )
    core_sg._fit_for_tests(X, 50)
    return core_sg


class TestClassMatchesReferenceWeights:
    def test_class_core_sg_matches_reference_weights_for_each_k(
        self, weighted_class_core_sg, validation_dataset, reference_mst_builder
    ):
        n = validation_dataset["n"]

        for k_iter in range(50, 2, -2):
            mst_hdb = reference_mst_builder(
                weighted_class_core_sg.distance_matrix_, k_iter
            )
            weighted = weighted_class_core_sg.get_core_sg_mutual_reachability_distance(
                k_iter
            )
            result = validate_weights_in_core_sg(
                weighted,
                mst_hdb,
                weighted_class_core_sg.distance_matrix_,
                weighted_class_core_sg.get_core_distance(k_iter),
                n,
                k_iter,
            )

            if not result.ok:
                print(result)
                raise ValueError(f"A MST para k = {k_iter} nao esta contida no Core-SG")
