from __future__ import annotations

import numpy as np
import pytest

from core_sg.core_sg import build_core_sg_from_data, mst_from_core_sg
from tests.validate import validate_mst_from_core_sg

pytestmark = [pytest.mark.validation, pytest.mark.slow]


@pytest.fixture(scope="module")
def extracted_core_sg(validation_dataset):
    X = validation_dataset["X"]
    core_sg, metric_edges, core_k_list, D, hdb_obj = build_core_sg_from_data(
        X,
        k_max=30,
        metric="euclidean",
        pairwise_dtype=np.float64,
        _round_distances=True,
    )
    return {
        "core_sg": core_sg,
        "metric_edges": metric_edges,
        "core_k_list": core_k_list,
        "D": D,
        "hdb_obj": hdb_obj,
    }


class TestExtractedMSTMatchesReference:
    def test_extracted_mst_matches_reference_for_each_k(
        self, extracted_core_sg, validation_dataset, reference_mst_builder
    ):
        n = validation_dataset["n"]

        for k_iter in range(30, 2, -2):
            mst_core = mst_from_core_sg(
                core_sg=extracted_core_sg["core_sg"],
                metric_edges=extracted_core_sg["metric_edges"],
                core_k_list=extracted_core_sg["core_k_list"],
                n_nodes=n,
                k=k_iter,
            )
            mst_hdb = reference_mst_builder(extracted_core_sg["D"], k_iter)
            result = validate_mst_from_core_sg(mst_core, mst_hdb, n, k_iter)

            if not result.ok:
                print(result)
                raise ValueError(
                    f"A MST para k = {k_iter} nao eh igual à extraída via HDBSCAN"
                )
