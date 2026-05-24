from __future__ import annotations

import numpy as np
import pytest

from core_sg.core_sg import build_core_sg_from_data
from core_sg.reweight import reweight_core_sg_mutual_reachability
from tests.validate import validate_weights_in_core_sg

pytestmark = [pytest.mark.validation, pytest.mark.slow]


@pytest.fixture(scope="module")
def weighted_core_sg(validation_dataset):
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


class TestCoreSGMatchesReferenceWeights:
    def test_core_sg_matches_reference_weights_for_each_k(
        self, weighted_core_sg, validation_dataset, reference_mst_builder
    ):
        n = validation_dataset["n"]

        for k_iter in range(30, 2, -2):
            mst_hdb = reference_mst_builder(weighted_core_sg["D"], k_iter)
            core_k = weighted_core_sg["core_k_list"][:, k_iter - 1]
            core_k = np.ascontiguousarray(core_k, dtype=np.float64)
            weighted = reweight_core_sg_mutual_reachability(
                core_sg=weighted_core_sg["core_sg"],
                core_k=core_k,
                metric_edges=weighted_core_sg["metric_edges"],
                n_nodes=n,
            )
            result = validate_weights_in_core_sg(
                weighted, mst_hdb, weighted_core_sg["D"], core_k, n, k_iter
            )

            if not result.ok:
                print(result)
                raise ValueError(f"A MST para k = {k_iter} nao esta contida no Core-SG")
