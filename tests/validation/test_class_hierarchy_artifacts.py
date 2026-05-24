from __future__ import annotations

import pytest

from core_sg.core_sg import CoreSG
from tests.validate import validate_core_sg_atributtes

pytestmark = [pytest.mark.validation, pytest.mark.slow]


class TestClassHierarchyArtifacts:
    def test_extract_hierarchy_exposes_wrapped_artifacts_for_each_k(
        self, validation_dataset
    ):
        X = validation_dataset["X"]
        n = validation_dataset["n"]
        k_max = 30

        core_sg = CoreSG(
            metric="euclidean", p=2, verbose=1, match_reference_implementation=True
        )
        core_sg._fit_for_tests(X, k_max)

        for k_iter in range(k_max, 2, -2):
            try:
                core_sg.extract_hierarchy_from_core_sg(k_iter)

                attributes = {
                    "mst": core_sg.minimum_spanning_tree_,
                    "mst_k": core_sg.minimum_spanning_tree_k_max_,
                    "slt": core_sg.single_linkage_tree_,
                    "slt_k": core_sg.single_linkage_tree_k_max_,
                    "condensed": core_sg.condensed_tree_,
                    "condensed_k": core_sg.condensed_tree_k_max_,
                }

                for key, artifact in attributes.items():
                    result = validate_core_sg_atributtes(
                        artifact, n, k_iter, key in ("condensed", "condensed_k")
                    )

                    if not result.ok:
                        raise RuntimeError(
                            f"Erro na transformacao dos atributos em pandas. {result}"
                        )
            except Exception as exc:
                raise RuntimeError(f"Erro no uso da classe CoreSG {exc}") from exc
