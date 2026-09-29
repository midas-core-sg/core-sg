from __future__ import annotations

import numpy as np

from tests.validate import validate_mst_from_core_sg


def test_validate_mst_from_core_sg_accepts_tie_equivalent_mst_weights():
    mst_core = np.array(
        [
            [0, 1, 1.0],
            [1, 2, 1.0],
            [2, 3, 2.0],
        ],
        dtype=np.float64,
    )
    mst_hdb = np.array(
        [
            [0, 2, 1.0],
            [0, 3, 1.0],
            [1, 3, 2.0],
        ],
        dtype=np.float64,
    )

    result = validate_mst_from_core_sg(mst_core, mst_hdb, n=3, k=2)

    assert result.ok
    assert result.missing_in_core == 3


def test_validate_mst_from_core_sg_rejects_different_mst_weights():
    mst_core = np.array(
        [
            [0, 1, 1.0],
            [1, 2, 1.0],
            [2, 3, 3.0],
        ],
        dtype=np.float64,
    )
    mst_hdb = np.array(
        [
            [0, 2, 1.0],
            [0, 3, 1.0],
            [1, 3, 2.0],
        ],
        dtype=np.float64,
    )

    result = validate_mst_from_core_sg(mst_core, mst_hdb, n=3, k=2)

    assert not result.ok
    assert result.missing_in_core == 3
