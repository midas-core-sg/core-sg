from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs

hdbscan = pytest.importorskip("hdbscan")


@pytest.fixture(scope="module")
def validation_dataset():
    n = 5000
    d = 10
    centers = 10
    seed = 42
    X, _ = make_blobs(
        n_samples=n,
        n_features=d,
        centers=centers,
        random_state=seed,
    )
    return {
        "X": X,
        "n": n,
        "d": d,
        "centers": centers,
        "seed": seed,
    }


@pytest.fixture()
def reference_mst_builder():
    def _build(D: np.ndarray, k: int) -> np.ndarray:
        ref = hdbscan.HDBSCAN(
            min_cluster_size=k,
            min_samples=k,
            metric="precomputed",
            algorithm="generic",
            approx_min_span_tree=False,
            gen_min_span_tree=True,
            match_reference_implementation=True,
        ).fit(D)
        mst_hdb = np.asarray(ref._min_spanning_tree, dtype=np.float64)
        return mst_hdb[np.argsort(mst_hdb[:, 2], kind="mergesort")]

    return _build
