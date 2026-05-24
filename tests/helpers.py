from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DummyHDBSCANObject:
    labels_: np.ndarray
    probabilities_: np.ndarray
    cluster_persistence_: np.ndarray
    _condensed_tree: np.ndarray
    _single_linkage_tree: np.ndarray
    _min_spanning_tree: np.ndarray


@dataclass
class ValidationSummary:
    ok: bool
    missing: int
    compared: int


def make_sample_X() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [5.0, 5.0],
            [5.0, 6.0],
            [6.0, 5.0],
        ],
        dtype=np.float64,
    )


def make_fit_payload(
    n: int = 6, k_max: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DummyHDBSCANObject]:
    core_sg = np.array(
        [
            [0, 1, -1.0],
            [0, 2, -1.0],
            [1, 2, -1.0],
            [3, 4, -1.0],
            [3, 5, -1.0],
            [4, 5, -1.0],
            [2, 3, -1.0],
        ],
        dtype=np.float64,
    )
    metric_edges = np.array(
        [
            [1, 0, 1.0],
            [2, 0, 1.0],
            [2, 1, 1.4142],
            [4, 3, 1.0],
            [5, 3, 1.0],
            [5, 4, 1.4142],
            [3, 2, 6.4031],
        ],
        dtype=np.float64,
    )
    core_k_list = np.array(
        [[0.0, 1.0, 1.0, 7.0711]] * n,
        dtype=np.float64,
    )
    D = np.array(
        [
            [0.0, 1.0, 1.0, 7.0711, 7.8102, 7.8102],
            [1.0, 0.0, 1.4142, 6.4031, 7.0711, 7.2111],
            [1.0, 1.4142, 0.0, 6.4031, 7.2111, 7.0711],
            [7.0711, 6.4031, 6.4031, 0.0, 1.0, 1.0],
            [7.8102, 7.0711, 7.2111, 1.0, 0.0, 1.4142],
            [7.8102, 7.2111, 7.0711, 1.0, 1.4142, 0.0],
        ],
        dtype=np.float64,
    )
    mst = np.array(
        [
            [0, 1, 1.0],
            [0, 2, 1.0],
            [3, 4, 1.0],
            [3, 5, 1.0],
            [2, 3, 6.4031],
        ],
        dtype=np.float64,
    )
    condensed = np.array(
        [
            [0, 0, 1.0, 3],
            [1, 1, 1.0, 3],
            [2, 2, 1.0, 3],
            [3, 3, 1.0, 3],
            [4, 4, 1.0, 3],
            [5, 5, 1.0, 3],
        ],
        dtype=np.float64,
    )
    hdb_obj = DummyHDBSCANObject(
        labels_=np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
        probabilities_=np.array([1.0, 0.9, 0.95, 1.0, 0.8, 0.85], dtype=np.float64),
        cluster_persistence_=np.array([0.7, 0.8], dtype=np.float64),
        _condensed_tree=condensed,
        _single_linkage_tree=mst.copy(),
        _min_spanning_tree=mst.copy(),
    )
    return core_sg, metric_edges, core_k_list[:, :k_max], D, hdb_obj


def normalize_undirected_edges(
    edges: np.ndarray, *, with_weight: bool = True, decimals: int = 4
) -> set[tuple[Any, ...]]:
    arr = np.asarray(edges, dtype=np.float64)
    normalized: set[tuple[Any, ...]] = set()
    for row in arr:
        u, v = sorted((int(row[0]), int(row[1])))
        if with_weight:
            normalized.add((u, v, round(float(row[2]), decimals)))
        else:
            normalized.add((u, v))
    return normalized


def validate_reference_edges_in_core(
    core_sg: np.ndarray, mst_hdb: np.ndarray
) -> ValidationSummary:
    core_pairs = normalize_undirected_edges(core_sg, with_weight=False)
    ref_pairs = normalize_undirected_edges(mst_hdb, with_weight=False)
    missing = len(ref_pairs - core_pairs)
    return ValidationSummary(ok=missing == 0, missing=missing, compared=len(ref_pairs))


def validate_reference_weights_in_core(
    reweighted_core_sg: np.ndarray, mst_hdb: np.ndarray
) -> ValidationSummary:
    core_edges = normalize_undirected_edges(reweighted_core_sg, with_weight=True)
    ref_edges = normalize_undirected_edges(mst_hdb, with_weight=True)
    missing = len(ref_edges - core_edges)
    return ValidationSummary(ok=missing == 0, missing=missing, compared=len(ref_edges))


def assert_dataframe_schema(df: pd.DataFrame, expected_columns: list[str]) -> None:
    assert list(df.columns) == expected_columns
    assert df.shape[1] == len(expected_columns)
