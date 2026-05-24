"""
Internal HDBSCAN adapter used by Core-SG.

This module centralizes direct interaction with HDBSCAN, including private APIs
required for HDBSCAN-style hierarchy outputs. It is not part of the public
Core-SG API. See the repository-level THIRD_PARTY_NOTICES.md file for the
HDBSCAN BSD-3-Clause attribution.
"""

from __future__ import annotations

from inspect import Parameter, signature
from typing import Any

import hdbscan
import numpy as np
from hdbscan._hdbscan_linkage import label
from hdbscan.hdbscan_ import _tree_to_labels
from hdbscan.plots import CondensedTree, MinimumSpanningTree, SingleLinkageTree


def reference_mst_original_distance(
    D: np.ndarray,
    *,
    k_max: int,
) -> tuple[Any, np.ndarray]:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=k_max,
        min_samples=k_max,
        metric="precomputed",
        algorithm="generic",
        approx_min_span_tree=False,
        gen_min_span_tree=True,
        match_reference_implementation=True,
    )
    clusterer.fit(D)
    return clusterer, np.asarray(clusterer._min_spanning_tree, dtype=np.float64)


def mst_to_single_linkage_tree(min_spanning_tree: np.ndarray) -> np.ndarray:
    return label(min_spanning_tree)


def filter_tree_to_labels_kwargs(tree_kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Keep only keyword arguments supported by the installed HDBSCAN private API.

    HDBSCAN's private `_tree_to_labels(...)` signature varies across releases.
    Core-SG accepts a stable public parameter set, but this adapter must forward
    only what the installed version can consume.
    """
    parameters = signature(_tree_to_labels).parameters
    if any(
        parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        return dict(tree_kwargs)

    return {key: value for key, value in tree_kwargs.items() if key in parameters}


def tree_to_labels(
    data: np.ndarray,
    single_linkage_tree: np.ndarray,
    *,
    tree_kwargs: dict[str, Any],
    min_spanning_tree: np.ndarray,
) -> tuple[Any, ...]:
    filtered_kwargs = filter_tree_to_labels_kwargs(tree_kwargs)
    return _tree_to_labels(data, single_linkage_tree, **filtered_kwargs) + (
        min_spanning_tree,
    )


def wrap_condensed_tree(raw_tree: np.ndarray, labels: np.ndarray) -> Any:
    return CondensedTree(raw_tree, labels)


def wrap_single_linkage_tree(raw_tree: np.ndarray) -> Any:
    return SingleLinkageTree(raw_tree)


def wrap_minimum_spanning_tree(raw_tree: np.ndarray, raw_data: np.ndarray) -> Any:
    return MinimumSpanningTree(raw_tree, raw_data)
