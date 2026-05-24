from __future__ import annotations

import heapq
from abc import ABC, abstractmethod

import numpy as np


def _validate_c(c: int) -> int:
    if not isinstance(c, int) or c < 1:
        raise ValueError("c must be an integer greater than or equal to 1.")
    return c


def _validate_labels(labels: np.ndarray, n_samples: int) -> np.ndarray:
    labels_array = np.asarray(labels)
    if labels_array.ndim != 1 or labels_array.shape[0] != n_samples:
        raise ValueError(
            "labels must be a one-dimensional array with length equal to n_samples."
        )
    return labels_array.copy()


def _validate_mst(min_spanning_tree: np.ndarray, n_samples: int) -> np.ndarray:
    mst = np.asarray(min_spanning_tree, dtype=np.float64)
    if mst.ndim != 2 or mst.shape[1] != 3:
        raise ValueError("min_spanning_tree must be an array with shape (n_edges, 3).")
    if n_samples > 0 and mst.shape[0] != n_samples - 1:
        raise ValueError(
            "min_spanning_tree must contain exactly n_samples - 1 edges for a tree."
        )
    return mst


def _build_tree_adjacency(
    min_spanning_tree: np.ndarray, n_samples: int
) -> list[list[tuple[int, float]]]:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n_samples)]
    for row in min_spanning_tree:
        u = int(row[0])
        v = int(row[1])
        weight = float(row[2])
        adjacency[u].append((v, weight))
        adjacency[v].append((u, weight))
    return adjacency


def _top_c_path_signature(
    previous_signature: tuple[float, ...], edge_weight: float, c: int
) -> tuple[float, ...]:
    signature = sorted((*previous_signature, edge_weight), reverse=True)[:c]
    return tuple(signature)


def _run_mst_label_propagation(
    *,
    labels: np.ndarray,
    min_spanning_tree: np.ndarray,
    n_samples: int,
    c: int,
) -> np.ndarray:
    labels_out = _validate_labels(labels, n_samples)
    mst = _validate_mst(min_spanning_tree, n_samples)

    if np.all(labels_out != -1) or np.all(labels_out == -1):
        return labels_out

    adjacency = _build_tree_adjacency(mst, n_samples)
    signatures: list[tuple[float, ...] | None] = [None] * n_samples
    queue: list[tuple[float, int, int]] = []

    zero_signature = tuple(0.0 for _ in range(c))
    for node in range(n_samples):
        if labels_out[node] == -1:
            continue
        signatures[node] = zero_signature
        for neighbor, weight in adjacency[node]:
            if labels_out[neighbor] == -1:
                heapq.heappush(queue, (weight, neighbor, node))

    while queue:
        weight, target, source = heapq.heappop(queue)
        if labels_out[target] != -1:
            continue

        same_target_sources = [source]
        while queue and queue[0][0] == weight and queue[0][1] == target:
            _, _, same_source = heapq.heappop(queue)
            same_target_sources.append(same_source)

        candidates: list[tuple[tuple[float, ...], int]] = []
        for source_node in same_target_sources:
            source_signature = signatures[source_node]
            if source_signature is None:
                continue
            path_signature = _top_c_path_signature(source_signature, weight, c)
            candidates.append((path_signature, int(labels_out[source_node])))

        if not candidates:
            continue

        best_signature, best_label = min(candidates, key=lambda item: item[0])
        labels_out[target] = best_label
        signatures[target] = best_signature

        for neighbor, neighbor_weight in adjacency[target]:
            if labels_out[neighbor] == -1:
                heapq.heappush(queue, (neighbor_weight, neighbor, target))

    return labels_out


class NoiseHandler(ABC):
    def __init__(self, c: int = 5) -> None:
        self.c = _validate_c(c)

    @abstractmethod
    def reassign(
        self,
        *,
        labels: np.ndarray,
        min_spanning_tree: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Return a reassigned copy of the label vector."""


class MSTLabelPropagationStrategy(NoiseHandler):
    def reassign(
        self,
        *,
        labels: np.ndarray,
        min_spanning_tree: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        return _run_mst_label_propagation(
            labels=labels,
            min_spanning_tree=min_spanning_tree,
            n_samples=n_samples,
            c=self.c,
        )


def build_noise_handler(
    strategy: str,
    *,
    c: int = 5,
) -> NoiseHandler:
    if strategy == "mst_label_propagation":
        return MSTLabelPropagationStrategy(c=c)

    raise ValueError(
        "Unknown noise_label_strategy: "
        f"{strategy!r}. Supported strategies: 'mst_label_propagation'."
    )


__all__ = [
    "NoiseHandler",
    "MSTLabelPropagationStrategy",
    "build_noise_handler",
]
