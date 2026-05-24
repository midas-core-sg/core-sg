from __future__ import annotations

from typing import Any

import numpy as np

try:
    from pynndescent import NNDescent
except ImportError as exc:
    NNDescent = None
    _PYNNDESCENT_IMPORT_ERROR = exc
else:
    _PYNNDESCENT_IMPORT_ERROR = None
from sklearn.metrics.pairwise import paired_distances
from sklearn.utils import check_random_state

from .edges import build_knng_vectors
from .reweight import sort_core_sg


def _resolve_metric(metric: str) -> str:
    if metric == "arccos":
        return "cosine"
    return metric


def _metric_kwargs(metric: str, p: int) -> dict[str, Any]:
    if metric == "minkowski":
        return {"p": p}
    return {}


def _get_pynndescent_class():
    if NNDescent is None:
        raise ImportError(
            "score-sg requires the 'pynndescent' dependency. "
            "Install the project with PyNNDescent available to use "
            "algorithm='score-sg'."
        ) from _PYNNDESCENT_IMPORT_ERROR
    return NNDescent


def _normalize_neighbor_graph(
    indices: np.ndarray,
    distances: np.ndarray,
    *,
    k_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float64)
    n_samples = indices.shape[0]

    neighbor_indices = np.empty((n_samples, k_max), dtype=np.int64)
    neighbor_distances = np.empty((n_samples, k_max), dtype=np.float64)

    for idx in range(n_samples):
        mask = indices[idx] != idx
        idx_row = indices[idx][mask]
        dist_row = distances[idx][mask]
        if idx_row.shape[0] < k_max:
            raise ValueError(
                "PyNNDescent did not return enough neighbors to build score-sg."
            )

        idx_row = idx_row[:k_max]
        dist_row = dist_row[:k_max]
        order = np.argsort(dist_row, kind="mergesort")
        neighbor_indices[idx] = idx_row[order]
        neighbor_distances[idx] = dist_row[order]

    return neighbor_indices, neighbor_distances


def build_approximate_knn_graph(
    X: np.ndarray,
    k_max: int,
    *,
    metric: str,
    p: int,
    random_state: int | np.random.RandomState | None,
    approx_knn_kwargs: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    NNDescent = _get_pynndescent_class()
    kwargs = {} if approx_knn_kwargs is None else dict(approx_knn_kwargs)
    if kwargs.get("compressed", False):
        raise ValueError(
            "score-sg requires access to NNDescent.neighbor_graph, so "
            "approx_knn_kwargs['compressed'] cannot be True."
        )

    requested_neighbors = min(X.shape[0], k_max + 1)
    kwargs["n_neighbors"] = max(int(kwargs.get("n_neighbors", 0)), requested_neighbors)
    kwargs["compressed"] = False
    kwargs.setdefault("random_state", random_state)
    if metric == "minkowski":
        kwargs.setdefault("metric_kwds", {"p": p})

    # For score-sg we only need the neighbor graph of the training set.
    # PyNNDescent documents `prepare()` as extra overhead for future queries,
    # so we intentionally build the index and read `neighbor_grap´h` directly.
    index = NNDescent(X, metric=_resolve_metric(metric), **kwargs)
    neighbor_graph = index.neighbor_graph

    if neighbor_graph is None:
        raise ValueError(
            "PyNNDescent did not expose neighbor_graph; score-sg requires the "
            "training-set neighbor graph to be available."
        )

    indices, distances = neighbor_graph

    return _normalize_neighbor_graph(indices, distances, k_max=k_max)


def build_approximate_core_k_list(
    neighbor_distances: np.ndarray,
    k_max: int,
) -> np.ndarray:
    n_samples = neighbor_distances.shape[0]
    core_k_list = np.empty((n_samples, k_max), dtype=np.float64)
    core_k_list[:, 0] = 0.0
    if k_max > 1:
        core_k_list[:, 1:] = neighbor_distances[:, : k_max - 1]
    return core_k_list


def compute_in_degrees(neighbor_indices: np.ndarray, n_samples: int) -> np.ndarray:
    return np.bincount(neighbor_indices.reshape(-1), minlength=n_samples).astype(
        np.int64,
        copy=False,
    )


def select_score_sg_anti_hubs(
    neighbor_indices: np.ndarray,
    in_degrees: np.ndarray,
    *,
    random_state: int | np.random.RandomState | None,
) -> np.ndarray:
    n_samples = neighbor_indices.shape[0]
    selected_size = max(1, int(np.floor(np.sqrt(n_samples))))
    order = np.lexsort((np.arange(n_samples, dtype=np.int64), in_degrees))

    if selected_size >= n_samples:
        return np.sort(order)

    cutoff_degree = in_degrees[order[selected_size - 1]]
    fixed = order[in_degrees[order] < cutoff_degree]
    remaining = selected_size - fixed.shape[0]
    tied = order[in_degrees[order] == cutoff_degree]

    if remaining <= 0:
        return np.sort(fixed[:selected_size])

    tie_scores = in_degrees[neighbor_indices[tied]].sum(axis=1)
    tie_order = np.argsort(tie_scores, kind="mergesort")
    ranked = tied[tie_order]
    ranked_scores = tie_scores[tie_order]

    rng = check_random_state(random_state)
    selected = fixed.tolist()
    cursor = 0
    while remaining > 0 and cursor < ranked.shape[0]:
        score = ranked_scores[cursor]
        end = cursor
        while end < ranked.shape[0] and ranked_scores[end] == score:
            end += 1

        bucket = ranked[cursor:end]
        if bucket.shape[0] <= remaining:
            selected.extend(bucket.tolist())
            remaining -= bucket.shape[0]
        else:
            chosen = rng.choice(bucket, size=remaining, replace=False)
            selected.extend(np.sort(chosen).tolist())
            remaining = 0
        cursor = end

    return np.sort(np.asarray(selected, dtype=np.int64))


def _compute_exact_edge_distances(
    X: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    metric: str,
    p: int,
) -> np.ndarray:
    return paired_distances(
        X[np.asarray(left, dtype=np.int64)],
        X[np.asarray(right, dtype=np.int64)],
        metric=_resolve_metric(metric),
        **_metric_kwargs(metric, p),
    ).astype(np.float64, copy=False)


def build_selected_clique(
    X: np.ndarray,
    selected: np.ndarray,
    *,
    metric: str,
    p: int,
) -> tuple[np.ndarray, np.ndarray]:
    if selected.shape[0] < 2:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, empty

    left_pos, right_pos = np.triu_indices(selected.shape[0], k=1)
    left = selected[left_pos]
    right = selected[right_pos]
    distances = _compute_exact_edge_distances(X, left, right, metric=metric, p=p)

    metric_edges = np.empty((distances.shape[0], 3), dtype=np.float64)
    metric_edges[:, 0] = np.maximum(left, right)
    metric_edges[:, 1] = np.minimum(left, right)
    metric_edges[:, 2] = distances

    support_edges = np.empty((distances.shape[0], 3), dtype=np.float64)
    support_edges[:, 0] = np.minimum(left, right)
    support_edges[:, 1] = np.maximum(left, right)
    support_edges[:, 2] = -1.0
    return metric_edges, support_edges


def is_graph_connected(core_sg: np.ndarray, n_nodes: int) -> bool:
    if n_nodes <= 1:
        return True
    if core_sg.shape[0] == 0:
        return False

    adjacency: list[list[int]] = [[] for _ in range(n_nodes)]
    for row in np.asarray(core_sg, dtype=np.float64):
        u = int(row[0])
        v = int(row[1])
        if u == v:
            continue
        adjacency[u].append(v)
        adjacency[v].append(u)

    visited = np.zeros(n_nodes, dtype=bool)
    stack = [0]
    visited[0] = True
    while stack:
        node = stack.pop()
        for neigh in adjacency[node]:
            if not visited[neigh]:
                visited[neigh] = True
                stack.append(neigh)
    return bool(np.all(visited))


def build_score_sg_from_data(
    X: np.ndarray,
    k_max: int,
    *,
    metric: str = "euclidean",
    p: int = 2,
    random_state: int | np.random.RandomState | None = None,
    approx_knn_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    neighbor_indices, neighbor_distances = build_approximate_knn_graph(
        X,
        k_max,
        metric=metric,
        p=p,
        random_state=random_state,
        approx_knn_kwargs=approx_knn_kwargs,
    )
    core_k_list = build_approximate_core_k_list(neighbor_distances, k_max)

    idx_a = np.repeat(np.arange(X.shape[0], dtype=np.int64), k_max)
    neigh = neighbor_indices.reshape(-1)
    exact_edge_distances = _compute_exact_edge_distances(
        X,
        idx_a,
        neigh,
        metric=metric,
        p=p,
    ).reshape(X.shape[0], k_max)

    metric_edges, knng_support = build_knng_vectors(
        neighbor_indices,
        exact_edge_distances,
        knng_size=X.shape[0],
        k_max=k_max,
    )

    in_degrees = compute_in_degrees(neighbor_indices, X.shape[0])
    selected = select_score_sg_anti_hubs(
        neighbor_indices,
        in_degrees,
        random_state=random_state,
    )

    clique_metric_edges, clique_support_edges = build_selected_clique(
        X,
        selected,
        metric=metric,
        p=p,
    )

    if clique_metric_edges.shape[0] > 0:
        metric_edges = np.vstack([metric_edges, clique_metric_edges])
        core_sg = np.vstack([knng_support, clique_support_edges])
    else:
        core_sg = knng_support

    core_sg = sort_core_sg(core_sg)
    return core_sg, metric_edges, core_k_list, np.asarray(X), selected
