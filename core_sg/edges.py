from __future__ import annotations
import numpy as np


def build_knng_vectors(
    idxs_arr: np.ndarray,
    distance_arr: np.ndarray,
    knng_size: int,
    k_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return:
      metric_edges: (E,3) [bigger, smaller, dist]
      knng_to_insert: (E,3) [i, neighbor, dist]
    where E = knng_size*k_max.
    """
    if idxs_arr.shape != (knng_size, k_max):
        raise ValueError(
            f"idxs_arr.shape must be {(knng_size, k_max)}, got {idxs_arr.shape}"
        )
    if distance_arr.shape != (knng_size, k_max):
        raise ValueError(
            f"distance_arr.shape must be {(knng_size, k_max)}, got {distance_arr.shape}"
        )

    idxs_arr = np.ascontiguousarray(idxs_arr, dtype=np.int64)
    distance_arr = np.ascontiguousarray(distance_arr, dtype=np.float64)

    idx_a = np.repeat(np.arange(knng_size, dtype=np.int64), k_max)
    neigh = idxs_arr.reshape(-1)
    dist = distance_arr.reshape(-1)

    knng_to_insert = np.empty((knng_size * k_max, 3), dtype=np.float64)
    knng_to_insert[:, 0] = idx_a
    knng_to_insert[:, 1] = neigh
    knng_to_insert[:, 2] = dist

    bigger = np.maximum(idx_a, neigh)
    smaller = np.minimum(idx_a, neigh)

    metric_edges = np.empty((knng_size * k_max, 3), dtype=np.float64)
    metric_edges[:, 0] = bigger
    metric_edges[:, 1] = smaller
    metric_edges[:, 2] = dist

    return metric_edges, knng_to_insert


def add_mst_edges_to_metric_edges(
    metric_edges: np.ndarray, mst: np.ndarray, *, n_nodes: int | None = None
) -> np.ndarray:
    """
    Add MST edges in metric_edges, without duplicates (bigger/smaller conection)
    metric_edges: (E,3) [bigger, smaller, dist]
    mst:          (M,3) [u, v, w] (any order) OR [bigger, smaller, w]
    """
    me = np.ascontiguousarray(metric_edges, dtype=np.float64)
    mst = np.ascontiguousarray(mst, dtype=np.float64)

    if me.ndim != 2 or me.shape[1] < 3:
        raise ValueError("metric_edges must be (E,3).")
    if mst.ndim != 2 or mst.shape[1] < 3:
        raise ValueError("mst must be (M,3).")

    me_b = me[:, 0].astype(np.int64, copy=False)
    me_s = me[:, 1].astype(np.int64, copy=False)

    u = mst[:, 0].astype(np.int64, copy=False)
    v = mst[:, 1].astype(np.int64, copy=False)
    w = mst[:, 2].astype(np.float64, copy=False)

    mst_b = np.maximum(u, v)
    mst_s = np.minimum(u, v)

    if n_nodes is None:
        max_idx = int(
            max(
                me_b.max(initial=0),
                me_s.max(initial=0),
                mst_b.max(initial=0),
                mst_s.max(initial=0),
            )
        )
        n_nodes = max_idx + 1

    base = int(n_nodes)
    me_key = me_b * base + me_s
    me_key_sorted = np.sort(me_key, kind="mergesort")

    mst_key = mst_b * base + mst_s
    pos = np.searchsorted(me_key_sorted, mst_key)
    exists = (pos < me_key_sorted.size) & (me_key_sorted[pos] == mst_key)
    add_mask = ~exists

    if not np.any(add_mask):
        return me

    to_add = np.empty((int(add_mask.sum()), 3), dtype=np.float64)
    to_add[:, 0] = mst_b[add_mask]
    to_add[:, 1] = mst_s[add_mask]
    to_add[:, 2] = w[add_mask]

    return np.vstack([me, to_add])
