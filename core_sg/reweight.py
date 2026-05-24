from __future__ import annotations

import weakref
from warnings import warn

import numpy as np

try:
    from ._reweight import reweight_core_sg_from_lookup as _reweight_core_sg_from_lookup
except ImportError:
    _reweight_core_sg_from_lookup = None

_METRIC_EDGE_LOOKUP_CACHE: dict[
    tuple[int, int], tuple[weakref.ReferenceType[np.ndarray], np.ndarray, np.ndarray]
] = {}
_REWEIGHT_CYTHON_WARNING_EMITTED = False


def sort_core_sg(core_sg: np.ndarray) -> np.ndarray:
    """
    CorSG to format [menor_idx, maior_idx, distancia]
    SOrting By (distancia, maior_idx, menor_idx).
    """
    if core_sg.ndim != 2 or core_sg.shape[1] < 3:
        raise ValueError("mst must have shape (n_edges, 3)")

    u = core_sg[:, 0].astype(np.int64, copy=False)
    v = core_sg[:, 1].astype(np.int64, copy=False)
    w = core_sg[:, 2].astype(np.float64, copy=False)

    u_min = np.minimum(u, v)
    v_max = np.maximum(u, v)

    core_sg_tmp = np.empty((core_sg.shape[0], 3), dtype=np.float64)
    core_sg_tmp[:, 0] = u_min
    core_sg_tmp[:, 1] = v_max
    core_sg_tmp[:, 2] = w

    order = np.lexsort((core_sg_tmp[:, 0], core_sg_tmp[:, 1], core_sg_tmp[:, 2]))
    core_sg_tmp = core_sg_tmp[order]

    return core_sg_tmp


def _build_metric_edge_lookup(
    metric_edges: np.ndarray, n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    me = np.ascontiguousarray(metric_edges, dtype=np.float64)
    me_b = me[:, 0].astype(np.int64, copy=False)
    me_s = me[:, 1].astype(np.int64, copy=False)
    me_w = me[:, 2].astype(np.float64, copy=False)

    base = int(n_nodes)
    key_me = me_b * base + me_s
    order = np.argsort(key_me, kind="mergesort")
    return np.ascontiguousarray(key_me[order], dtype=np.int64), np.ascontiguousarray(
        me_w[order], dtype=np.float64
    )


def _get_metric_edge_lookup(
    metric_edges: np.ndarray, n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    cache_key = (id(metric_edges), int(n_nodes))
    cached = _METRIC_EDGE_LOOKUP_CACHE.get(cache_key)
    if cached is not None:
        ref, key_me_sorted, w_sorted = cached
        if ref() is metric_edges:
            return key_me_sorted, w_sorted

    key_me_sorted, w_sorted = _build_metric_edge_lookup(metric_edges, n_nodes)
    _METRIC_EDGE_LOOKUP_CACHE[cache_key] = (
        weakref.ref(metric_edges),
        key_me_sorted,
        w_sorted,
    )
    return key_me_sorted, w_sorted


def _reweight_core_sg_python(
    core_sg: np.ndarray,
    core_k: np.ndarray,
    key_me_sorted: np.ndarray,
    w_sorted: np.ndarray,
    *,
    n_nodes: int,
) -> np.ndarray:
    e = np.ascontiguousarray(core_sg, dtype=np.float64)
    u = e[:, 0].astype(np.int64, copy=False)
    v = e[:, 1].astype(np.int64, copy=False)

    bigger = np.maximum(u, v)
    smaller = np.minimum(u, v)
    key_core = bigger * int(n_nodes) + smaller

    pos = np.searchsorted(key_me_sorted, key_core)
    ok = (pos < key_me_sorted.size) & (key_me_sorted[pos] == key_core)
    if not np.all(ok):
        raise KeyError("CoreSG edges are missing in metric_edges.")

    dist_uv = w_sorted[pos]
    ck = np.asarray(core_k, dtype=np.float64)
    e[:, 2] = np.maximum(np.maximum(ck[u], ck[v]), dist_uv)
    return e


def reweight_core_sg_mutual_reachability(
    core_sg: np.ndarray,
    core_k: np.ndarray,
    metric_edges: np.ndarray,
    *,
    n_nodes: int,
) -> np.ndarray:
    """
    Update Core-sg weights:
      w(u,v) = max(core_k[u], core_k[v], dist(u,v))
    Using vectorized lookup in metric_edges (bigger, smaller, dist).

    Retorna (E,3) float64.
    """
    assert np.all(core_sg[:, 2] >= -1)
    key_me_sorted, w_sorted = _get_metric_edge_lookup(metric_edges, n_nodes)

    if _reweight_core_sg_from_lookup is not None:
        e = _reweight_core_sg_from_lookup(
            core_sg,
            np.asarray(core_k, dtype=np.float64),
            key_me_sorted,
            w_sorted,
            int(n_nodes),
        )
    else:
        global _REWEIGHT_CYTHON_WARNING_EMITTED
        if not _REWEIGHT_CYTHON_WARNING_EMITTED:
            warn(
                "Cython backend for reweight_core_sg_mutual_reachability is not available; using the Python fallback implementation.",
                RuntimeWarning,
                stacklevel=2,
            )
            _REWEIGHT_CYTHON_WARNING_EMITTED = True
        e = _reweight_core_sg_python(
            core_sg,
            core_k,
            key_me_sorted,
            w_sorted,
            n_nodes=n_nodes,
        )

    assert np.all(e[:, 2] != -1), "Placeholder values not overwritten"
    e = sort_core_sg(e)
    return e
