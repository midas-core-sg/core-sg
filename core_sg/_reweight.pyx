from __future__ import annotations

import numpy as np

cdef Py_ssize_t _binary_search(long long[:] arr, long long key) except -2:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = arr.shape[0]
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if arr[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    if lo >= arr.shape[0] or arr[lo] != key:
        return -1
    return lo


def reweight_core_sg_from_lookup(
    object core_sg,
    object core_k,
    object key_me_sorted,
    object w_sorted,
    int n_nodes,
):
    cdef object e = np.ascontiguousarray(core_sg, dtype=np.float64)
    cdef object ck = np.asarray(core_k, dtype=np.float64)
    cdef object keys = np.asarray(key_me_sorted, dtype=np.int64)
    cdef object weights = np.asarray(w_sorted, dtype=np.float64)
    cdef double[:, :] e_view = e
    cdef double[:] ck_view = ck
    cdef long long[:] key_view = keys
    cdef double[:] weight_view = weights
    cdef Py_ssize_t i
    cdef Py_ssize_t pos
    cdef long long ui
    cdef long long vi
    cdef long long bigger
    cdef long long smaller
    cdef long long key
    cdef double edge_weight
    cdef double core_weight

    for i in range(e_view.shape[0]):
        ui = <long long>e_view[i, 0]
        vi = <long long>e_view[i, 1]
        if ui >= vi:
            bigger = ui
            smaller = vi
        else:
            bigger = vi
            smaller = ui
        key = bigger * n_nodes + smaller
        pos = _binary_search(key_view, key)
        if pos < 0:
            raise KeyError("CoreSG edges are missing in metric_edges.")

        edge_weight = weight_view[pos]
        core_weight = ck_view[ui]
        if ck_view[vi] > core_weight:
            core_weight = ck_view[vi]
        if edge_weight > core_weight:
            core_weight = edge_weight
        e_view[i, 2] = core_weight

    return e
