from __future__ import annotations

import numpy as np


cdef class UnionFindC:
    cdef long long[:] parent
    cdef unsigned char[:] rank

    def __cinit__(self, Py_ssize_t n):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.uint8)

    cdef long long find(self, long long x):
        cdef long long[:] parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cdef bint union(self, long long a, long long b):
        cdef long long ra = self.find(a)
        cdef long long rb = self.find(b)
        cdef long long[:] parent
        cdef unsigned char[:] rank
        if ra == rb:
            return False

        parent = self.parent
        rank = self.rank
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True


def kruskal_mst_impl(object edges, int n_nodes, object mst_edge_dtype):
    cdef object e = np.asarray(edges)
    cdef object u_all = np.asarray(e[:, 0], dtype=np.int64)
    cdef object v_all = np.asarray(e[:, 1], dtype=np.int64)
    cdef object w_all = np.asarray(e[:, 2], dtype=np.float64)
    cdef object order = np.lexsort((u_all, v_all, w_all))
    cdef object mst_u = np.empty(n_nodes - 1, dtype=np.int64)
    cdef object mst_v = np.empty(n_nodes - 1, dtype=np.int64)
    cdef object mst_w = np.empty(n_nodes - 1, dtype=np.float64)
    cdef long long[:] u_view
    cdef long long[:] v_view
    cdef double[:] w_view
    cdef long long[:] mst_u_view = mst_u
    cdef long long[:] mst_v_view = mst_v
    cdef double[:] mst_w_view = mst_w
    cdef UnionFindC uf = UnionFindC(n_nodes)
    cdef Py_ssize_t i
    cdef Py_ssize_t m = 0
    cdef object out

    u_all = u_all[order]
    v_all = v_all[order]
    w_all = w_all[order]
    u_view = u_all
    v_view = v_all
    w_view = w_all

    for i in range(len(w_all)):
        if uf.union(u_view[i], v_view[i]):
            mst_u_view[m] = u_view[i]
            mst_v_view[m] = v_view[i]
            mst_w_view[m] = w_view[i]
            m += 1
            if m == n_nodes - 1:
                break

    if m != n_nodes - 1:
        raise ValueError("Disconex Graph: MST extraction is not possible.")

    out = np.empty(n_nodes - 1, dtype=mst_edge_dtype).view(np.recarray)
    out.u = mst_u
    out.v = mst_v
    out.distance = mst_w
    return out
