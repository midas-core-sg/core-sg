from __future__ import annotations

import builtins
import inspect
import importlib
import sys
import warnings

import numpy as np
import pytest
import core_sg.mst_kruskal as mst_kruskal_module
import core_sg.reweight as reweight_module

from core_sg.mst_kruskal import _kruskal_mst_python, kruskal_mst
from core_sg.reweight import (
    _METRIC_EDGE_LOOKUP_CACHE,
    _get_metric_edge_lookup,
    _reweight_core_sg_python,
    reweight_core_sg_mutual_reachability,
)

pytestmark = pytest.mark.unit


def _import_module_with_blocked_import(
    module_name: str,
    blocked_names: set[str],
    *,
    monkeypatch: pytest.MonkeyPatch,
):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in blocked_names or name.rsplit(".", 1)[-1] in blocked_names:
            raise ImportError(f"blocked optional import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for name in list(sys.modules):
        if name == "core_sg" or name.startswith("core_sg."):
            del sys.modules[name]
    return importlib.import_module(module_name)


class TestGraphHelpers:
    def test_knn_from_precomputed_excludes_self_and_returns_sorted_neighbors(
        self, core_sg_module
    ):
        D = np.array(
            [
                [0.0, 2.0, 1.0],
                [2.0, 0.0, 3.0],
                [1.0, 3.0, 0.0],
            ]
        )

        idx, dist = core_sg_module.knn_from_precomputed(D, k=2, include_self=False)

        assert idx.shape == (3, 2)
        assert dist.shape == (3, 2)
        assert np.array_equal(idx[0], np.array([2, 1]))
        assert np.all(dist[:, 0] <= dist[:, 1])

    def test_build_knng_vectors_preserves_shape_and_distances(self, core_sg_module):
        idxs = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
        dists = np.array([[1.0, 2.0], [1.0, 3.0], [2.0, 3.0]], dtype=np.float64)

        metric_edges, knng = core_sg_module.build_knng_vectors(
            idxs, dists, knng_size=3, k_max=2
        )

        assert metric_edges.shape == (6, 3)
        assert knng.shape == (6, 3)
        assert np.array_equal(knng[0], np.array([0.0, 1.0, 1.0]))
        assert np.array_equal(metric_edges[0], np.array([1.0, 0.0, 1.0]))

    def test_reweight_core_sg_mutual_reachability_uses_max_rule(self, core_sg_module):
        core_sg = np.array([[0, 1, -1.0], [1, 2, -1.0]], dtype=np.float64)
        metric_edges = np.array([[1, 0, 1.0], [2, 1, 4.0]], dtype=np.float64)
        core_k = np.array([0.5, 2.0, 3.0], dtype=np.float64)

        weighted = core_sg_module.reweight_core_sg_mutual_reachability(
            core_sg=core_sg,
            core_k=core_k,
            metric_edges=metric_edges,
            n_nodes=3,
        )

        assert np.array_equal(weighted[:, :2], np.array([[0.0, 1.0], [1.0, 2.0]]))
        assert np.allclose(weighted[:, 2], np.array([2.0, 4.0]))

    def test_kruskal_mst_builds_tree_with_n_minus_one_edges(self, core_sg_module):
        edges = np.array(
            [
                [0, 1, 1.0],
                [1, 2, 2.0],
                [0, 2, 10.0],
                [2, 3, 1.0],
            ],
            dtype=np.float64,
        )

        mst = core_sg_module.kruskal_mst(edges, n_nodes=4)

        assert mst.shape == (3,)
        assert np.array_equal(mst.u, np.array([0, 2, 1]))
        assert np.array_equal(mst.v, np.array([1, 3, 2]))
        assert np.allclose(mst.distance, np.array([1.0, 1.0, 2.0]))

    def test_kruskal_mst_preserves_signature(self):
        assert (
            str(inspect.signature(kruskal_mst))
            == "(edges: 'np.ndarray', n_nodes: 'int') -> 'np.recarray'"
        )

    def test_kruskal_mst_matches_python_backend_on_ties(self):
        edges = np.array(
            [
                [2, 0, 1.0],
                [0, 1, 1.0],
                [2, 1, 1.0],
                [2, 3, 2.0],
                [1, 3, 2.0],
            ],
            dtype=np.float64,
        )

        mst_python = _kruskal_mst_python(edges, n_nodes=4)
        mst_accelerated = kruskal_mst(edges, n_nodes=4)

        assert np.array_equal(mst_accelerated.u, mst_python.u)
        assert np.array_equal(mst_accelerated.v, mst_python.v)
        assert np.array_equal(mst_accelerated.distance, mst_python.distance)

    def test_kruskal_mst_preserves_disconnected_graph_error(self):
        edges = np.array([[0, 1, 1.0], [2, 3, 1.0]], dtype=np.float64)

        with pytest.raises(
            ValueError, match="Disconex Graph: MST extraction is not possible."
        ):
            kruskal_mst(edges, n_nodes=4)

    def test_kruskal_mst_warns_when_cython_backend_is_unavailable(self, monkeypatch):
        edges = np.array([[0, 1, 1.0], [1, 2, 2.0]], dtype=np.float64)
        monkeypatch.setattr(mst_kruskal_module, "_kruskal_mst_impl", None)
        monkeypatch.setattr(
            mst_kruskal_module, "_KRUSKAL_CYTHON_WARNING_EMITTED", False
        )

        with pytest.warns(
            RuntimeWarning,
            match="Cython backend for kruskal_mst is not available; using the Python fallback implementation.",
        ):
            kruskal_mst(edges, n_nodes=3)

    def test_reweight_preserves_signature(self):
        assert (
            str(inspect.signature(reweight_core_sg_mutual_reachability))
            == "(core_sg: 'np.ndarray', core_k: 'np.ndarray', metric_edges: 'np.ndarray', *, n_nodes: 'int') -> 'np.ndarray'"
        )

    def test_reweight_matches_python_backend(self, core_sg_module):
        core_sg = np.array([[0, 1, -1.0], [1, 2, -1.0], [0, 2, -1.0]], dtype=np.float64)
        metric_edges = np.array(
            [[1, 0, 1.0], [2, 1, 4.0], [2, 0, 2.5]], dtype=np.float64
        )
        core_k = np.array([0.5, 2.0, 3.0], dtype=np.float64)

        key_me_sorted, w_sorted = _get_metric_edge_lookup(metric_edges, 3)
        weighted_python = _reweight_core_sg_python(
            core_sg,
            core_k,
            key_me_sorted,
            w_sorted,
            n_nodes=3,
        )
        weighted_accelerated = reweight_core_sg_mutual_reachability(
            core_sg=core_sg,
            core_k=core_k,
            metric_edges=metric_edges,
            n_nodes=3,
        )

        assert np.array_equal(
            weighted_accelerated, core_sg_module.sort_core_sg(weighted_python)
        )

    def test_reweight_reuses_cached_metric_edge_lookup_for_multi_k(self):
        core_sg = np.array([[0, 1, -1.0], [1, 2, -1.0], [0, 2, -1.0]], dtype=np.float64)
        metric_edges = np.array(
            [[1, 0, 1.0], [2, 1, 4.0], [2, 0, 2.5]], dtype=np.float64
        )
        cache_key = (id(metric_edges), 3)
        _METRIC_EDGE_LOOKUP_CACHE.pop(cache_key, None)

        weighted_k2 = reweight_core_sg_mutual_reachability(
            core_sg=core_sg,
            core_k=np.array([0.5, 2.0, 3.0], dtype=np.float64),
            metric_edges=metric_edges,
            n_nodes=3,
        )
        assert cache_key in _METRIC_EDGE_LOOKUP_CACHE
        cached_key_arr, cached_weight_arr = _METRIC_EDGE_LOOKUP_CACHE[cache_key][1:]

        weighted_k3 = reweight_core_sg_mutual_reachability(
            core_sg=core_sg,
            core_k=np.array([1.5, 1.0, 2.2], dtype=np.float64),
            metric_edges=metric_edges,
            n_nodes=3,
        )
        assert _METRIC_EDGE_LOOKUP_CACHE[cache_key][1] is cached_key_arr
        assert _METRIC_EDGE_LOOKUP_CACHE[cache_key][2] is cached_weight_arr
        assert weighted_k2.shape == weighted_k3.shape

    def test_reweight_warns_when_cython_backend_is_unavailable(self, monkeypatch):
        core_sg = np.array([[0, 1, -1.0], [1, 2, -1.0]], dtype=np.float64)
        metric_edges = np.array([[1, 0, 1.0], [2, 1, 4.0]], dtype=np.float64)
        core_k = np.array([0.5, 2.0, 3.0], dtype=np.float64)

        monkeypatch.setattr(reweight_module, "_reweight_core_sg_from_lookup", None)
        monkeypatch.setattr(reweight_module, "_REWEIGHT_CYTHON_WARNING_EMITTED", False)

        with pytest.warns(
            RuntimeWarning,
            match="Cython backend for reweight_core_sg_mutual_reachability is not available; using the Python fallback implementation.",
        ):
            reweight_core_sg_mutual_reachability(
                core_sg=core_sg,
                core_k=core_k,
                metric_edges=metric_edges,
                n_nodes=3,
            )

    def test_knn_from_precomputed_rejects_non_square_distance_matrix(
        self, core_sg_module
    ):
        D = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="NxN"):
            core_sg_module.knn_from_precomputed(D, k=1)

    def test_knn_from_precomputed_validates_k_with_and_without_self(
        self, core_sg_module
    ):
        D = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="excluindo self"):
            core_sg_module.knn_from_precomputed(D, k=0, include_self=False)
        with pytest.raises(ValueError, match="incluindo self"):
            core_sg_module.knn_from_precomputed(D, k=3, include_self=True)

        idx, dist = core_sg_module.knn_from_precomputed(D, k=2, include_self=True)

        assert np.array_equal(idx[:, 0], np.array([0, 1], dtype=np.int64))
        assert np.array_equal(dist[:, 0], np.array([0.0, 0.0]))

    def test_build_knng_vectors_validates_input_shapes(self, core_sg_module):
        idxs = np.array([[1, 0]], dtype=np.int64)
        dists = np.array([[1.0, 1.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="idxs_arr.shape"):
            core_sg_module.build_knng_vectors(idxs, dists, knng_size=2, k_max=1)

        with pytest.raises(ValueError, match="distance_arr.shape"):
            core_sg_module.build_knng_vectors(
                np.array([[1], [0]], dtype=np.int64),
                dists,
                knng_size=2,
                k_max=1,
            )

    def test_add_mst_edges_validates_shapes_and_returns_original_when_all_exist(
        self, core_sg_module
    ):
        with pytest.raises(ValueError, match="metric_edges"):
            core_sg_module.add_mst_edges_to_metric_edges(
                np.array([1.0, 2.0]), np.array([[0.0, 1.0, 1.0]])
            )

        with pytest.raises(ValueError, match="mst"):
            core_sg_module.add_mst_edges_to_metric_edges(
                np.array([[1.0, 0.0, 1.0]]), np.array([0.0, 1.0, 1.0])
            )

        metric_edges = np.array([[1.0, 0.0, 1.0]], dtype=np.float64)
        mst = np.array([[0.0, 1.0, 1.0]], dtype=np.float64)

        result = core_sg_module.add_mst_edges_to_metric_edges(metric_edges, mst)

        assert np.array_equal(result, metric_edges)

    def test_kruskal_python_fallback_and_union_rank_branches(
        self, fake_hdbscan_modules, monkeypatch
    ):
        del fake_hdbscan_modules
        module = _import_module_with_blocked_import(
            "core_sg.mst_kruskal",
            {"_mst_kruskal"},
            monkeypatch=monkeypatch,
        )
        assert module._kruskal_mst_impl is None

        edges = np.array([[0.0, 1.0, 1.0], [1.0, 2.0, 2.0]], dtype=np.float64)
        with pytest.warns(RuntimeWarning, match="Cython backend"):
            module.kruskal_mst(edges, n_nodes=3)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            module.kruskal_mst(edges, n_nodes=3)

        uf = module.UnionFind(3)
        assert uf.union(0, 1) is True
        assert uf.union(2, 0) is True
        assert uf.find(2) == uf.find(0)

        with pytest.raises(ValueError, match="shape"):
            module.kruskal_mst(np.array([1.0, 2.0, 3.0]), n_nodes=3)
        with pytest.raises(ValueError, match="n_nodes"):
            module.kruskal_mst(edges, n_nodes=1)
        with pytest.raises(ValueError, match="Disconex Graph"):
            module._kruskal_mst_python(np.array([[0.0, 1.0, 1.0]]), n_nodes=3)

    def test_reweight_python_fallback_validation_and_cache_miss(
        self, fake_hdbscan_modules, monkeypatch
    ):
        del fake_hdbscan_modules
        module = _import_module_with_blocked_import(
            "core_sg.reweight",
            {"_reweight"},
            monkeypatch=monkeypatch,
        )
        assert module._reweight_core_sg_from_lookup is None

        with pytest.raises(ValueError, match="mst"):
            module.sort_core_sg(np.array([1.0, 2.0, 3.0]))

        w_sorted = np.array([1.0], dtype=np.float64)
        with pytest.raises(KeyError, match="missing"):
            module._reweight_core_sg_python(
                np.array([[0.0, 1.0, -1.0]]),
                np.ones(3, dtype=np.float64),
                np.array([4], dtype=np.int64),
                w_sorted,
                n_nodes=3,
            )

        core_sg = np.array([[0.0, 1.0, -1.0]], dtype=np.float64)
        metric_edges = np.array([[1.0, 0.0, 1.0]], dtype=np.float64)
        core_k = np.array([0.5, 0.7], dtype=np.float64)
        with pytest.warns(RuntimeWarning, match="Cython backend"):
            module.reweight_core_sg_mutual_reachability(
                core_sg, core_k, metric_edges, n_nodes=2
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            module.reweight_core_sg_mutual_reachability(
                core_sg, core_k, metric_edges, n_nodes=2
            )
