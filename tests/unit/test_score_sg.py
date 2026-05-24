from __future__ import annotations

import builtins
import importlib
import sys

import numpy as np
import pytest

from tests.helpers import make_sample_X, normalize_undirected_edges

pytestmark = pytest.mark.unit


def _import_score_sg_with_blocked_pynndescent(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pynndescent":
            raise ImportError("blocked optional import: pynndescent")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for name in list(sys.modules):
        if name == "core_sg" or name.startswith("core_sg."):
            del sys.modules[name]
    return importlib.import_module("core_sg.score_sg")


@pytest.fixture()
def score_sg_module(fake_hdbscan_modules):
    return importlib.import_module("core_sg.score_sg")


class TestScoreSGUtilities:
    def test_metric_resolution_and_kwargs_helpers(self, score_sg_module):
        assert score_sg_module._resolve_metric("arccos") == "cosine"
        assert score_sg_module._metric_kwargs("minkowski", 3) == {"p": 3}

    def test_build_approximate_core_k_list_uses_approximate_neighbor_distances(
        self, score_sg_module
    ):
        neighbor_distances = np.array(
            [[0.2, 0.5, 0.9], [0.3, 0.7, 1.4]],
            dtype=np.float64,
        )

        result = score_sg_module.build_approximate_core_k_list(neighbor_distances, 3)

        assert np.array_equal(
            result,
            np.array([[0.0, 0.2, 0.5], [0.0, 0.3, 0.7]], dtype=np.float64),
        )

    def test_build_approximate_core_k_list_handles_k_max_one(self, score_sg_module):
        result = score_sg_module.build_approximate_core_k_list(
            np.array([[2.0], [3.0]], dtype=np.float64),
            1,
        )

        assert np.array_equal(result, np.zeros((2, 1), dtype=np.float64))

    def test_compute_in_degrees_counts_directed_occurrences(self, score_sg_module):
        neighbor_indices = np.array(
            [
                [1, 2],
                [2, 3],
                [1, 3],
                [1, 2],
            ],
            dtype=np.int64,
        )

        result = score_sg_module.compute_in_degrees(neighbor_indices, 4)

        assert np.array_equal(result, np.array([0, 3, 3, 2], dtype=np.int64))

    def test_select_score_sg_anti_hubs_uses_secondary_tie_break(self, score_sg_module):
        neighbor_indices = np.array(
            [
                [3, 4],
                [0, 4],
                [1, 4],
                [4, 1],
                [3, 1],
            ],
            dtype=np.int64,
        )
        in_degrees = np.array([1, 3, 0, 2, 4], dtype=np.int64)

        selected = score_sg_module.select_score_sg_anti_hubs(
            neighbor_indices,
            in_degrees,
            random_state=7,
        )

        assert np.array_equal(selected, np.array([0, 2], dtype=np.int64))

    def test_select_score_sg_anti_hubs_random_tie_is_reproducible(
        self, score_sg_module
    ):
        neighbor_indices = np.array(
            [
                [2, 3],
                [2, 3],
                [0, 1],
                [0, 1],
            ],
            dtype=np.int64,
        )
        in_degrees = np.array([2, 2, 2, 2], dtype=np.int64)

        selected_a = score_sg_module.select_score_sg_anti_hubs(
            neighbor_indices,
            in_degrees,
            random_state=11,
        )
        selected_b = score_sg_module.select_score_sg_anti_hubs(
            neighbor_indices,
            in_degrees,
            random_state=11,
        )

        assert np.array_equal(selected_a, selected_b)

    def test_select_score_sg_anti_hubs_returns_all_points_when_selection_is_full(
        self, score_sg_module
    ):
        selected = score_sg_module.select_score_sg_anti_hubs(
            np.empty((1, 0), dtype=np.int64),
            np.array([0], dtype=np.int64),
            random_state=0,
        )

        assert np.array_equal(selected, np.array([0], dtype=np.int64))

    def test_normalize_neighbor_graph_rejects_rows_without_enough_neighbors(
        self, score_sg_module
    ):
        with pytest.raises(ValueError, match="enough neighbors"):
            score_sg_module._normalize_neighbor_graph(
                np.array([[0, 1]], dtype=np.int64),
                np.array([[0.0, 1.0]], dtype=np.float64),
                k_max=2,
            )

    def test_build_selected_clique_returns_empty_arrays_for_single_selection(
        self, score_sg_module
    ):
        metric_edges, support_edges = score_sg_module.build_selected_clique(
            np.array([[0.0, 0.0]], dtype=np.float64),
            np.array([0], dtype=np.int64),
            metric="euclidean",
            p=2,
        )

        assert metric_edges.shape == (0, 3)
        assert support_edges.shape == (0, 3)

    def test_is_graph_connected_handles_trivial_empty_and_self_loop_graphs(
        self, score_sg_module
    ):
        assert score_sg_module.is_graph_connected(np.empty((0, 3)), 1) is True
        assert score_sg_module.is_graph_connected(np.empty((0, 3)), 2) is False
        assert (
            score_sg_module.is_graph_connected(
                np.array([[0.0, 0.0, -1.0], [0.0, 1.0, -1.0]]),
                2,
            )
            is True
        )


class TestScoreSGBuild:
    def test_build_approximate_knn_graph_uses_neighbor_graph_and_removes_self_neighbor(
        self, score_sg_module, monkeypatch
    ):
        seen = {}

        class FakeNNDescent:
            def __init__(self, X, metric="euclidean", **kwargs):
                seen["metric"] = metric
                seen["kwargs"] = kwargs
                self.prepare_called = False
                self.query_called = False
                self._neighbor_graph = (
                    np.array(
                        [
                            [0, 2, 1],
                            [1, 0, 2],
                            [2, 0, 1],
                        ],
                        dtype=np.int64,
                    ),
                    np.array(
                        [
                            [0.0, 0.3, 0.6],
                            [0.0, 0.2, 0.5],
                            [0.0, 0.4, 0.7],
                        ],
                        dtype=np.float64,
                    ),
                )

            @property
            def neighbor_graph(self):
                return self._neighbor_graph

            def prepare(self):
                self.prepare_called = True

            def query(self, *args, **kwargs):
                self.query_called = True
                raise AssertionError("query should not be used for score-sg")

        monkeypatch.setattr(
            score_sg_module, "_get_pynndescent_class", lambda: FakeNNDescent
        )

        indices, distances = score_sg_module.build_approximate_knn_graph(
            make_sample_X()[:3],
            2,
            metric="euclidean",
            p=2,
            random_state=17,
            approx_knn_kwargs=None,
        )

        assert np.array_equal(
            indices, np.array([[2, 1], [0, 2], [0, 1]], dtype=np.int64)
        )
        assert np.array_equal(
            distances,
            np.array([[0.3, 0.6], [0.2, 0.5], [0.4, 0.7]], dtype=np.float64),
        )
        assert seen["metric"] == "euclidean"
        assert seen["kwargs"]["n_neighbors"] == 3
        assert seen["kwargs"]["compressed"] is False
        assert seen["kwargs"]["random_state"] == 17

    def test_build_approximate_knn_graph_rejects_compressed_indexes(
        self, score_sg_module
    ):
        with pytest.raises(ValueError, match="compressed"):
            score_sg_module.build_approximate_knn_graph(
                make_sample_X(),
                2,
                metric="euclidean",
                p=2,
                random_state=0,
                approx_knn_kwargs={"compressed": True},
            )

    def test_build_approximate_knn_graph_does_not_mutate_kwargs(
        self, score_sg_module, monkeypatch
    ):
        class FakeNNDescent:
            def __init__(self, X, metric="euclidean", **kwargs):
                del X, metric, kwargs
                self._neighbor_graph = (
                    np.array(
                        [
                            [0, 1, 2],
                            [1, 0, 2],
                            [2, 1, 0],
                        ],
                        dtype=np.int64,
                    ),
                    np.array(
                        [
                            [0.0, 1.0, 2.0],
                            [0.0, 1.0, 1.5],
                            [0.0, 1.5, 2.0],
                        ],
                        dtype=np.float64,
                    ),
                )

            @property
            def neighbor_graph(self):
                return self._neighbor_graph

        monkeypatch.setattr(
            score_sg_module, "_get_pynndescent_class", lambda: FakeNNDescent
        )
        approx_knn_kwargs = {"n_neighbors": 1, "random_state": 99}

        score_sg_module.build_approximate_knn_graph(
            make_sample_X()[:3],
            2,
            metric="euclidean",
            p=2,
            random_state=17,
            approx_knn_kwargs=approx_knn_kwargs,
        )

        assert approx_knn_kwargs == {"n_neighbors": 1, "random_state": 99}

    def test_build_approximate_knn_graph_passes_minkowski_metric_kwargs(
        self, score_sg_module, monkeypatch
    ):
        seen = {}

        class FakeNNDescent:
            def __init__(self, X, metric="euclidean", **kwargs):
                del X
                seen["metric"] = metric
                seen["kwargs"] = kwargs
                self.neighbor_graph = (
                    np.array([[0, 1], [1, 0]], dtype=np.int64),
                    np.array([[0.0, 2.0], [0.0, 2.0]], dtype=np.float64),
                )

        monkeypatch.setattr(
            score_sg_module, "_get_pynndescent_class", lambda: FakeNNDescent
        )

        idx, dist = score_sg_module.build_approximate_knn_graph(
            np.array([[0.0], [2.0]], dtype=np.float64),
            1,
            metric="minkowski",
            p=5,
            random_state=13,
            approx_knn_kwargs=None,
        )

        assert seen["metric"] == "minkowski"
        assert seen["kwargs"]["metric_kwds"] == {"p": 5}
        assert np.array_equal(idx, np.array([[1], [0]], dtype=np.int64))
        assert np.array_equal(dist, np.array([[2.0], [2.0]], dtype=np.float64))

    def test_build_approximate_knn_graph_rejects_missing_neighbor_graph(
        self, score_sg_module, monkeypatch
    ):
        class NoGraphNNDescent:
            def __init__(self, X, metric="euclidean", **kwargs):
                del X, metric, kwargs
                self.neighbor_graph = None

        monkeypatch.setattr(
            score_sg_module, "_get_pynndescent_class", lambda: NoGraphNNDescent
        )

        with pytest.raises(ValueError, match="neighbor_graph"):
            score_sg_module.build_approximate_knn_graph(
                np.array([[0.0], [1.0]], dtype=np.float64),
                1,
                metric="euclidean",
                p=2,
                random_state=0,
                approx_knn_kwargs=None,
            )

    def test_build_score_sg_from_data_returns_shared_payload_without_dense_D(
        self, score_sg_module, monkeypatch
    ):
        X = make_sample_X()[:4]
        approx_neighbors = np.array(
            [
                [1, 2],
                [0, 2],
                [1, 3],
                [2, 1],
            ],
            dtype=np.int64,
        )

        monkeypatch.setattr(
            score_sg_module,
            "build_approximate_knn_graph",
            lambda *args, **kwargs: (
                approx_neighbors,
                np.ones_like(approx_neighbors, dtype=np.float64),
            ),
        )
        monkeypatch.setattr(
            score_sg_module,
            "select_score_sg_anti_hubs",
            lambda *args, **kwargs: np.array([0, 3], dtype=np.int64),
        )
        core_sg, metric_edges, core_k_list, tree_data, anti_hubs = (
            score_sg_module.build_score_sg_from_data(X, 2, random_state=5)
        )

        assert tree_data.shape == X.shape
        assert np.array_equal(tree_data, X)
        assert np.array_equal(anti_hubs, np.array([0, 3], dtype=np.int64))
        assert core_k_list.shape == (4, 2)
        assert np.array_equal(core_k_list[:, 1], np.ones(4, dtype=np.float64))

        undirected_metric_edges = normalize_undirected_edges(metric_edges)
        undirected_core_sg = normalize_undirected_edges(core_sg, decimals=6)

        assert (0, 3, round(np.linalg.norm(X[0] - X[3]), 4)) in undirected_metric_edges
        assert (0, 3, -1.0) in undirected_core_sg

    def test_build_score_sg_from_data_uses_knn_support_when_no_clique_is_built(
        self, score_sg_module, monkeypatch
    ):
        X = np.array([[0.0, 0.0]], dtype=np.float64)
        monkeypatch.setattr(
            score_sg_module,
            "build_approximate_knn_graph",
            lambda *args, **kwargs: (
                np.array([[0]], dtype=np.int64),
                np.array([[0.0]], dtype=np.float64),
            ),
        )
        monkeypatch.setattr(
            score_sg_module,
            "select_score_sg_anti_hubs",
            lambda *args, **kwargs: np.array([0], dtype=np.int64),
        )

        core_sg, metric_edges, core_k_list, tree_data, selected = (
            score_sg_module.build_score_sg_from_data(X, 1)
        )

        assert core_sg.shape == (1, 3)
        assert metric_edges.shape == (1, 3)
        assert core_k_list.shape == (1, 1)
        assert np.array_equal(tree_data, X)
        assert np.array_equal(selected, np.array([0], dtype=np.int64))

    def test_is_graph_connected_detects_connected_and_disconnected_graphs(
        self, score_sg_module
    ):
        connected = np.array([[0, 1, -1.0], [1, 2, -1.0], [2, 3, -1.0]])
        disconnected = np.array([[0, 1, -1.0], [2, 3, -1.0]])

        assert score_sg_module.is_graph_connected(connected, 4) is True
        assert score_sg_module.is_graph_connected(disconnected, 4) is False

    def test_build_approximate_knn_graph_raises_helpful_error_without_dependency(
        self, score_sg_module, monkeypatch
    ):
        monkeypatch.setattr(score_sg_module, "NNDescent", None)
        monkeypatch.setattr(
            score_sg_module,
            "_PYNNDESCENT_IMPORT_ERROR",
            ImportError("missing pynndescent"),
        )

        with pytest.raises(
            ImportError, match="score-sg requires the 'pynndescent' dependency"
        ):
            score_sg_module.build_approximate_knn_graph(
                make_sample_X(),
                2,
                metric="euclidean",
                p=2,
                random_state=0,
                approx_knn_kwargs=None,
            )
        assert "missing pynndescent" in str(score_sg_module._PYNNDESCENT_IMPORT_ERROR)

    def test_score_sg_import_handles_missing_pynndescent(
        self, fake_hdbscan_modules, monkeypatch
    ):
        del fake_hdbscan_modules
        module = _import_score_sg_with_blocked_pynndescent(monkeypatch)

        assert module.NNDescent is None
        with pytest.raises(ImportError, match="pynndescent"):
            module._get_pynndescent_class()
