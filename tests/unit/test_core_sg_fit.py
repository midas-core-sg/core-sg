from __future__ import annotations

import numpy as np
import pytest

from tests.helpers import assert_dataframe_schema, make_fit_payload

pytestmark = pytest.mark.unit


@pytest.fixture()
def patched_fit_dependencies(core_sg_module, monkeypatch):
    payload = make_fit_payload()

    def fake_build_core_sg_from_data(X, k_max, metric, p, _round_distances=False):
        assert X.shape[0] == payload[3].shape[0]
        assert k_max == 4
        assert metric == "euclidean"
        assert p == 2
        assert _round_distances is True
        return payload

    monkeypatch.setattr(
        core_sg_module, "build_core_sg_from_data", fake_build_core_sg_from_data
    )
    return payload


class TestCoreSGFit:
    def test_fit_returns_self_and_stores_artifacts(
        self, core_sg_module, patched_fit_dependencies, sample_X
    ):
        obj = core_sg_module.CoreSG(metric="euclidean", p=2, verbose=0)

        returned = obj._fit_for_tests(sample_X, 4)
        core_sg, metric_edges, core_k_list, D, hdb_obj = patched_fit_dependencies

        assert returned is obj
        assert obj.n_samples_ == sample_X.shape[0]
        assert obj.k_max_ == 4
        assert obj._raw_data_ is sample_X
        assert np.array_equal(obj.support_graph_, core_sg)
        assert np.array_equal(obj.metric_edges_, metric_edges)
        assert np.array_equal(obj.core_distances_, core_k_list)
        assert np.array_equal(obj.distance_matrix_, D)
        assert np.array_equal(obj._tree_to_labels_data_, D)
        with pytest.raises(
            AttributeError,
            match="Attribute 'anti_hubs_' is available only when algorithm='score-sg'",
        ):
            _ = obj.anti_hubs_

        assert np.array_equal(obj.labels_k_max_, hdb_obj.labels_)
        assert np.array_equal(obj.probabilities_k_max_, hdb_obj.probabilities_)
        assert np.array_equal(
            obj.cluster_persistence_k_max_, hdb_obj.cluster_persistence_
        )
        assert np.array_equal(
            obj._single_linkage_tree_k_max_array_, hdb_obj._single_linkage_tree
        )
        assert np.array_equal(
            obj._min_spanning_tree_k_max_array_, hdb_obj._min_spanning_tree
        )

    def test_fit_reports_progress_through_callback(
        self, core_sg_module, patched_fit_dependencies, sample_X
    ):
        events = []
        obj = core_sg_module.CoreSG(
            metric="euclidean",
            p=2,
            progress_callback=lambda event, elapsed, info: events.append(
                (event, elapsed, info)
            ),
        )

        obj._fit_for_tests(sample_X, 4)

        assert len(events) == 1
        event, elapsed, info = events[0]
        assert event == "build"
        assert elapsed >= 0.0
        assert info == {"k_max": 4, "algorithm": "core-sg"}

    def test_extract_mst_from_core_sg_returns_cached_mst_for_k_max(
        self, core_sg_module, patched_fit_dependencies, sample_X
    ):
        obj = core_sg_module.CoreSG()
        obj._fit_for_tests(sample_X, 4)

        result = obj.extract_mst_from_core_sg(4)

        assert np.array_equal(result, obj._min_spanning_tree_k_max_array_)
        assert isinstance(result, np.ndarray)
        assert result.shape == (sample_X.shape[0] - 1, 3)
        assert np.isfinite(result[:, 2]).all()

    def test_extract_mst_from_core_sg_to_dataframe_has_expected_schema(
        self, core_sg_module, patched_fit_dependencies, sample_X
    ):
        obj = core_sg_module.CoreSG()
        obj._fit_for_tests(sample_X, 4)

        df = obj.extract_mst_from_core_sg(4, toDF=True)

        assert_dataframe_schema(df, ["to", "from", "weight"])
        assert str(df["to"].dtype) == "int64"
        assert str(df["from"].dtype) == "int64"
        assert str(df["weight"].dtype) == "float64"
        assert df.shape == (sample_X.shape[0] - 1, 3)

    def test_get_fitted_hdbscan_objects_raw_returns_saved_arrays(
        self, core_sg_module, patched_fit_dependencies, sample_X
    ):
        obj = core_sg_module.CoreSG()
        obj._fit_for_tests(sample_X, 4)

        fitted = obj.get_fitted_hdbscan_objects(wrapped=False)

        assert set(fitted) == {
            "labels_",
            "probabilities_",
            "cluster_persistence_",
            "condensed_tree_",
            "single_linkage_tree_",
            "minimum_spanning_tree_",
        }
        assert np.array_equal(fitted["labels_"], obj.labels_k_max_)
        assert np.array_equal(
            fitted["minimum_spanning_tree_"], obj._min_spanning_tree_k_max_array_
        )
        assert isinstance(fitted["condensed_tree_"], np.ndarray)
        assert isinstance(fitted["single_linkage_tree_"], np.ndarray)
        assert isinstance(fitted["minimum_spanning_tree_"], np.ndarray)

    def test_get_fitted_hdbscan_objects_wrapped_returns_hdbscan_like_wrappers(
        self, core_sg_module, patched_fit_dependencies, sample_X
    ):
        obj = core_sg_module.CoreSG()
        obj._fit_for_tests(sample_X, 4)

        fitted = obj.get_fitted_hdbscan_objects(wrapped=True)

        assert fitted["condensed_tree_"].to_pandas().shape[0] >= obj.n_samples_
        assert fitted["single_linkage_tree_"].to_pandas().shape[0] == obj.n_samples_ - 1
        assert (
            fitted["minimum_spanning_tree_"].to_pandas().shape[0] == obj.n_samples_ - 1
        )
        assert not isinstance(fitted["condensed_tree_"], np.ndarray)
        assert not isinstance(fitted["single_linkage_tree_"], np.ndarray)
        assert not isinstance(fitted["minimum_spanning_tree_"], np.ndarray)

    def test_minimum_spanning_tree_fit_wrapper_warns_without_raw_data(
        self, core_sg_module, patched_fit_dependencies, sample_X
    ):
        obj = core_sg_module.CoreSG()
        obj._fit_for_tests(sample_X, 4)
        obj._raw_data_ = None

        with pytest.warns(UserWarning, match="No raw data is available"):
            wrapped = obj.minimum_spanning_tree_k_max_

        assert wrapped is None

    def test_fit_passes_score_sg_configuration_to_builder(
        self, core_sg_module, monkeypatch, sample_X
    ):
        payload = make_fit_payload()
        seen = {}

        def fake_build_score_sg_from_data(
            X,
            k_max,
            metric,
            p,
            random_state=None,
            approx_knn_kwargs=None,
            _round_distances=False,
        ):
            seen["shape"] = X.shape
            seen["k_max"] = k_max
            seen["metric"] = metric
            seen["p"] = p
            seen["random_state"] = random_state
            seen["approx_knn_kwargs"] = approx_knn_kwargs
            seen["_round_distances"] = _round_distances
            return (
                payload[0],
                payload[1],
                payload[2],
                sample_X.copy(),
                np.array([0, 2], dtype=np.int64),
            )

        monkeypatch.setattr(
            core_sg_module, "build_score_sg_from_data", fake_build_score_sg_from_data
        )
        monkeypatch.setattr(core_sg_module, "is_graph_connected", lambda *a, **k: True)
        monkeypatch.setattr(
            core_sg_module,
            "mst_from_core_sg",
            lambda **kwargs: payload[4]._min_spanning_tree,
        )
        monkeypatch.setattr(
            core_sg_module,
            "mst_to_single_linkage_tree",
            lambda mst: payload[4]._single_linkage_tree,
        )
        monkeypatch.setattr(
            core_sg_module,
            "tree_to_labels",
            lambda instance, single_linkage_tree, min_spanning_tree: (
                payload[4].labels_,
                payload[4].probabilities_,
                payload[4].cluster_persistence_,
                payload[4]._condensed_tree,
                single_linkage_tree,
                min_spanning_tree,
            ),
        )

        obj = core_sg_module.CoreSG(
            algorithm="score-sg",
            random_state=13,
            approx_knn_kwargs={"n_trees": 8},
        )
        obj._fit_for_tests(sample_X, 4)

        assert seen == {
            "shape": sample_X.shape,
            "k_max": 4,
            "metric": "euclidean",
            "p": 2,
            "random_state": 13,
            "approx_knn_kwargs": {"n_trees": 8},
            "_round_distances": False,
        }
        with pytest.raises(
            AttributeError,
            match="Attribute 'distance_matrix_' is available only when algorithm='core-sg'",
        ):
            _ = obj.distance_matrix_
        assert np.array_equal(obj._tree_to_labels_data_, sample_X)
        assert np.array_equal(obj.anti_hubs_, np.array([0, 2], dtype=np.int64))
        assert obj.anti_hubs_.shape == (int(np.floor(np.sqrt(sample_X.shape[0]))),)
        assert np.issubdtype(obj.anti_hubs_.dtype, np.integer)

    def test_score_sg_fit_raises_for_disconnected_support_graph(
        self, core_sg_module, monkeypatch, sample_X
    ):
        payload = make_fit_payload()
        monkeypatch.setattr(
            core_sg_module,
            "build_score_sg_from_data",
            lambda *args, **kwargs: (
                payload[0],
                payload[1],
                payload[2],
                sample_X.copy(),
                np.array([1, 3], dtype=np.int64),
            ),
        )
        monkeypatch.setattr(core_sg_module, "is_graph_connected", lambda *a, **k: False)

        obj = core_sg_module.CoreSG(algorithm="score-sg")

        with pytest.raises(ValueError, match="support graph is disconnected"):
            obj._fit_for_tests(sample_X, 4)
