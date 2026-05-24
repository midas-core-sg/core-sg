from __future__ import annotations

import numpy as np
import pytest

from tests.helpers import make_fit_payload

pytestmark = pytest.mark.unit


@pytest.fixture()
def fitted_obj(core_sg_module, monkeypatch, sample_X):
    payload = make_fit_payload()

    def fake_build_core_sg_from_data(X, k_max, metric, p, _round_distances=False):
        return payload

    monkeypatch.setattr(
        core_sg_module, "build_core_sg_from_data", fake_build_core_sg_from_data
    )
    obj = core_sg_module.CoreSG(
        cluster_selection_method="leaf", allow_single_cluster=True
    )
    obj._fit_for_tests(sample_X, 4)
    return obj, payload


class TestCoreSGHierarchy:
    def test_public_methods_raise_before_fit(self, core_sg_module):
        obj = core_sg_module.CoreSG()

        with pytest.raises(AttributeError, match="CoreSG is not fitted yet"):
            obj.get_core_distance(2)
        with pytest.raises(AttributeError, match="CoreSG is not fitted yet"):
            obj.get_core_sg_mutual_reachability_distance(2)
        with pytest.raises(AttributeError, match="CoreSG is not fitted yet"):
            obj.extract_mst_from_core_sg(2)
        with pytest.raises(AttributeError, match="CoreSG is not fitted yet"):
            obj.extract_hierarchy_from_core_sg(2)

    def test_get_core_distance_returns_requested_column(self, fitted_obj):
        obj, payload = fitted_obj
        expected = payload[2][:, 1]

        result = obj.get_core_distance(2)

        assert np.array_equal(result, expected)

    def test_get_core_sg_mutual_reachability_distance_delegates_to_helper(
        self, core_sg_module, fitted_obj, monkeypatch
    ):
        obj, _ = fitted_obj
        sentinel = np.array([[0.0, 1.0, 3.14]])

        def fake_core_sg_mutual_reachability_distance(
            core_sg, metric_edges, core_k_list, n_nodes, k_max, k
        ):
            assert core_sg is obj.support_graph_
            assert metric_edges is obj.metric_edges_
            assert core_k_list is obj.core_distances_
            assert n_nodes == obj.n_samples_
            assert k_max == obj.k_max_
            assert k == 3
            return sentinel

        monkeypatch.setattr(
            core_sg_module,
            "core_sg_mutual_reachability_distance",
            fake_core_sg_mutual_reachability_distance,
        )

        result = obj.get_core_sg_mutual_reachability_distance(3)

        assert result is sentinel

    def test_extract_hierarchy_for_smaller_k_updates_current_attributes(
        self, core_sg_module, fitted_obj, monkeypatch
    ):
        obj, _ = fitted_obj
        mst_small = np.array(
            [
                [0, 1, 1.0],
                [1, 2, 1.4],
                [2, 3, 6.4],
                [3, 4, 1.0],
                [4, 5, 1.4],
            ],
            dtype=np.float64,
        )
        slt_small = mst_small + 0.5
        condensed = np.column_stack(
            [
                np.arange(obj.n_samples_),
                np.arange(obj.n_samples_),
                np.ones(obj.n_samples_),
                np.full(obj.n_samples_, 2.0),
            ]
        ).astype(np.float64)
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        probabilities = np.linspace(0.5, 1.0, obj.n_samples_)
        persistence = np.array([0.55, 0.88], dtype=np.float64)

        monkeypatch.setattr(
            core_sg_module, "mst_from_core_sg", lambda **kwargs: mst_small
        )
        monkeypatch.setattr(
            core_sg_module, "mst_to_single_linkage_tree", lambda mst: slt_small
        )

        seen = {}

        def fake_tree_to_labels(instance, single_linkage_tree, min_spanning_tree):
            seen["kwargs"] = instance._get_tree_to_labels_kwargs()
            assert np.array_equal(single_linkage_tree, slt_small)
            assert np.array_equal(min_spanning_tree, mst_small)
            return (
                labels,
                probabilities,
                persistence,
                condensed,
                single_linkage_tree,
                min_spanning_tree,
            )

        monkeypatch.setattr(core_sg_module, "tree_to_labels", fake_tree_to_labels)

        obj.extract_hierarchy_from_core_sg(3)

        assert seen["kwargs"] == {
            "cluster_selection_method": "leaf",
            "allow_single_cluster": True,
        }
        assert np.array_equal(obj.labels_, labels)
        assert np.array_equal(obj.probabilities_, probabilities)
        assert np.array_equal(obj.cluster_persistence_, persistence)
        assert np.array_equal(obj._condensed_tree_array_, condensed)
        assert np.array_equal(obj._single_linkage_tree_array_, slt_small)
        assert np.array_equal(obj._min_spanning_tree_array_, mst_small)

        assert obj.minimum_spanning_tree_.to_pandas().shape[0] == obj.n_samples_ - 1
        assert obj.single_linkage_tree_.to_pandas().shape[0] == obj.n_samples_ - 1
        assert obj.condensed_tree_.to_pandas().shape[0] >= obj.n_samples_

    def test_extract_hierarchy_for_k_max_reuses_saved_fit_outputs(self, fitted_obj):
        obj, _ = fitted_obj

        obj.extract_hierarchy_from_core_sg(obj.k_max_)

        assert np.array_equal(obj.labels_, obj.labels_k_max_)
        assert np.array_equal(obj.probabilities_, obj.probabilities_k_max_)
        assert np.array_equal(obj.cluster_persistence_, obj.cluster_persistence_k_max_)
        assert np.array_equal(
            obj._condensed_tree_array_, obj._condensed_tree_k_max_array_
        )
        assert np.array_equal(
            obj._single_linkage_tree_array_, obj._single_linkage_tree_k_max_array_
        )
        assert np.array_equal(
            obj._min_spanning_tree_array_, obj._min_spanning_tree_k_max_array_
        )

    def test_extracting_smaller_k_updates_current_artifacts_without_mutating_k_max(
        self, core_sg_module, fitted_obj, monkeypatch
    ):
        obj, _ = fitted_obj
        labels_k_max = obj.labels_k_max_.copy()
        condensed_k_max = obj._condensed_tree_k_max_array_.copy()
        single_linkage_k_max = obj._single_linkage_tree_k_max_array_.copy()
        mst_k_max = obj._min_spanning_tree_k_max_array_.copy()

        obj.extract_hierarchy_from_core_sg(obj.k_max_)
        first_current_mst = obj._min_spanning_tree_array_.copy()

        def fake_mst_from_core_sg(
            core_sg, metric_edges, core_k_list, n_nodes, k, verbose, progress_callback
        ):
            del core_sg, metric_edges, core_k_list, n_nodes, verbose
            del progress_callback
            return np.array(
                [
                    [0, 1, float(k)],
                    [1, 2, float(k) + 0.1],
                    [2, 3, float(k) + 0.2],
                    [3, 4, float(k) + 0.3],
                    [4, 5, float(k) + 0.4],
                ],
                dtype=np.float64,
            )

        def fake_tree_to_labels(instance, single_linkage_tree, min_spanning_tree):
            k_marker = int(min_spanning_tree[0, 2])
            condensed = np.column_stack(
                [
                    np.arange(instance.n_samples_),
                    np.arange(instance.n_samples_),
                    np.full(instance.n_samples_, float(k_marker)),
                    np.full(instance.n_samples_, 2.0),
                ]
            ).astype(np.float64)
            return (
                np.full(instance.n_samples_, k_marker, dtype=np.int64),
                np.full(instance.n_samples_, 1.0 / k_marker, dtype=np.float64),
                np.array([float(k_marker)], dtype=np.float64),
                condensed,
                single_linkage_tree,
                min_spanning_tree,
            )

        monkeypatch.setattr(core_sg_module, "mst_from_core_sg", fake_mst_from_core_sg)
        monkeypatch.setattr(
            core_sg_module,
            "mst_to_single_linkage_tree",
            lambda mst: np.asarray(mst, dtype=np.float64) + 10.0,
        )
        monkeypatch.setattr(core_sg_module, "tree_to_labels", fake_tree_to_labels)

        obj.extract_hierarchy_from_core_sg(3)
        second_current_mst = obj._min_spanning_tree_array_.copy()
        obj.extract_hierarchy_from_core_sg(2)

        assert np.array_equal(obj.labels_k_max_, labels_k_max)
        assert np.array_equal(obj._condensed_tree_k_max_array_, condensed_k_max)
        assert np.array_equal(
            obj._single_linkage_tree_k_max_array_, single_linkage_k_max
        )
        assert np.array_equal(obj._min_spanning_tree_k_max_array_, mst_k_max)
        assert np.array_equal(first_current_mst, mst_k_max)
        assert not np.array_equal(second_current_mst, obj._min_spanning_tree_array_)
        assert np.array_equal(obj.labels_, np.full(obj.n_samples_, 2))

    def test_extract_hierarchy_uses_default_c_for_noise_handler(
        self, core_sg_module, fitted_obj, monkeypatch
    ):
        obj, _ = fitted_obj
        labels = np.array([0, -1, 0, 1, 1, 1], dtype=np.int64)

        monkeypatch.setattr(
            core_sg_module,
            "tree_to_labels",
            lambda instance, single_linkage_tree, min_spanning_tree: (
                labels.copy(),
                np.linspace(0.5, 1.0, obj.n_samples_),
                np.array([0.6, 0.8], dtype=np.float64),
                np.column_stack(
                    [
                        np.arange(obj.n_samples_),
                        np.arange(obj.n_samples_),
                        np.ones(obj.n_samples_),
                        np.full(obj.n_samples_, 2.0),
                    ]
                ).astype(np.float64),
                single_linkage_tree,
                min_spanning_tree,
            ),
        )
        monkeypatch.setattr(
            core_sg_module,
            "mst_to_single_linkage_tree",
            lambda mst: np.asarray(mst, dtype=np.float64),
        )

        seen = {}

        class DummyHandler:
            def reassign(self, *, labels, min_spanning_tree, n_samples):
                seen["labels"] = labels.copy()
                seen["mst"] = np.asarray(min_spanning_tree)
                seen["n_samples"] = n_samples
                return np.where(labels == -1, 9, labels)

        def fake_build_noise_handler(strategy, *, c):
            seen["strategy"] = strategy
            seen["c"] = c
            return DummyHandler()

        monkeypatch.setattr(
            core_sg_module, "build_noise_handler", fake_build_noise_handler
        )

        obj.extract_hierarchy_from_core_sg(3)

        assert seen["strategy"] == "mst_label_propagation"
        assert seen["c"] == 5
        assert seen["n_samples"] == obj.n_samples_
        assert np.array_equal(seen["labels"], labels)
        assert np.array_equal(obj.labels_, np.array([0, 9, 0, 1, 1, 1]))

    def test_extract_hierarchy_k_max_can_apply_noise_handler(
        self, core_sg_module, fitted_obj, monkeypatch
    ):
        obj, _ = fitted_obj
        obj.labels_k_max_ = np.array([0, -1, 0, 1, 1, 1], dtype=np.int64)
        obj.probabilities_k_max_ = np.linspace(0.5, 1.0, obj.n_samples_)
        obj.cluster_persistence_k_max_ = np.array([0.6, 0.8], dtype=np.float64)

        seen = {}

        class DummyHandler:
            def reassign(self, *, labels, min_spanning_tree, n_samples):
                seen["labels"] = labels.copy()
                seen["n_samples"] = n_samples
                return np.where(labels == -1, 5, labels)

        monkeypatch.setattr(
            core_sg_module,
            "build_noise_handler",
            lambda strategy, *, c: DummyHandler(),
        )

        obj.extract_hierarchy_from_core_sg(obj.k_max_)

        assert seen["n_samples"] == obj.n_samples_
        assert np.array_equal(seen["labels"], np.array([0, -1, 0, 1, 1, 1]))
        assert np.array_equal(obj.labels_, np.array([0, 5, 0, 1, 1, 1]))

    def test_extract_hierarchy_skips_noise_handler_when_disabled(
        self, core_sg_module, fitted_obj, monkeypatch
    ):
        obj, _ = fitted_obj
        obj.no_noise = False
        labels = np.array([0, -1, 0, 1, 1, 1], dtype=np.int64)

        monkeypatch.setattr(
            core_sg_module,
            "tree_to_labels",
            lambda instance, single_linkage_tree, min_spanning_tree: (
                labels.copy(),
                np.linspace(0.5, 1.0, obj.n_samples_),
                np.array([0.6, 0.8], dtype=np.float64),
                np.column_stack(
                    [
                        np.arange(obj.n_samples_),
                        np.arange(obj.n_samples_),
                        np.ones(obj.n_samples_),
                        np.full(obj.n_samples_, 2.0),
                    ]
                ).astype(np.float64),
                single_linkage_tree,
                min_spanning_tree,
            ),
        )
        monkeypatch.setattr(
            core_sg_module,
            "mst_to_single_linkage_tree",
            lambda mst: np.asarray(mst, dtype=np.float64),
        )

        def fail_build_noise_handler(*args, **kwargs):
            raise AssertionError("noise handler should not be built")

        monkeypatch.setattr(
            core_sg_module, "build_noise_handler", fail_build_noise_handler
        )

        obj.extract_hierarchy_from_core_sg(3)

        assert np.array_equal(obj.labels_, labels)

    def test_extract_hierarchy_skips_noise_handler_when_no_noise_labels_exist(
        self, core_sg_module, fitted_obj, monkeypatch
    ):
        obj, _ = fitted_obj
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

        monkeypatch.setattr(
            core_sg_module,
            "tree_to_labels",
            lambda instance, single_linkage_tree, min_spanning_tree: (
                labels.copy(),
                np.linspace(0.5, 1.0, obj.n_samples_),
                np.array([0.6, 0.8], dtype=np.float64),
                np.column_stack(
                    [
                        np.arange(obj.n_samples_),
                        np.arange(obj.n_samples_),
                        np.ones(obj.n_samples_),
                        np.full(obj.n_samples_, 2.0),
                    ]
                ).astype(np.float64),
                single_linkage_tree,
                min_spanning_tree,
            ),
        )
        monkeypatch.setattr(
            core_sg_module,
            "mst_to_single_linkage_tree",
            lambda mst: np.asarray(mst, dtype=np.float64),
        )

        def fail_build_noise_handler(*args, **kwargs):
            raise AssertionError("noise handler should not be built")

        monkeypatch.setattr(
            core_sg_module, "build_noise_handler", fail_build_noise_handler
        )

        obj.extract_hierarchy_from_core_sg(3)

        assert np.array_equal(obj.labels_, labels)

    def test_extract_hierarchy_noise_handler_only_updates_labels(
        self, core_sg_module, fitted_obj, monkeypatch
    ):
        obj, _ = fitted_obj
        labels = np.array([0, -1, 0, 1, 1, 1], dtype=np.int64)
        probabilities = np.linspace(0.5, 1.0, obj.n_samples_)
        persistence = np.array([0.55, 0.88], dtype=np.float64)
        mst_small = np.array(
            [
                [0, 1, 1.0],
                [1, 2, 1.4],
                [2, 3, 6.4],
                [3, 4, 1.0],
                [4, 5, 1.4],
            ],
            dtype=np.float64,
        )
        condensed = np.column_stack(
            [
                np.arange(obj.n_samples_),
                np.arange(obj.n_samples_),
                np.ones(obj.n_samples_),
                np.full(obj.n_samples_, 2.0),
            ]
        ).astype(np.float64)

        monkeypatch.setattr(
            core_sg_module, "mst_from_core_sg", lambda **kwargs: mst_small
        )
        monkeypatch.setattr(
            core_sg_module, "mst_to_single_linkage_tree", lambda mst: mst
        )
        monkeypatch.setattr(
            core_sg_module,
            "tree_to_labels",
            lambda instance, single_linkage_tree, min_spanning_tree: (
                labels.copy(),
                probabilities.copy(),
                persistence.copy(),
                condensed.copy(),
                single_linkage_tree.copy(),
                min_spanning_tree.copy(),
            ),
        )

        class DummyHandler:
            def reassign(self, *, labels, min_spanning_tree, n_samples):
                return np.where(labels == -1, 7, labels)

        monkeypatch.setattr(
            core_sg_module,
            "build_noise_handler",
            lambda strategy, *, c: DummyHandler(),
        )

        obj.extract_hierarchy_from_core_sg(3, c=3)

        assert np.array_equal(obj.labels_, np.array([0, 7, 0, 1, 1, 1]))
        assert np.array_equal(obj.probabilities_, probabilities)
        assert np.array_equal(obj.cluster_persistence_, persistence)
        assert np.array_equal(obj._condensed_tree_array_, condensed)
        assert np.array_equal(obj._single_linkage_tree_array_, mst_small)
        assert np.array_equal(obj._min_spanning_tree_array_, mst_small)

    def test_extract_mst_from_core_sg_uses_recomputed_path_for_smaller_k(
        self, core_sg_module, fitted_obj, monkeypatch
    ):
        obj, _ = fitted_obj
        mst_small = np.array(
            [[0, 1, 1.0], [1, 2, 1.2], [2, 3, 2.0], [3, 4, 1.1], [4, 5, 1.2]],
            dtype=np.float64,
        )

        def fake_mst_from_core_sg(
            core_sg, metric_edges, core_k_list, n_nodes, k, verbose, progress_callback
        ):
            assert core_sg is obj.support_graph_
            assert metric_edges is obj.metric_edges_
            assert core_k_list is obj.core_distances_
            assert n_nodes == obj.n_samples_
            assert k == 3
            assert verbose == obj.verbose
            assert progress_callback == obj.progress_callback
            return mst_small

        monkeypatch.setattr(core_sg_module, "mst_from_core_sg", fake_mst_from_core_sg)

        result = obj.extract_mst_from_core_sg(3)

        assert np.array_equal(result, mst_small)
        assert isinstance(result, np.ndarray)
        assert result.shape == (obj.n_samples_ - 1, 3)
        assert np.isfinite(result[:, 2]).all()

    def test_extract_mst_from_core_sg_smaller_k_to_dataframe(
        self, core_sg_module, fitted_obj, monkeypatch
    ):
        obj, _ = fitted_obj
        mst_small = np.array(
            [[0, 1, 1.0], [1, 2, 1.2], [2, 3, 2.0], [3, 4, 1.1], [4, 5, 1.2]],
            dtype=np.float64,
        )
        monkeypatch.setattr(
            core_sg_module, "mst_from_core_sg", lambda *args, **kwargs: mst_small
        )

        df = obj.extract_mst_from_core_sg(3, toDF=True)

        assert list(df.columns) == ["to", "from", "weight"]
        assert str(df["to"].dtype) == "int64"
        assert str(df["from"].dtype) == "int64"
        assert str(df["weight"].dtype) == "float64"
        assert df.shape == (obj.n_samples_ - 1, 3)

    def test_score_sg_disconnected_mst_error_is_reworded(
        self, core_sg_module, fitted_obj, monkeypatch
    ):
        obj, _ = fitted_obj
        obj.algorithm = "score-sg"

        def raise_disconnected(**kwargs):
            del kwargs
            raise ValueError("Disconex Graph")

        monkeypatch.setattr(core_sg_module, "mst_from_core_sg", raise_disconnected)

        with pytest.raises(ValueError, match="support graph is disconnected"):
            obj.extract_mst_from_core_sg(3)

    def test_extract_mst_reraises_unrelated_value_errors(
        self, core_sg_module, fitted_obj, monkeypatch
    ):
        obj, _ = fitted_obj
        obj.algorithm = "core-sg"

        def raise_unrelated_error(*args, **kwargs):
            del args, kwargs
            raise ValueError("plain failure")

        monkeypatch.setattr(core_sg_module, "mst_from_core_sg", raise_unrelated_error)

        with pytest.raises(ValueError, match="plain failure"):
            obj.extract_mst_from_core_sg(3)

    def test_standalone_mst_and_reweight_helpers_validate_k(self, core_sg_module):
        core_sg = np.array([[0.0, 1.0, -1.0]], dtype=np.float64)
        metric_edges = np.array([[1.0, 0.0, 1.0]], dtype=np.float64)
        core_k_list = np.ones((2, 2), dtype=np.float64)

        with pytest.raises(ValueError, match="1 <= k_max <= n-1"):
            core_sg_module.mst_from_core_sg(
                core_sg, metric_edges, core_k_list, n_nodes=2, k=0
            )
        with pytest.raises(ValueError, match="reproduzir min_samples"):
            core_sg_module.mst_from_core_sg(
                core_sg, metric_edges, core_k_list, n_nodes=2, k=1
            )
        with pytest.raises(ValueError, match="1 <= k <= k_max"):
            core_sg_module.core_sg_mutual_reachability_distance(
                core_sg, metric_edges, core_k_list, n_nodes=2, k_max=2, k=0
            )
        with pytest.raises(ValueError, match="k must be >= 2"):
            core_sg_module.core_sg_mutual_reachability_distance(
                core_sg, metric_edges, core_k_list, n_nodes=2, k_max=2, k=1
            )

    def test_extract_hierarchy_rejects_invalid_k_values(self, fitted_obj):
        obj, _ = fitted_obj

        with pytest.raises(ValueError, match="k invalid"):
            obj.get_core_distance(0)
        with pytest.raises(ValueError, match="k must be >= 2"):
            obj.get_core_distance(1)
        with pytest.raises(ValueError, match="k invalid"):
            obj.get_core_distance(obj.k_max_ + 1)

        with pytest.raises(ValueError, match="k invalid"):
            obj.get_core_sg_mutual_reachability_distance(0)

        with pytest.raises(ValueError, match="k must be >= 2"):
            obj.get_core_sg_mutual_reachability_distance(1)

        with pytest.raises(ValueError, match="k invalid"):
            obj.extract_mst_from_core_sg(0)
        with pytest.raises(ValueError, match="k must be >= 2"):
            obj.extract_mst_from_core_sg(1)
        with pytest.raises(ValueError, match="k invalid"):
            obj.extract_mst_from_core_sg(obj.k_max_ + 1)

        with pytest.raises(ValueError, match="k invalid"):
            obj.extract_hierarchy_from_core_sg(0)
        with pytest.raises(ValueError, match="k must be >= 2"):
            obj.extract_hierarchy_from_core_sg(1)
        with pytest.raises(ValueError, match="k invalid"):
            obj.extract_hierarchy_from_core_sg(obj.k_max_ + 1)

    def test_extract_hierarchy_rejects_invalid_c_values(
        self, core_sg_module, fitted_obj, monkeypatch
    ):
        obj, _ = fitted_obj
        monkeypatch.setattr(
            core_sg_module,
            "tree_to_labels",
            lambda instance, single_linkage_tree, min_spanning_tree: (
                np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
                np.linspace(0.5, 1.0, obj.n_samples_),
                np.array([0.6, 0.8], dtype=np.float64),
                np.column_stack(
                    [
                        np.arange(obj.n_samples_),
                        np.arange(obj.n_samples_),
                        np.ones(obj.n_samples_),
                        np.full(obj.n_samples_, 2.0),
                    ]
                ).astype(np.float64),
                single_linkage_tree,
                min_spanning_tree,
            ),
        )
        monkeypatch.setattr(
            core_sg_module, "mst_to_single_linkage_tree", lambda mst: mst
        )

        with pytest.raises(ValueError, match="greater than or equal to 1"):
            obj.extract_hierarchy_from_core_sg(3, c=0)
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            obj.extract_hierarchy_from_core_sg(3, c=1.5)

    def test_minimum_spanning_tree_wrapper_warns_without_raw_data(
        self, core_sg_module, fitted_obj, monkeypatch
    ):
        obj, _ = fitted_obj
        mst_small = np.array(
            [[0, 1, 1.0], [1, 2, 1.2], [2, 3, 2.0], [3, 4, 1.1], [4, 5, 1.2]],
            dtype=np.float64,
        )

        monkeypatch.setattr(
            core_sg_module, "mst_from_core_sg", lambda **kwargs: mst_small
        )
        monkeypatch.setattr(
            core_sg_module, "mst_to_single_linkage_tree", lambda mst: mst
        )
        monkeypatch.setattr(
            core_sg_module,
            "tree_to_labels",
            lambda instance, single_linkage_tree, min_spanning_tree: (
                np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
                np.linspace(0.5, 1.0, obj.n_samples_),
                np.array([0.6, 0.8], dtype=np.float64),
                np.column_stack(
                    [
                        np.arange(obj.n_samples_),
                        np.arange(obj.n_samples_),
                        np.ones(obj.n_samples_),
                        np.full(obj.n_samples_, 2.0),
                    ]
                ).astype(np.float64),
                single_linkage_tree,
                min_spanning_tree,
            ),
        )

        obj.extract_hierarchy_from_core_sg(3)
        obj._raw_data_ = None

        with pytest.warns(UserWarning, match="No raw data is available"):
            wrapped = obj.minimum_spanning_tree_

        assert wrapped is None
