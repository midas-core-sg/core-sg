from __future__ import annotations

import importlib

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from tests.helpers import make_fit_payload

pytestmark = pytest.mark.unit


class FakeCoreSG:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_calls = []
        self.extract_calls = []
        self.hierarchy_by_k = {}
        FakeCoreSG.instances.append(self)

    def fit(self, X, k_max):
        self.fit_calls.append((X, k_max))
        self.n_samples_ = X.shape[0]
        payload = make_fit_payload(n=X.shape[0], k_max=k_max)
        self._raw_data_ = X
        self.k_max_ = k_max
        self.labels_k_max_ = payload[4].labels_
        self.probabilities_k_max_ = payload[4].probabilities_
        self.cluster_persistence_k_max_ = payload[4].cluster_persistence_
        self._condensed_tree_k_max_array_ = payload[4]._condensed_tree
        self._single_linkage_tree_k_max_array_ = payload[4]._single_linkage_tree
        self._min_spanning_tree_k_max_array_ = payload[4]._min_spanning_tree
        return self

    def extract_hierarchy_from_core_sg(self, k):
        self.extract_calls.append(k)
        base = np.arange(self.n_samples_, dtype=np.int64)
        self.labels_ = base + (100 * k)
        self.probabilities_ = np.full(self.n_samples_, 1.0 / k)
        self.cluster_persistence_ = np.array([float(k)])
        self.condensed_tree_ = f"condensed-{k}"
        self.single_linkage_tree_ = f"single-{k}"
        self.minimum_spanning_tree_ = f"mst-{k}"


@pytest.fixture()
def estimators_module(fake_hdbscan_modules, monkeypatch):
    module = importlib.import_module("core_sg.estimators")
    FakeCoreSG.instances = []
    monkeypatch.setattr(module, "CoreSG", FakeCoreSG)
    return module


class TestCoreSGClusterer:
    def test_fit_returns_self_and_exposes_requested_k_artifacts(
        self, estimators_module, sample_X
    ):
        clusterer = estimators_module.CoreSGClusterer(
            k_max=4,
            metric="euclidean",
            no_noise=False,
            match_reference_implementation=True,
        )

        returned = clusterer.fit(sample_X, k=3)

        assert returned is clusterer
        assert clusterer.k_max_ == 4
        assert clusterer.k_ == 3
        assert clusterer.core_sg_ is FakeCoreSG.instances[0]
        assert np.array_equal(clusterer.labels_, np.arange(sample_X.shape[0]) + 300)
        assert np.array_equal(clusterer.labels_, clusterer.core_sg_.labels_)
        assert np.array_equal(
            clusterer.probabilities_, clusterer.core_sg_.probabilities_
        )
        assert np.array_equal(
            clusterer.cluster_persistence_, clusterer.core_sg_.cluster_persistence_
        )
        assert clusterer.condensed_tree_ == "condensed-3"
        assert clusterer.single_linkage_tree_ == "single-3"
        assert clusterer.minimum_spanning_tree_ == "mst-3"
        assert clusterer.core_sg_.fit_calls[0][1] == 4
        assert clusterer.core_sg_.extract_calls == [3]

    def test_fit_with_k_none_uses_k_max(self, estimators_module, sample_X):
        clusterer = estimators_module.CoreSGClusterer(k_max=4)

        clusterer.fit(sample_X, k=None)

        assert clusterer.k_ == 4
        assert clusterer.core_sg_.extract_calls == [4]
        assert np.array_equal(clusterer.labels_, np.arange(sample_X.shape[0]) + 400)

    def test_second_fit_reuses_existing_core_sg_and_extracts_only(
        self, estimators_module, sample_X
    ):
        clusterer = estimators_module.CoreSGClusterer(k_max=4)

        clusterer.fit(sample_X, k=3)
        core_sg = clusterer.core_sg_
        clusterer.fit(sample_X.copy(), k=2)

        assert clusterer.core_sg_ is core_sg
        assert FakeCoreSG.instances == [core_sg]
        assert len(core_sg.fit_calls) == 1
        assert core_sg.fit_calls[0][1] == 4
        assert core_sg.extract_calls == [3, 2]
        assert clusterer.k_max_ == 4
        assert clusterer.k_ == 2
        assert np.array_equal(clusterer.labels_, np.arange(sample_X.shape[0]) + 200)

    def test_second_fit_does_not_rebuild_after_set_params_changes_k_max(
        self, estimators_module, sample_X
    ):
        clusterer = estimators_module.CoreSGClusterer(k_max=4)
        clusterer.fit(sample_X, k=3)
        core_sg = clusterer.core_sg_

        clusterer.set_params(k_max=5)
        clusterer.fit(sample_X, k=2)

        assert clusterer.core_sg_ is core_sg
        assert len(FakeCoreSG.instances) == 1
        assert core_sg.fit_calls[0][1] == 4
        assert core_sg.extract_calls == [3, 2]
        assert clusterer.k_max == 5
        assert clusterer.k_max_ == 4

    def test_second_fit_uses_existing_core_sg_even_if_build_params_change(
        self, estimators_module, sample_X
    ):
        clusterer = estimators_module.CoreSGClusterer(k_max=4)
        clusterer.fit(sample_X, k=3)
        core_sg = clusterer.core_sg_

        clusterer.set_params(metric="precomputed")
        clusterer.fit(None, k=2)

        assert clusterer.core_sg_ is core_sg
        assert len(FakeCoreSG.instances) == 1
        assert len(core_sg.fit_calls) == 1
        assert core_sg.extract_calls == [3, 2]
        assert clusterer.k_ == 2

    def test_fit_predict_returns_labels_attribute(self, estimators_module, sample_X):
        clusterer = estimators_module.CoreSGClusterer(k_max=4)

        labels = clusterer.fit_predict(sample_X, k=2)

        assert np.array_equal(labels, clusterer.labels_)
        assert clusterer.k_ == 2

    def test_fit_predict_after_fit_reuses_existing_core_sg(
        self, estimators_module, sample_X
    ):
        clusterer = estimators_module.CoreSGClusterer(k_max=4)
        clusterer.fit(sample_X, k=3)
        core_sg = clusterer.core_sg_

        labels = clusterer.fit_predict(sample_X.copy(), k=2)

        assert clusterer.core_sg_ is core_sg
        assert FakeCoreSG.instances == [core_sg]
        assert len(core_sg.fit_calls) == 1
        assert core_sg.extract_calls == [3, 2]
        assert np.array_equal(labels, clusterer.labels_)

    def test_constructor_parameters_are_forwarded_without_mutating_dict(
        self, estimators_module, sample_X
    ):
        approx_knn_kwargs = {"n_trees": 8}
        clusterer = estimators_module.CoreSGClusterer(
            k_max=4,
            metric="minkowski",
            p=3,
            algorithm="score-sg",
            no_noise=False,
            noise_label_strategy="mst_label_propagation",
            random_state=13,
            approx_knn_kwargs=approx_knn_kwargs,
            verbose=1,
            cluster_selection_method="leaf",
            allow_single_cluster=True,
            match_reference_implementation=True,
            cluster_selection_epsilon=0.1,
            cluster_selection_persistence=0.2,
            max_cluster_size=9,
            cluster_selection_epsilon_max=2.5,
        )

        clusterer.fit(sample_X, k=3)

        kwargs = clusterer.core_sg_.kwargs
        assert kwargs == {
            "metric": "minkowski",
            "p": 3,
            "algorithm": "score-sg",
            "no_noise": False,
            "noise_label_strategy": "mst_label_propagation",
            "random_state": 13,
            "approx_knn_kwargs": {"n_trees": 8},
            "verbose": 1,
            "progress_callback": None,
            "cluster_selection_method": "leaf",
            "allow_single_cluster": True,
            "match_reference_implementation": True,
            "cluster_selection_epsilon": 0.1,
            "cluster_selection_persistence": 0.2,
            "max_cluster_size": 9,
            "cluster_selection_epsilon_max": 2.5,
        }
        assert kwargs["approx_knn_kwargs"] is not approx_knn_kwargs
        assert approx_knn_kwargs == {"n_trees": 8}

    def test_get_params_set_params_and_clone(self, estimators_module):
        clusterer = estimators_module.CoreSGClusterer(k_max=4, metric="euclidean")

        params = clusterer.get_params()
        assert params["k_max"] == 4
        assert params["metric"] == "euclidean"

        clusterer.set_params(k_max=5, metric="manhattan")
        assert clusterer.k_max == 5
        assert clusterer.metric == "manhattan"

        cloned = clone(clusterer)
        assert cloned.get_params() == clusterer.get_params()
        assert not hasattr(cloned, "core_sg_")
        assert not hasattr(cloned, "labels_")

    def test_clone_fitted_clusterer_drops_learned_attributes(
        self, estimators_module, sample_X
    ):
        clusterer = estimators_module.CoreSGClusterer(k_max=4)
        clusterer.fit(sample_X, k=3)

        cloned = clone(clusterer)

        assert cloned.get_params() == clusterer.get_params()
        for attribute in [
            "core_sg_",
            "k_max_",
            "k_",
            "labels_",
            "probabilities_",
            "cluster_persistence_",
            "condensed_tree_",
            "single_linkage_tree_",
            "minimum_spanning_tree_",
        ]:
            assert not hasattr(cloned, attribute)

    def test_get_fitted_core_sg_returns_native_object(
        self, estimators_module, sample_X
    ):
        clusterer = estimators_module.CoreSGClusterer(k_max=4)
        clusterer.fit(sample_X, k=3)

        assert clusterer.get_fitted_core_sg() is clusterer.core_sg_

    def test_get_fitted_core_sg_raises_before_fit(self, estimators_module):
        clusterer = estimators_module.CoreSGClusterer(k_max=4)

        with pytest.raises(NotFittedError):
            clusterer.get_fitted_core_sg()

        with pytest.raises(AttributeError):
            _ = clusterer.labels_

    @pytest.mark.parametrize("k_max", [1, 0, -1, 6, 1.5, True])
    def test_invalid_k_max_values_raise_clear_value_error(
        self, estimators_module, sample_X, k_max
    ):
        clusterer = estimators_module.CoreSGClusterer(k_max=k_max)

        with pytest.raises(ValueError, match="k_max"):
            clusterer.fit(sample_X)

    @pytest.mark.parametrize("k", [1, 0, -1, 5, 2.5, True, False])
    def test_invalid_k_values_raise_clear_value_error(
        self, estimators_module, sample_X, k
    ):
        clusterer = estimators_module.CoreSGClusterer(k_max=4)

        with pytest.raises(ValueError, match="k"):
            clusterer.fit(sample_X, k=k)

    @pytest.mark.parametrize(
        "bad_X",
        [
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.array([[0.0, 1.0]], dtype=np.float64),
            np.array([[0.0, 1.0], [np.nan, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([[0.0, 1.0], [np.inf, 2.0], [3.0, 4.0]], dtype=np.float64),
        ],
    )
    def test_first_fit_rejects_invalid_dense_X(self, estimators_module, bad_X):
        clusterer = estimators_module.CoreSGClusterer(k_max=2)

        with pytest.raises(ValueError):
            clusterer.fit(bad_X)

    def test_first_fit_rejects_sparse_X(self, estimators_module, sample_X):
        sparse = pytest.importorskip("scipy.sparse")
        clusterer = estimators_module.CoreSGClusterer(k_max=4)

        with pytest.raises(TypeError):
            clusterer.fit(sparse.csr_matrix(sample_X))

    def test_later_fit_skips_X_validation_but_still_validates_k(
        self, estimators_module, sample_X
    ):
        clusterer = estimators_module.CoreSGClusterer(k_max=4)
        clusterer.fit(sample_X, k=3)
        core_sg = clusterer.core_sg_
        invalid_X = np.array([np.nan], dtype=np.float64)

        clusterer.fit(invalid_X, k=None)

        assert clusterer.core_sg_ is core_sg
        assert len(core_sg.fit_calls) == 1
        assert core_sg.extract_calls == [3, 4]
        assert clusterer.k_ == clusterer.k_max_ == 4

        with pytest.raises(ValueError, match="k"):
            clusterer.fit(invalid_X, k=5)

    def test_precomputed_metric_is_rejected_in_wrapper(
        self, estimators_module, sample_X
    ):
        clusterer = estimators_module.CoreSGClusterer(k_max=4, metric="precomputed")

        with pytest.raises(ValueError, match="precomputed"):
            clusterer.fit(sample_X)

    def test_validate_x_falls_back_when_validate_data_is_unavailable(
        self, estimators_module, monkeypatch
    ):
        monkeypatch.setattr(estimators_module, "validate_data", None)
        clusterer = estimators_module.CoreSGClusterer(k_max=2)

        X_checked = clusterer._validate_X([[0.0, 1.0], [1.0, 0.0]])

        assert X_checked.dtype == np.float64
        assert clusterer.n_features_in_ == 2
