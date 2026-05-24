from __future__ import annotations

import numpy as np
import pytest

hdbscan = pytest.importorskip("hdbscan")
from sklearn.datasets import make_blobs

from core_sg import CoreSG, CoreSGClusterer
from tests.helpers import (
    normalize_undirected_edges,
    validate_reference_edges_in_core,
    validate_reference_weights_in_core,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def dataset():
    X, _ = make_blobs(n_samples=60, n_features=3, centers=4, random_state=42)
    return X


def _reference_mst(D: np.ndarray, k: int) -> np.ndarray:
    ref = hdbscan.HDBSCAN(
        min_cluster_size=k,
        min_samples=k,
        metric="precomputed",
        algorithm="generic",
        approx_min_span_tree=False,
        gen_min_span_tree=True,
        match_reference_implementation=True,
    ).fit(D)
    mst = np.asarray(ref._min_spanning_tree, dtype=np.float64)
    return mst[np.argsort(mst[:, 2], kind="mergesort")]


def _assert_clusterer_matches_native_core(
    clusterer: CoreSGClusterer, core: CoreSG, requested_k: int
):
    assert np.array_equal(clusterer.labels_, core.labels_)
    assert np.allclose(clusterer.probabilities_, core.probabilities_)
    assert np.allclose(clusterer.cluster_persistence_, core.cluster_persistence_)
    assert np.allclose(
        clusterer.core_sg_._min_spanning_tree_array_,
        core._min_spanning_tree_array_,
    )
    assert np.allclose(
        clusterer.core_sg_._single_linkage_tree_array_,
        core._single_linkage_tree_array_,
    )
    assert np.array_equal(
        clusterer.core_sg_._condensed_tree_array_,
        core._condensed_tree_array_,
    )
    assert clusterer.k_ == requested_k
    assert clusterer.k_max_ == core.k_max_
    assert clusterer.core_sg_.k_max_ == core.k_max_


def _assert_smoke_artifacts(core: CoreSG, X: np.ndarray, *, k_max: int):
    assert core.n_samples_ == X.shape[0]
    assert core.k_max_ == k_max
    assert core.labels_ is not None
    assert core.labels_.shape == (X.shape[0],)
    assert core.probabilities_ is not None
    assert core.probabilities_.shape == (X.shape[0],)
    assert core.cluster_persistence_ is not None
    assert core.condensed_tree_.to_pandas().shape[0] >= X.shape[0]
    assert core.single_linkage_tree_.to_pandas().shape[0] == X.shape[0] - 1
    assert core.minimum_spanning_tree_.to_pandas().shape[0] == X.shape[0] - 1


class TestReferenceEquivalence:
    @pytest.mark.parametrize("k", [6, 4, 2])
    def test_reference_mst_edges_are_contained_in_core_sg(self, dataset, k):
        core = CoreSG(
            metric="euclidean", p=2, verbose=0, match_reference_implementation=True
        )
        core._fit_for_tests(dataset, 6)

        mst_hdb = _reference_mst(core.distance_matrix_, k)
        summary = validate_reference_edges_in_core(core.support_graph_, mst_hdb)

        assert summary.ok, (
            f"Missing {summary.missing} reference edges out of {summary.compared}."
        )

    @pytest.mark.parametrize("k", [6, 4, 2])
    def test_reference_mrd_weights_are_present_in_reweighted_core_sg(self, dataset, k):
        core = CoreSG(
            metric="euclidean", p=2, verbose=0, match_reference_implementation=True
        )
        core._fit_for_tests(dataset, 6)

        mst_hdb = _reference_mst(core.distance_matrix_, k)
        weighted_core = core.get_core_sg_mutual_reachability_distance(k)
        summary = validate_reference_weights_in_core(weighted_core, mst_hdb)

        assert summary.ok, (
            f"Missing {summary.missing} weighted reference edges out of "
            f"{summary.compared}."
        )

    @pytest.mark.parametrize("k", [6, 4, 2])
    def test_reference_and_core_sg_mst_are_reported_without_failing(
        self, dataset, k, capsys
    ):
        core = CoreSG(
            metric="euclidean", p=2, verbose=0, match_reference_implementation=True
        )
        core._fit_for_tests(dataset, 6)

        mst_hdb = _reference_mst(core.distance_matrix_, k)
        mst_core = core.extract_mst_from_core_sg(k)

        same_weighted_edges = normalize_undirected_edges(
            mst_hdb
        ) == normalize_undirected_edges(mst_core)
        print(f"k={k} | exact weighted match={same_weighted_edges}")

        captured = capsys.readouterr()
        assert f"k={k}" in captured.out
        assert "exact weighted match=" in captured.out


class TestIntegrationSmoke:
    def test_public_package_import_exposes_core_sg_class(self):
        from core_sg import CoreSG as ExportedCoreSG
        from core_sg import CoreSGClusterer as ExportedCoreSGClusterer

        assert ExportedCoreSG.__name__ == CoreSG.__name__ == "CoreSG"
        assert ExportedCoreSG.__module__ == CoreSG.__module__ == "core_sg.core_sg"
        assert (
            ExportedCoreSGClusterer.__name__
            == CoreSGClusterer.__name__
            == "CoreSGClusterer"
        )
        assert (
            ExportedCoreSGClusterer.__module__
            == CoreSGClusterer.__module__
            == "core_sg.estimators"
        )

    def test_core_sg_clusterer_fit_exposes_selected_k_outputs(self, dataset):
        clusterer = CoreSGClusterer(
            k_max=6,
            metric="euclidean",
            p=2,
            verbose=0,
            no_noise=False,
            match_reference_implementation=True,
        )

        returned = clusterer.fit(dataset, k=4)

        assert returned is clusterer
        assert clusterer.k_max_ == 6
        assert clusterer.k_ == 4
        assert clusterer.core_sg_.k_max_ == 6
        core_sg = clusterer.core_sg_
        assert np.array_equal(clusterer.labels_, clusterer.core_sg_.labels_)
        assert clusterer.condensed_tree_.to_pandas().shape[0] >= dataset.shape[0]
        assert clusterer.single_linkage_tree_.to_pandas().shape[0] == (
            dataset.shape[0] - 1
        )
        assert clusterer.minimum_spanning_tree_.to_pandas().shape[0] == (
            dataset.shape[0] - 1
        )

        labels = clusterer.fit_predict(dataset, k=2)

        assert clusterer.core_sg_ is core_sg
        assert clusterer.k_max_ == 6
        assert clusterer.k_ == 2
        assert np.array_equal(labels, clusterer.labels_)

    @pytest.mark.parametrize("k", [6, 4, None])
    def test_core_sg_clusterer_outputs_match_native_core_sg(self, dataset, k):
        k_max = 6
        requested_k = k_max if k is None else k
        clusterer = CoreSGClusterer(
            k_max=k_max,
            metric="euclidean",
            p=2,
            verbose=0,
            no_noise=False,
            match_reference_implementation=True,
        )
        core = CoreSG(
            metric="euclidean",
            p=2,
            verbose=0,
            no_noise=False,
            match_reference_implementation=True,
        )

        clusterer.fit(dataset, k=k)
        core.fit(dataset, k_max=k_max)
        core.extract_hierarchy_from_core_sg(requested_k)

        _assert_clusterer_matches_native_core(clusterer, core, requested_k)

    def test_repeated_core_sg_clusterer_fit_matches_native_for_each_k(self, dataset):
        clusterer = CoreSGClusterer(
            k_max=6,
            metric="euclidean",
            p=2,
            verbose=0,
            no_noise=False,
            match_reference_implementation=True,
        )

        clusterer.fit(dataset, k=6)
        core_sg = clusterer.core_sg_
        labels_k_max = clusterer.labels_.copy()
        clusterer.fit(dataset, k=4)
        labels_k4 = clusterer.labels_.copy()
        labels_from_fit_predict = clusterer.fit_predict(dataset, k=2)

        assert clusterer.core_sg_ is core_sg
        assert np.array_equal(labels_from_fit_predict, clusterer.labels_)

        for requested_k, observed_labels in [
            (6, labels_k_max),
            (4, labels_k4),
            (2, labels_from_fit_predict),
        ]:
            core = CoreSG(
                metric="euclidean",
                p=2,
                verbose=0,
                no_noise=False,
                match_reference_implementation=True,
            )
            core.fit(dataset, k_max=6)
            core.extract_hierarchy_from_core_sg(requested_k)
            assert np.array_equal(observed_labels, core.labels_)

    def test_full_fit_extract_hierarchy_and_wrapped_accessors_with_real_hdbscan(
        self, dataset
    ):
        core = CoreSG(
            metric="euclidean", p=2, verbose=0, match_reference_implementation=True
        )
        core._fit_for_tests(dataset, 6)
        core.extract_hierarchy_from_core_sg(4)

        assert core.labels_ is not None
        assert core.probabilities_ is not None
        assert core.cluster_persistence_ is not None
        assert core.condensed_tree_.to_pandas().shape[0] >= core.n_samples_
        assert core.single_linkage_tree_.to_pandas().shape[0] == core.n_samples_ - 1
        assert core.minimum_spanning_tree_.to_pandas().shape[0] == core.n_samples_ - 1

    def test_core_sg_smoke_runs_two_instances_with_different_parameters(self, dataset):
        first = CoreSG(
            metric="euclidean",
            p=2,
            verbose=0,
            no_noise=False,
            match_reference_implementation=True,
        )
        second = CoreSG(
            metric="manhattan",
            p=2,
            verbose=0,
            no_noise=False,
            cluster_selection_method="leaf",
            allow_single_cluster=True,
            match_reference_implementation=True,
        )

        first.fit(dataset, k_max=6)
        first.extract_hierarchy_from_core_sg(4)
        second.fit(dataset, k_max=5)
        second.extract_hierarchy_from_core_sg(3)

        _assert_smoke_artifacts(first, dataset, k_max=6)
        _assert_smoke_artifacts(second, dataset, k_max=5)
        assert first is not second
        assert first.metric == "euclidean"
        assert second.metric == "manhattan"
        assert first._raw_data_ is dataset
        assert second._raw_data_ is dataset
        assert not np.array_equal(
            first._min_spanning_tree_array_,
            second._min_spanning_tree_array_,
        )

    def test_score_sg_fit_extract_hierarchy_and_wrapped_accessors(self, dataset):
        connected_dataset, _ = make_blobs(
            n_samples=60,
            n_features=3,
            centers=1,
            cluster_std=1.0,
            random_state=42,
        )
        core = CoreSG(
            metric="euclidean",
            p=2,
            algorithm="score-sg",
            random_state=42,
            verbose=0,
            match_reference_implementation=True,
        )
        core._fit_for_tests(connected_dataset, 6)
        core.extract_hierarchy_from_core_sg(4)

        with pytest.raises(
            AttributeError,
            match="Attribute 'distance_matrix_' is available only when algorithm='core-sg'",
        ):
            _ = core.distance_matrix_
        assert core._tree_to_labels_data_.shape == connected_dataset.shape
        assert core.anti_hubs_ is not None
        assert core.anti_hubs_.ndim == 1
        assert core.anti_hubs_.shape[0] == int(
            np.floor(np.sqrt(connected_dataset.shape[0]))
        )
        assert core.labels_ is not None
        assert core.probabilities_ is not None
        assert core.cluster_persistence_ is not None
        assert core.condensed_tree_.to_pandas().shape[0] >= core.n_samples_
        assert core.single_linkage_tree_.to_pandas().shape[0] == core.n_samples_ - 1
        assert core.minimum_spanning_tree_.to_pandas().shape[0] == core.n_samples_ - 1

    def test_score_sg_smoke_runs_two_instances_with_different_parameters(self):
        connected_dataset, _ = make_blobs(
            n_samples=40,
            n_features=3,
            centers=1,
            cluster_std=0.75,
            random_state=123,
        )
        first = CoreSG(
            metric="euclidean",
            p=2,
            algorithm="score-sg",
            random_state=42,
            verbose=0,
            no_noise=False,
            match_reference_implementation=True,
        )
        second = CoreSG(
            metric="euclidean",
            p=2,
            algorithm="score-sg",
            random_state=7,
            approx_knn_kwargs={"n_trees": 8},
            verbose=0,
            no_noise=False,
            cluster_selection_method="leaf",
            allow_single_cluster=True,
            match_reference_implementation=True,
        )

        first.fit(connected_dataset, k_max=6)
        first.extract_hierarchy_from_core_sg(4)
        second.fit(connected_dataset, k_max=5)
        second.extract_hierarchy_from_core_sg(3)

        _assert_smoke_artifacts(first, connected_dataset, k_max=6)
        _assert_smoke_artifacts(second, connected_dataset, k_max=5)
        assert first is not second
        assert first.algorithm == second.algorithm == "score-sg"
        assert first.random_state == 42
        assert second.random_state == 7
        assert second.approx_knn_kwargs == {"n_trees": 8}
        assert first.anti_hubs_.shape == (
            int(np.floor(np.sqrt(connected_dataset.shape[0]))),
        )
        assert second.anti_hubs_.shape == first.anti_hubs_.shape

    def test_score_sg_fit_raises_explicitly_when_support_graph_is_disconnected(
        self, dataset
    ):
        core = CoreSG(
            metric="euclidean",
            p=2,
            algorithm="score-sg",
            random_state=42,
            verbose=0,
            match_reference_implementation=True,
        )

        with pytest.raises(ValueError, match="support graph is disconnected"):
            core._fit_for_tests(dataset, 6)
