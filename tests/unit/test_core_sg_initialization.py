from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestCoreSGInitialization:
    def test_init_sets_defaults_and_empty_state(self, core_sg_module):
        obj = core_sg_module.CoreSG()

        assert obj.metric == "euclidean"
        assert obj.p == 2
        assert obj.verbose == 0
        assert obj.no_noise is True
        assert obj.noise_label_strategy == "mst_label_propagation"
        assert obj.algorithm == "core-sg"
        assert obj.random_state is None
        assert obj.approx_knn_kwargs is None
        assert obj.hdbscan_kwargs == {}

        assert obj.n_samples_ is None
        assert obj.k_max_ is None
        assert obj.support_graph_ is None
        assert obj.metric_edges_ is None
        assert obj.core_distances_ is None
        assert obj.distance_matrix_ is None
        assert obj._tree_to_labels_data_ is None
        with pytest.raises(
            AttributeError,
            match="Attribute 'anti_hubs_' is available only when algorithm='score-sg'",
        ):
            _ = obj.anti_hubs_

        assert obj.labels_ is None
        assert obj.probabilities_ is None
        assert obj.cluster_persistence_ is None
        assert obj._condensed_tree_array_ is None
        assert obj._single_linkage_tree_array_ is None
        assert obj._min_spanning_tree_array_ is None

    def test_init_accepts_noise_configuration(self, core_sg_module):
        obj = core_sg_module.CoreSG(
            no_noise=False,
            noise_label_strategy="mst_label_propagation",
        )

        assert obj.no_noise is False
        assert obj.noise_label_strategy == "mst_label_propagation"

    def test_init_rejects_non_boolean_no_noise(self, core_sg_module):
        with pytest.raises(TypeError, match="no_noise must be a boolean"):
            core_sg_module.CoreSG(no_noise="yes")

    def test_init_rejects_non_string_noise_label_strategy(self, core_sg_module):
        with pytest.raises(TypeError, match="noise_label_strategy must be a string"):
            core_sg_module.CoreSG(noise_label_strategy=123)

    def test_init_rejects_unknown_noise_label_strategy(self, core_sg_module):
        with pytest.raises(ValueError, match="Unknown noise_label_strategy"):
            core_sg_module.CoreSG(noise_label_strategy="unknown_strategy")

    def test_init_accepts_algorithm_random_state_and_approx_kwargs(
        self, core_sg_module
    ):
        obj = core_sg_module.CoreSG(
            algorithm="score-sg",
            random_state=7,
            approx_knn_kwargs={"n_trees": 4},
        )

        assert obj.algorithm == "score-sg"
        assert obj.random_state == 7
        assert obj.approx_knn_kwargs == {"n_trees": 4}

    def test_init_rejects_invalid_algorithm(self, core_sg_module):
        with pytest.raises(ValueError, match="algorithm must be one of"):
            core_sg_module.CoreSG(algorithm="unknown")

    def test_init_rejects_non_dict_approx_knn_kwargs(self, core_sg_module):
        with pytest.raises(TypeError, match="approx_knn_kwargs must be a dictionary"):
            core_sg_module.CoreSG(approx_knn_kwargs=["bad"])

    def test_init_rejects_invalid_verbose(self, core_sg_module):
        with pytest.raises(
            ValueError, match="verbose must be an integer greater than or equal to 0"
        ):
            core_sg_module.CoreSG(verbose=-1)

    def test_init_rejects_non_callable_progress_callback(self, core_sg_module):
        with pytest.raises(TypeError, match="progress_callback must be callable"):
            core_sg_module.CoreSG(progress_callback="not-callable")

    def test_score_sg_blocks_access_to_core_sg_specific_D_attribute(
        self, core_sg_module
    ):
        obj = core_sg_module.CoreSG(algorithm="score-sg")

        with pytest.raises(
            AttributeError,
            match="Attribute 'distance_matrix_' is available only when algorithm='core-sg'",
        ):
            _ = obj.distance_matrix_

    def test_score_sg_allows_access_to_anti_hubs_attribute_before_fit(
        self, core_sg_module
    ):
        obj = core_sg_module.CoreSG(algorithm="score-sg")

        assert obj.anti_hubs_ is None

    def test_property_setters_store_raw_arrays(self, core_sg_module):
        obj = core_sg_module.CoreSG()

        obj.distance_matrix_ = np.eye(2)
        obj.condensed_tree_ = np.ones((1, 4), dtype=np.float64)
        obj.single_linkage_tree_ = np.ones((1, 3), dtype=np.float64)
        obj.minimum_spanning_tree_ = np.ones((1, 3), dtype=np.float64)

        assert obj.distance_matrix_.shape == (2, 2)
        assert np.array_equal(obj._condensed_tree_array_, np.ones((1, 4)))
        assert np.array_equal(obj._single_linkage_tree_array_, np.ones((1, 3)))
        assert np.array_equal(obj._min_spanning_tree_array_, np.ones((1, 3)))

    def test_set_verbose_validates_and_updates_value(self, core_sg_module):
        obj = core_sg_module.CoreSG()

        with pytest.raises(ValueError, match="verbose"):
            obj.set_verbose(-1)

        obj.set_verbose(2)

        assert obj.verbose == 2

    def test_get_tree_to_labels_kwargs_filters_supported_non_none_values(
        self, core_sg_module
    ):
        obj = core_sg_module.CoreSG(
            cluster_selection_method="leaf",
            allow_single_cluster=True,
            cluster_selection_epsilon=0.25,
            unsupported_flag="ignored",
            max_cluster_size=None,
        )

        filtered = obj._get_tree_to_labels_kwargs()

        assert filtered == {
            "cluster_selection_method": "leaf",
            "allow_single_cluster": True,
            "cluster_selection_epsilon": 0.25,
        }

    @pytest.mark.parametrize(
        ("parameter", "value"),
        [
            ("cluster_selection_method", "leaf"),
            ("allow_single_cluster", True),
            ("match_reference_implementation", True),
            ("cluster_selection_epsilon", 0.25),
            ("cluster_selection_persistence", 0.3),
            ("max_cluster_size", 7),
            ("cluster_selection_epsilon_max", 2.5),
        ],
    )
    def test_hdbscan_style_parameters_are_stored_and_considered_for_labels(
        self, core_sg_module, parameter, value
    ):
        obj = core_sg_module.CoreSG(**{parameter: value})

        assert obj.hdbscan_kwargs[parameter] == value
        assert obj._get_tree_to_labels_kwargs() == {parameter: value}

    def test_hdbscan_style_parameters_with_none_values_are_not_forwarded(
        self, core_sg_module
    ):
        obj = core_sg_module.CoreSG(
            cluster_selection_method=None,
            cluster_selection_persistence=None,
            unsupported_flag="ignored",
        )

        assert obj.hdbscan_kwargs == {
            "cluster_selection_method": None,
            "cluster_selection_persistence": None,
            "unsupported_flag": "ignored",
        }
        assert obj._get_tree_to_labels_kwargs() == {}

    def test_properties_raise_before_fit_or_extract(self, core_sg_module):
        obj = core_sg_module.CoreSG()

        with pytest.raises(AttributeError):
            _ = obj.minimum_spanning_tree_
        with pytest.raises(AttributeError):
            _ = obj.minimum_spanning_tree_k_max_
        with pytest.raises(AttributeError):
            _ = obj.single_linkage_tree_
        with pytest.raises(AttributeError):
            _ = obj.single_linkage_tree_k_max_
        with pytest.raises(AttributeError):
            _ = obj.condensed_tree_
        with pytest.raises(AttributeError):
            _ = obj.condensed_tree_k_max_
        with pytest.raises(AttributeError):
            _ = obj.get_fitted_hdbscan_objects()


class TestBuildCoreSGInputValidation:
    def test_build_core_sg_from_data_uses_minkowski_and_arccos_metrics(
        self, core_sg_module
    ):
        X = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]], dtype=np.float64)

        _, _, _, D_minkowski, _ = core_sg_module.build_core_sg_from_data(
            X, k_max=2, metric="minkowski", p=3
        )
        _, _, _, D_arccos, _ = core_sg_module.build_core_sg_from_data(
            X, k_max=2, metric="arccos"
        )

        assert D_minkowski.shape == (3, 3)
        assert D_arccos.shape == (3, 3)

    def test_build_by_algorithm_rejects_invalid_internal_algorithm(
        self, core_sg_module
    ):
        obj = core_sg_module.CoreSG()
        obj.algorithm = "invalid"
        X = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="algorithm"):
            obj._build_by_algorithm(X, k_max=2)

    def test_build_core_sg_from_data_rejects_single_point_input(self, core_sg_module):
        X = np.array([[0.0, 1.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="shape > 1"):
            core_sg_module.build_core_sg_from_data(X, 1)

    def test_build_core_sg_from_data_rejects_non_positive_k_max(self, core_sg_module):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="1 <= k_max <= n-1"):
            core_sg_module.build_core_sg_from_data(X, 0)

    def test_build_core_sg_from_data_rejects_k_max_greater_than_or_equal_to_n(
        self, core_sg_module
    ):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="1 <= k_max <= n-1"):
            core_sg_module.build_core_sg_from_data(X, 3)

    def test_build_core_sg_from_data_rejects_k_max_smaller_than_two(
        self, core_sg_module
    ):
        X = np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            dtype=np.float64,
        )

        with pytest.raises(ValueError, match="must be >= 2"):
            core_sg_module.build_core_sg_from_data(X, 1)
