from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestNoiseHandlerFactory:
    def test_build_noise_handler_returns_mst_strategy(self, noise_handler_module):
        handler = noise_handler_module.build_noise_handler("mst_label_propagation", c=5)

        assert isinstance(handler, noise_handler_module.MSTLabelPropagationStrategy)
        assert handler.c == 5

    def test_build_noise_handler_rejects_unknown_strategy(self, noise_handler_module):
        with pytest.raises(ValueError, match="Unknown noise_label_strategy"):
            noise_handler_module.build_noise_handler("unknown_strategy", c=5)

    def test_noise_handler_validates_c(self, noise_handler_module):
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            noise_handler_module.MSTLabelPropagationStrategy(c=0)


class TestMSTLabelPropagationStrategy:
    def test_reassign_single_seed_labels_entire_chain(self, noise_handler_module):
        labels = np.array([2, -1, -1, -1], dtype=np.int64)
        mst = np.array(
            [
                [0, 1, 1.0],
                [1, 2, 1.5],
                [2, 3, 2.0],
            ],
            dtype=np.float64,
        )
        handler = noise_handler_module.MSTLabelPropagationStrategy(c=5)

        result = handler.reassign(labels=labels, min_spanning_tree=mst, n_samples=4)

        assert np.array_equal(result, np.array([2, 2, 2, 2], dtype=np.int64))

    def test_reassign_returns_copy_when_no_noise_exists(self, noise_handler_module):
        labels = np.array([0, 0, 1], dtype=np.int64)
        mst = np.array([[0, 1, 1.0], [1, 2, 2.0]], dtype=np.float64)
        handler = noise_handler_module.MSTLabelPropagationStrategy(c=5)

        result = handler.reassign(labels=labels, min_spanning_tree=mst, n_samples=3)

        assert np.array_equal(result, labels)
        assert result is not labels

    def test_reassign_returns_copy_when_all_points_are_noise(
        self, noise_handler_module
    ):
        labels = np.array([-1, -1, -1], dtype=np.int64)
        mst = np.array([[0, 1, 1.0], [1, 2, 2.0]], dtype=np.float64)
        handler = noise_handler_module.MSTLabelPropagationStrategy(c=5)

        result = handler.reassign(labels=labels, min_spanning_tree=mst, n_samples=3)

        assert np.array_equal(result, labels)
        assert result is not labels

    def test_reassign_propagates_labels_along_tree(self, noise_handler_module):
        labels = np.array([0, -1, -1, 1], dtype=np.int64)
        mst = np.array(
            [
                [0, 1, 1.0],
                [1, 2, 2.0],
                [2, 3, 3.0],
            ],
            dtype=np.float64,
        )
        handler = noise_handler_module.MSTLabelPropagationStrategy(c=5)

        result = handler.reassign(labels=labels, min_spanning_tree=mst, n_samples=4)

        assert np.array_equal(result, np.array([0, 0, 0, 1], dtype=np.int64))

    def test_reassign_resolves_competition_with_top_c_rule(self, noise_handler_module):
        labels = np.array([0, 1, -1, -1], dtype=np.int64)
        mst = np.array(
            [
                [0, 2, 3.0],
                [1, 2, 2.0],
                [2, 3, 1.0],
            ],
            dtype=np.float64,
        )
        handler = noise_handler_module.MSTLabelPropagationStrategy(c=1)

        result = handler.reassign(labels=labels, min_spanning_tree=mst, n_samples=4)

        assert np.array_equal(result, np.array([0, 1, 1, 1], dtype=np.int64))

    def test_reassign_prefers_smaller_lexicographic_top_c_signature(
        self, noise_handler_module
    ):
        labels = np.array([0, 1, -1, -1, -1], dtype=np.int64)
        mst = np.array(
            [
                [0, 2, 5.0],
                [1, 3, 4.0],
                [2, 4, 10.0],
                [3, 4, 10.0],
            ],
            dtype=np.float64,
        )
        handler = noise_handler_module.MSTLabelPropagationStrategy(c=2)

        result = handler.reassign(labels=labels, min_spanning_tree=mst, n_samples=5)

        assert np.array_equal(result, np.array([0, 1, 0, 1, 1], dtype=np.int64))

    def test_reassign_preserves_original_labeled_points(self, noise_handler_module):
        labels = np.array([3, -1, -1, 7], dtype=np.int64)
        mst = np.array(
            [
                [0, 1, 1.0],
                [1, 2, 2.0],
                [2, 3, 3.0],
            ],
            dtype=np.float64,
        )
        handler = noise_handler_module.MSTLabelPropagationStrategy(c=5)

        result = handler.reassign(labels=labels, min_spanning_tree=mst, n_samples=4)

        assert result[0] == 3
        assert result[3] == 7

    def test_reassign_leaves_disconnected_noise_without_seed_unchanged(
        self, noise_handler_module
    ):
        labels = np.array([-1], dtype=np.int64)
        mst = np.empty((0, 3), dtype=np.float64)
        handler = noise_handler_module.MSTLabelPropagationStrategy(c=5)

        result = handler.reassign(labels=labels, min_spanning_tree=mst, n_samples=1)

        assert np.array_equal(result, labels)

    def test_reassign_validates_labels_shape(self, noise_handler_module):
        labels = np.array([[0, -1, 1]], dtype=np.int64)
        mst = np.array([[0, 1, 1.0], [1, 2, 2.0]], dtype=np.float64)
        handler = noise_handler_module.MSTLabelPropagationStrategy(c=5)

        with pytest.raises(ValueError, match="one-dimensional array"):
            handler.reassign(labels=labels, min_spanning_tree=mst, n_samples=3)

    def test_reassign_validates_mst_shape(self, noise_handler_module):
        labels = np.array([0, -1, 1], dtype=np.int64)
        mst = np.array([[0, 1], [1, 2]], dtype=np.float64)
        handler = noise_handler_module.MSTLabelPropagationStrategy(c=5)

        with pytest.raises(ValueError, match="shape \\(n_edges, 3\\)"):
            handler.reassign(labels=labels, min_spanning_tree=mst, n_samples=3)

    def test_reassign_validates_mst_edge_count(self, noise_handler_module):
        labels = np.array([0, -1, 1], dtype=np.int64)
        mst = np.array([[0.0, 1.0, 1.0]], dtype=np.float64)
        handler = noise_handler_module.MSTLabelPropagationStrategy(c=2)

        with pytest.raises(ValueError, match="n_samples - 1"):
            handler.reassign(labels=labels, min_spanning_tree=mst, n_samples=3)

    def test_reassign_skips_already_labeled_neighbors(self, noise_handler_module):
        labels = np.array([0, 1, -1], dtype=np.int64)
        mst = np.array([[0.0, 1.0, 1.0], [1.0, 2.0, 2.0]], dtype=np.float64)
        handler = noise_handler_module.MSTLabelPropagationStrategy(c=2)

        result = handler.reassign(labels=labels, min_spanning_tree=mst, n_samples=3)

        assert np.array_equal(result, np.array([0, 1, 1], dtype=np.int64))
