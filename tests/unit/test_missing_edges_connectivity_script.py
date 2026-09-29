from __future__ import annotations

import numpy as np

from benchmarking.missing_edges.missing_edges_connectivity import (
    DEFAULT_DISTRIBUTIONS,
    build_clique_support,
    component_stats,
    make_separated_cluster_distribution,
    make_synthetic_distribution,
    preprocess_features,
    select_random_reinforcement_vertices,
    summarize_ties,
)


def test_component_stats_detects_connected_graph():
    edges = np.array(
        [
            [0, 1, -1.0],
            [1, 2, -1.0],
            [2, 3, -1.0],
        ],
        dtype=np.float64,
    )

    result = component_stats(edges, n_nodes=4)

    assert result.connected is True
    assert result.n_components == 1
    assert result.largest_component_size == 4
    assert result.smallest_component_size == 4


def test_component_stats_reports_component_sizes_for_disconnected_graph():
    edges = np.array(
        [
            [0, 1, -1.0],
            [2, 3, -1.0],
        ],
        dtype=np.float64,
    )

    result = component_stats(edges, n_nodes=5)

    assert result.connected is False
    assert result.n_components == 3
    assert result.largest_component_size == 2
    assert result.smallest_component_size == 1


def test_all_default_synthetic_distributions_return_finite_arrays():
    for distribution in DEFAULT_DISTRIBUTIONS:
        X = make_synthetic_distribution(
            distribution,
            n_samples=12,
            n_features=3,
            seed=42,
        )

        assert X.shape == (12, 3)
        assert np.isfinite(X).all()


def test_separated_cluster_distribution_preserves_shape_and_separates_centers():
    X = make_separated_cluster_distribution(
        "gaussian",
        n_samples=120,
        n_features=2,
        seed=42,
        n_clusters=6,
        separation=8.0,
        cluster_scale=0.1,
    )

    assert X.shape == (120, 2)
    assert np.isfinite(X).all()
    assert np.linalg.norm(X.mean(axis=0)) < 1.0
    assert X.std(axis=0).min() > 3.0


def test_preprocess_features_imputes_and_normalizes():
    X = np.array(
        [
            [1.0, np.nan],
            [2.0, 10.0],
            [3.0, 20.0],
        ],
        dtype=np.float64,
    )

    result = preprocess_features(X, normalize=True)

    assert result.shape == X.shape
    assert np.isfinite(result).all()
    assert np.allclose(result.mean(axis=0), np.zeros(2))


def test_random_reinforcement_selects_unique_sqrt_n_vertices():
    result = select_random_reinforcement_vertices(
        n_samples=25,
        selected_size=5,
        random_state=42,
    )

    assert result.shape == (5,)
    assert np.unique(result).shape == result.shape
    assert result.min() >= 0
    assert result.max() < 25


def test_build_clique_support_connects_selected_vertices():
    result = build_clique_support(np.array([4, 1, 3], dtype=np.int64))

    assert result.shape == (3, 3)
    assert set(map(tuple, result[:, :2].astype(np.int64))) == {
        (1, 4),
        (3, 4),
        (1, 3),
    }


def test_summarize_ties_compares_degree_only_and_score_tie_break():
    neighbor_indices = np.array(
        [
            [0, 0],
            [5, 0],
            [6, 0],
            [6, 0],
            [7, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ],
        dtype=np.int64,
    )
    in_degrees = np.array([0, 0, 0, 0, 0, 1, 2, 3, 4], dtype=np.int64)
    anti_hubs = np.array([0, 1, 2], dtype=np.int64)

    result = summarize_ties(
        neighbor_indices=neighbor_indices,
        in_degrees=in_degrees,
        anti_hubs=anti_hubs,
    )

    assert result["anti_hub_count"] == 3
    assert result["candidate_tie_count"] == 5
    assert result["selected_from_ties_degree_only"] == 3
    assert result["selected_from_ties_score_tie_break"] == 1
    assert result["score_tie_candidate_count"] == 2
