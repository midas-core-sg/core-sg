from __future__ import annotations

import numpy as np

from benchmarking.quality import quality_comparison
from benchmarking.quality.quality_comparison import (
    HAI_DEFINITION,
    MethodResult,
    QualityDataset,
    exact_hai,
    reusable_mst_array,
    result_row,
    summarize_rows,
)


def balanced_single_linkage_tree() -> np.ndarray:
    return np.array(
        [
            [0, 1, 0.1, 2],
            [2, 3, 0.2, 2],
            [4, 5, 0.3, 4],
        ],
        dtype=np.float64,
    )


def chain_single_linkage_tree() -> np.ndarray:
    return np.array(
        [
            [0, 1, 0.1, 2],
            [4, 2, 0.2, 3],
            [5, 3, 0.3, 4],
        ],
        dtype=np.float64,
    )


def test_exact_hai_uses_single_linkage_smallest_common_cluster_sizes():
    score = exact_hai(
        chain_single_linkage_tree(),
        balanced_single_linkage_tree(),
        n_samples=4,
    )

    assert score == 0.875


def test_result_row_hai_is_independent_from_mst_edge_jaccard():
    dataset = QualityDataset(
        name="tiny",
        benchmark_group="unit",
        family="synthetic",
        distribution="manual",
        structure="manual",
        X=np.zeros((4, 2), dtype=np.float64),
        y_true=None,
        normalized=False,
        seed=42,
    )
    reference = MethodResult(
        method="hdbscan_generic",
        labels=np.zeros(4, dtype=np.int64),
        mst=np.array(
            [
                [0, 1, 0.1],
                [1, 2, 0.2],
                [2, 3, 0.3],
            ],
            dtype=np.float64,
        ),
        single_linkage_tree=balanced_single_linkage_tree(),
        fit_seconds=0.0,
        extract_seconds=0.0,
        status="ok",
    )
    result = MethodResult(
        method="score_sg",
        labels=np.zeros(4, dtype=np.int64),
        mst=np.array(
            [
                [0, 2, 0.1],
                [0, 3, 0.2],
                [1, 3, 0.3],
            ],
            dtype=np.float64,
        ),
        single_linkage_tree=balanced_single_linkage_tree(),
        fit_seconds=0.0,
        extract_seconds=0.0,
        status="ok",
    )

    row = result_row(dataset, k=2, k_max=2, result=result, reference=reference)

    assert row["hai_definition"] == HAI_DEFINITION
    assert row["hai"] == 1.0
    assert row["mst_edge_jaccard"] == 0.0


def test_reusable_mst_array_falls_back_for_k_max():
    class Core:
        _min_spanning_tree_array_ = None
        _min_spanning_tree_k_max_array_ = np.array([[0, 1, 0.5]], dtype=np.float64)

    result = reusable_mst_array(
        Core(),
        k=5,
        k_max=5,
    )

    assert np.array_equal(result, Core._min_spanning_tree_k_max_array_)


def test_reusable_mst_array_reconstructs_when_cache_is_missing(monkeypatch):
    expected = np.array([[0, 1, 0.5]], dtype=np.float64)

    class Core:
        _min_spanning_tree_array_ = None
        _min_spanning_tree_k_max_array_ = None
        support_graph_ = np.array([[0, 1, -1]], dtype=np.float64)
        metric_edges_ = np.array([[1, 0, 0.5]], dtype=np.float64)
        core_distances_ = np.array([[0.0, 0.1], [0.0, 0.1]], dtype=np.float64)
        n_samples_ = 2

    def fake_mst_from_core_sg(**kwargs):
        assert kwargs["n_nodes"] == 2
        assert kwargs["k"] == 2
        return expected

    monkeypatch.setattr(quality_comparison, "mst_from_core_sg", fake_mst_from_core_sg)

    result = reusable_mst_array(Core(), k=2, k_max=2)

    assert np.array_equal(result, expected)


def test_summary_aggregates_seed_replicates_by_configuration():
    rows = [
        {
            "benchmark_group": "synthetic",
            "dataset": "gaussian-iid-n20-d2-seed1",
            "dataset_config": "gaussian-iid-n20-d2",
            "family": "synthetic",
            "distribution": "gaussian",
            "structure": "iid",
            "n_samples": 20,
            "n_features": 2,
            "method": "ScoreSG",
            "method_key": "score_sg",
            "k": 2,
            "seed": 1,
            "status": "ok",
            "ari_vs_hdbscan_generic": 0.8,
            "ari_vs_true": np.nan,
            "hai": 0.9,
            "total_seconds": 1.0,
            "workflow_seconds_contribution": 1.0,
            "n_clusters_found": 2,
        },
        {
            "benchmark_group": "synthetic",
            "dataset": "gaussian-iid-n20-d2-seed2",
            "dataset_config": "gaussian-iid-n20-d2",
            "family": "synthetic",
            "distribution": "gaussian",
            "structure": "iid",
            "n_samples": 20,
            "n_features": 2,
            "method": "ScoreSG",
            "method_key": "score_sg",
            "k": 2,
            "seed": 2,
            "status": "ok",
            "ari_vs_hdbscan_generic": 1.0,
            "ari_vs_true": np.nan,
            "hai": 1.0,
            "total_seconds": 2.0,
            "workflow_seconds_contribution": 2.0,
            "n_clusters_found": 3,
        },
    ]
    rows.extend(
        [
            {
                **rows[0],
                "k": 3,
                "ari_vs_hdbscan_generic": 1.0,
                "hai": 1.0,
                "total_seconds": 3.0,
                "workflow_seconds_contribution": 3.0,
            },
            {
                **rows[1],
                "k": 3,
                "ari_vs_hdbscan_generic": 1.0,
                "hai": 1.0,
                "total_seconds": 4.0,
                "workflow_seconds_contribution": 4.0,
            },
        ]
    )

    summary = summarize_rows(rows)

    assert summary.shape[0] == 1
    row = summary.iloc[0]
    assert row["dataset_config"] == "gaussian-iid-n20-d2"
    assert row["seed_values"] == 2
    assert row["dataset_replicates"] == 2
    assert row["k_values"] == 4
    assert row["distinct_k_values"] == 2
    assert np.isclose(row["mean_hai"], 0.975)
    assert np.isclose(row["std_hai"], np.sqrt(0.00125))
