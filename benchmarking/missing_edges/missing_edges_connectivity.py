from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core_sg.edges import build_knng_vectors  # noqa: E402
from core_sg.score_sg import (  # noqa: E402
    build_approximate_knn_graph,
    compute_in_degrees,
    select_score_sg_anti_hubs,
)

LOGGER = logging.getLogger("benchmarking.missing_edges_connectivity")

DEFAULT_DISTRIBUTIONS = (
    "gaussian",
    "poisson",
    "chi_square",
    "gamma",
    "beta",
    "von_mises",
    "gumbel",
    "logistic",
    "gaussian_sparse",
)
DEFAULT_REAL_DATASETS = ("iris", "wine", "breast_cancer", "digits")
PAPER_SAMPLE_SIZES = (1000, 5000, 10000, 25000, 50000, 100000, 200000, 500000, 1000000)
PAPER_DIMENSIONS = (2, 10, 32, 64, 128)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    family: str
    distribution: str
    structure: str
    X: np.ndarray
    normalized: bool
    seed: int | None = None


@dataclass(frozen=True)
class ConnectivityStats:
    connected: bool
    n_components: int
    largest_component_size: int
    smallest_component_size: int


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def parse_str_list(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = raw.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(path)


def parse_drop_column(raw: str) -> tuple[str, str]:
    if ":" not in raw:
        raise argparse.ArgumentTypeError("expected DATASET:COLUMN")
    name, column = raw.split(":", 1)
    name = name.strip()
    column = column.strip()
    if not name or not column:
        raise argparse.ArgumentTypeError("expected non-empty DATASET:COLUMN")
    return name, column


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def make_synthetic_distribution(
    distribution: str,
    *,
    n_samples: int,
    n_features: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (n_samples, n_features)

    if distribution == "gaussian":
        X = rng.normal(loc=0.0, scale=1.0, size=shape)
    elif distribution == "poisson":
        X = rng.poisson(lam=5.0, size=shape).astype(np.float64)
    elif distribution == "chi_square":
        X = rng.chisquare(df=2.0, size=shape)
    elif distribution == "gamma":
        X = rng.gamma(shape=2.0, scale=2.0, size=shape)
    elif distribution == "beta":
        X = rng.beta(a=2.0, b=5.0, size=shape)
    elif distribution == "von_mises":
        X = rng.vonmises(mu=0.0, kappa=4.0, size=shape)
    elif distribution == "gumbel":
        X = rng.gumbel(loc=0.0, scale=1.0, size=shape)
    elif distribution == "logistic":
        X = rng.logistic(loc=0.0, scale=1.0, size=shape)
    elif distribution == "gaussian_sparse":
        X = rng.normal(loc=0.0, scale=1.0, size=shape)
        # mask = rng.random(size=shape) < 0.9
        # X[mask] = 0.0
    else:
        raise ValueError(f"unknown synthetic distribution: {distribution}")

    return np.asarray(X, dtype=np.float64)


def make_separated_cluster_distribution(
    distribution: str,
    *,
    n_samples: int,
    n_features: int,
    seed: int,
    n_clusters: int,
    separation: float,
    cluster_scale: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.arange(n_samples, dtype=np.int64) % n_clusters
    rng.shuffle(labels)

    centers = np.zeros((n_clusters, n_features), dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, num=n_clusters, endpoint=False)
    if n_features == 1:
        centers[:, 0] = np.linspace(
            -separation,
            separation,
            num=n_clusters,
            dtype=np.float64,
        )
    else:
        centers[:, 0] = separation * np.cos(angles)
        centers[:, 1] = separation * np.sin(angles)
        if n_features > 2:
            centers[:, 2:] = rng.normal(
                loc=0.0,
                scale=0.25 * separation,
                size=(n_clusters, n_features - 2),
            )

    noise = make_synthetic_distribution(
        distribution,
        n_samples=n_samples,
        n_features=n_features,
        seed=seed + 104729,
    )
    noise = preprocess_features(noise, normalize=True)
    return centers[labels] + cluster_scale * noise


def preprocess_features(X: np.ndarray, *, normalize: bool) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X = SimpleImputer(strategy="median").fit_transform(X)
    if normalize:
        X = StandardScaler().fit_transform(X)
    return np.ascontiguousarray(X, dtype=np.float64)


def iter_synthetic_datasets(
    *,
    distributions: Iterable[str],
    sample_sizes: Iterable[int],
    dimensions: Iterable[int],
    seeds: Iterable[int],
    normalize: bool,
    include_separated: bool,
    only_separated: bool,
    separated_clusters: int,
    separated_cluster_scale: float,
    separated_cluster_separation: float,
) -> Iterable[DatasetSpec]:
    for distribution in distributions:
        for seed in seeds:
            for n_samples in sample_sizes:
                for n_features in dimensions:
                    X = make_synthetic_distribution(
                        distribution,
                        n_samples=n_samples,
                        n_features=n_features,
                        seed=seed,
                    )
                    X = preprocess_features(X, normalize=normalize)
                    if not only_separated:
                        yield DatasetSpec(
                            name=(
                                f"{distribution}-iid-n{n_samples}-d{n_features}-seed{seed}"
                            ),
                            family="synthetic",
                            distribution=distribution,
                            structure="iid",
                            X=X,
                            normalized=normalize,
                            seed=seed,
                        )
                    if include_separated:
                        separated = make_separated_cluster_distribution(
                            distribution,
                            n_samples=n_samples,
                            n_features=n_features,
                            seed=seed,
                            n_clusters=separated_clusters,
                            separation=separated_cluster_separation,
                            cluster_scale=separated_cluster_scale,
                        )
                        separated = preprocess_features(separated, normalize=normalize)
                        yield DatasetSpec(
                            name=(
                                f"{distribution}-separated-n{n_samples}"
                                f"-d{n_features}-seed{seed}"
                            ),
                            family="synthetic",
                            distribution=distribution,
                            structure="separated_clusters",
                            X=separated,
                            normalized=normalize,
                            seed=seed,
                        )


def load_builtin_real_dataset(name: str) -> DatasetSpec:
    loaders = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "digits": load_digits,
    }
    if name not in loaders:
        raise ValueError(f"unknown built-in real dataset: {name}")

    data = loaders[name]()
    X = preprocess_features(data.data, normalize=True)
    return DatasetSpec(
        name=name,
        family="real",
        distribution="real_builtin",
        structure="real",
        X=X,
        normalized=True,
        seed=None,
    )


def load_csv_real_dataset(
    name: str,
    path: Path,
    *,
    drop_columns: Iterable[str],
    normalize: bool,
) -> DatasetSpec:
    frame = pd.read_csv(path)
    for column in drop_columns:
        if column in frame.columns:
            frame = frame.drop(columns=[column])

    numeric = frame.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError(f"{path} has no numeric feature columns after preprocessing")

    X = preprocess_features(numeric.to_numpy(), normalize=normalize)
    return DatasetSpec(
        name=name,
        family="real",
        distribution="real_csv",
        structure="real",
        X=X,
        normalized=normalize,
        seed=None,
    )


def component_stats(edges: np.ndarray, n_nodes: int) -> ConnectivityStats:
    parent = np.arange(n_nodes, dtype=np.int64)
    size = np.ones(n_nodes, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        size[root_left] += size[root_right]

    if edges.size:
        endpoints = np.asarray(edges[:, :2], dtype=np.int64)
        for left, right in endpoints:
            if left != right:
                union(int(left), int(right))

    roots = np.array([find(node) for node in range(n_nodes)], dtype=np.int64)
    _, counts = np.unique(roots, return_counts=True)
    return ConnectivityStats(
        connected=counts.shape[0] == 1,
        n_components=int(counts.shape[0]),
        largest_component_size=int(counts.max()),
        smallest_component_size=int(counts.min()),
    )


def build_clique_support(vertices: np.ndarray) -> np.ndarray:
    if vertices.shape[0] < 2:
        return np.empty((0, 3), dtype=np.float64)

    left_pos, right_pos = np.triu_indices(vertices.shape[0], k=1)
    left = vertices[left_pos]
    right = vertices[right_pos]
    clique_support = np.empty((left.shape[0], 3), dtype=np.float64)
    clique_support[:, 0] = np.minimum(left, right)
    clique_support[:, 1] = np.maximum(left, right)
    clique_support[:, 2] = -1.0
    return clique_support


def select_random_reinforcement_vertices(
    *,
    n_samples: int,
    selected_size: int,
    random_state: int,
) -> np.ndarray:
    if n_samples <= 0 or selected_size <= 0:
        return np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(random_state)
    size = min(n_samples, selected_size)
    return np.sort(rng.choice(n_samples, size=size, replace=False)).astype(np.int64)


def build_support_variants(
    X: np.ndarray,
    *,
    k_max: int,
    random_state: int,
    approx_knn_kwargs: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    neighbor_indices, neighbor_distances = build_approximate_knn_graph(
        X,
        k_max,
        metric="euclidean",
        p=2,
        random_state=random_state,
        approx_knn_kwargs=approx_knn_kwargs,
    )
    _, knn_support = build_knng_vectors(
        neighbor_indices,
        np.zeros_like(neighbor_distances, dtype=np.float64),
        knng_size=X.shape[0],
        k_max=k_max,
    )

    in_degrees = compute_in_degrees(neighbor_indices, X.shape[0])
    anti_hubs = select_score_sg_anti_hubs(
        neighbor_indices,
        in_degrees,
        random_state=random_state,
    )
    random_vertices = select_random_reinforcement_vertices(
        n_samples=X.shape[0],
        selected_size=anti_hubs.shape[0],
        random_state=random_state,
    )

    anti_hub_clique = build_clique_support(anti_hubs)
    random_clique = build_clique_support(random_vertices)
    score_sg_support = (
        np.vstack([knn_support, anti_hub_clique])
        if anti_hub_clique.size
        else knn_support
    )
    random_support = (
        np.vstack([knn_support, random_clique]) if random_clique.size else knn_support
    )

    return (
        {
            "approx_knn_only": knn_support,
            "random_sqrt_clique": random_support,
            "score_sg_antihub": score_sg_support,
        },
        {
            "approx_knn_only": np.empty(0, dtype=np.int64),
            "random_sqrt_clique": random_vertices,
            "score_sg_antihub": anti_hubs,
        },
        anti_hubs,
        in_degrees,
        neighbor_indices,
    )


def summarize_ties(
    *,
    neighbor_indices: np.ndarray,
    in_degrees: np.ndarray,
    anti_hubs: np.ndarray,
) -> dict[str, int]:
    if anti_hubs.shape[0] == 0:
        return {
            "anti_hub_count": 0,
            "cutoff_in_degree": -1,
            "candidate_tie_count": 0,
            "selected_from_ties_degree_only": 0,
            "selected_from_ties_score_tie_break": 0,
            "score_tie_candidate_count": 0,
        }

    n_samples = in_degrees.shape[0]
    selected_size = int(anti_hubs.shape[0])
    order = np.lexsort((np.arange(n_samples, dtype=np.int64), in_degrees))
    cutoff = int(in_degrees[order[selected_size - 1]])
    fixed_count = int(np.sum(in_degrees < cutoff))
    selected_from_ties_degree_only = selected_size - fixed_count
    tied = order[in_degrees[order] == cutoff]
    score_tie_selected = 0
    score_tie_candidate_count = 0

    if selected_from_ties_degree_only > 0 and tied.shape[0] > 0:
        tie_scores = in_degrees[neighbor_indices[tied]].sum(axis=1)
        sorted_scores = np.sort(tie_scores, kind="mergesort")
        selected_scores = sorted_scores[:selected_from_ties_degree_only]
        boundary_score = selected_scores[-1]
        score_tie_selected = int(np.sum(selected_scores == boundary_score))
        score_tie_candidate_count = int(np.sum(tie_scores == boundary_score))

    return {
        "anti_hub_count": selected_size,
        "cutoff_in_degree": cutoff,
        "candidate_tie_count": int(np.sum(in_degrees == cutoff)),
        "selected_from_ties_degree_only": selected_from_ties_degree_only,
        "selected_from_ties_score_tie_break": score_tie_selected,
        "score_tie_candidate_count": score_tie_candidate_count,
    }


def run_dataset(
    spec: DatasetSpec,
    *,
    k_min: int,
    k_max: int,
    random_state: int,
    approx_knn_kwargs: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    n_samples, n_features = spec.X.shape
    effective_k_max = min(k_max, n_samples - 1)
    if effective_k_max < k_min:
        raise ValueError(
            f"{spec.name} has n={n_samples}; cannot evaluate k_min={k_min}"
        )

    LOGGER.info(
        "Building supports | dataset=%s | n=%s | d=%s | k_max=%s",
        spec.name,
        n_samples,
        n_features,
        effective_k_max,
    )
    start = perf_counter()
    variants, selected_vertices, anti_hubs, in_degrees, neighbor_indices = (
        build_support_variants(
            spec.X,
            k_max=effective_k_max,
            random_state=random_state,
            approx_knn_kwargs=approx_knn_kwargs,
        )
    )
    build_seconds = perf_counter() - start
    tie_summary = summarize_ties(
        neighbor_indices=neighbor_indices,
        in_degrees=in_degrees,
        anti_hubs=anti_hubs,
    )

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for variant, support in variants.items():
        reinforcement_count = int(selected_vertices[variant].shape[0])
        check_start = perf_counter()
        stats = component_stats(support, n_samples)
        check_seconds = perf_counter() - check_start
        n_edges = int(support.shape[0])

        connected_count = 0
        total_k = 0
        for k in range(effective_k_max, k_min - 1, -1):
            total_k += 1
            connected_count += int(stats.connected)
            detail_rows.append(
                {
                    "dataset": spec.name,
                    "family": spec.family,
                    "distribution": spec.distribution,
                    "structure": spec.structure,
                    "variant": variant,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "seed": spec.seed,
                    "normalized": spec.normalized,
                    "k": k,
                    "connected": stats.connected,
                    "n_components": stats.n_components,
                    "largest_component_size": stats.largest_component_size,
                    "smallest_component_size": stats.smallest_component_size,
                    "n_edges": n_edges,
                    "reinforcement_vertex_count": reinforcement_count,
                    "anti_hub_count": tie_summary["anti_hub_count"],
                    "cutoff_in_degree": tie_summary["cutoff_in_degree"],
                    "candidate_tie_count": tie_summary["candidate_tie_count"],
                    "selected_from_ties_degree_only": tie_summary[
                        "selected_from_ties_degree_only"
                    ],
                    "selected_from_ties_score_tie_break": tie_summary[
                        "selected_from_ties_score_tie_break"
                    ],
                    "score_tie_candidate_count": tie_summary[
                        "score_tie_candidate_count"
                    ],
                    "build_seconds": build_seconds,
                    "connectivity_check_seconds": check_seconds,
                }
            )

        summary_rows.append(
            {
                "dataset": spec.name,
                "family": spec.family,
                "distribution": spec.distribution,
                "structure": spec.structure,
                "variant": variant,
                "n_samples": n_samples,
                "n_features": n_features,
                "seed": spec.seed,
                "normalized": spec.normalized,
                "k_min": k_min,
                "k_max": effective_k_max,
                "n_k_values": total_k,
                "connected_all_k": connected_count == total_k,
                "connected_rate": connected_count / total_k,
                "n_components": stats.n_components,
                "largest_component_size": stats.largest_component_size,
                "smallest_component_size": stats.smallest_component_size,
                "n_edges": n_edges,
                "reinforcement_vertex_count": reinforcement_count,
                "anti_hub_count": tie_summary["anti_hub_count"],
                "cutoff_in_degree": tie_summary["cutoff_in_degree"],
                "candidate_tie_count": tie_summary["candidate_tie_count"],
                "selected_from_ties_degree_only": tie_summary[
                    "selected_from_ties_degree_only"
                ],
                "selected_from_ties_score_tie_break": tie_summary[
                    "selected_from_ties_score_tie_break"
                ],
                "score_tie_candidate_count": tie_summary["score_tie_candidate_count"],
                "build_seconds": build_seconds,
                "connectivity_check_seconds": check_seconds,
            }
        )

    return detail_rows, summary_rows


def write_outputs(
    *,
    detail_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    output_dir: Path,
    tie_output_dir: Path,
    generate_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tie_output_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)
    detail.to_csv(output_dir / "missing_edges_connectivity_by_k.csv", index=False)
    summary.to_csv(output_dir / "missing_edges_connectivity_summary.csv", index=False)

    if not summary.empty:
        summary.pivot_table(
            index=["structure", "variant"],
            values="connected_all_k",
            aggfunc=["count", "sum"],
        ).to_csv(output_dir / "missing_edges_connectivity_by_structure.csv")

    failures = summary[~summary["connected_all_k"]]
    failures.to_csv(
        output_dir / "missing_edges_connectivity_failures.csv",
        index=False,
    )

    if not summary.empty and "score_sg_antihub" in set(summary["variant"]):
        tie_columns = [
            "dataset",
            "family",
            "distribution",
            "structure",
            "n_samples",
            "n_features",
            "seed",
            "normalized",
            "anti_hub_count",
            "cutoff_in_degree",
            "candidate_tie_count",
            "selected_from_ties_degree_only",
            "selected_from_ties_score_tie_break",
            "score_tie_candidate_count",
            "connected_all_k",
            "n_components",
        ]
        tie_detail = summary[summary["variant"] == "score_sg_antihub"][
            tie_columns
        ].copy()
        tie_detail.to_csv(
            tie_output_dir / "missing_edges_tie_diagnostics.csv",
            index=False,
        )
    else:
        tie_detail = pd.DataFrame()

    if not tie_detail.empty:
        synthetic_ties = tie_detail[tie_detail["family"] == "synthetic"].copy()
        if not synthetic_ties.empty:
            dimension_summary = (
                synthetic_ties.groupby("n_features", as_index=False)
                .agg(
                    selected_from_ties_degree_only=(
                        "selected_from_ties_degree_only",
                        "mean",
                    ),
                    selected_from_ties_score_tie_break=(
                        "selected_from_ties_score_tie_break",
                        "mean",
                    ),
                    candidate_tie_count=("candidate_tie_count", "mean"),
                )
                .sort_values("n_features")
            )
            dimension_summary.to_csv(
                tie_output_dir / "missing_edges_tie_by_dimension.csv",
                index=False,
            )

    if not generate_plots:
        return

    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib is not installed; skipping plot generation")
        return

    if summary.empty:
        return

    plot_data = summary.copy()
    plot_data["label"] = (
        plot_data["distribution"].astype(str)
        + "/"
        + plot_data["structure"].astype(str)
        + "\n"
        + plot_data["n_samples"].astype(str)
        + "x"
        + plot_data["n_features"].astype(str)
    )
    plot_data = plot_data.sort_values(
        ["family", "distribution", "structure", "n_samples"]
    )

    for variant, group in plot_data.groupby("variant", sort=True):
        fig_width = max(8.0, 0.45 * len(group))
        fig, ax = plt.subplots(figsize=(fig_width, 4.5))
        ax.bar(group["label"], group["connected_rate"], color="#3b6ea8")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Connected k fraction")
        ax.set_title(f"Support connectivity after missing-edge strategy: {variant}")
        ax.tick_params(axis="x", rotation=75, labelsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{variant}_connectivity_rate.png", dpi=200)
        plt.close(fig)

    if tie_detail.empty:
        return

    if "dimension_summary" not in locals() or dimension_summary.empty:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(
        dimension_summary["n_features"],
        dimension_summary["selected_from_ties_degree_only"],
        marker="o",
        linewidth=2.0,
        label="Degree-only selection",
    )
    ax.plot(
        dimension_summary["n_features"],
        dimension_summary["selected_from_ties_score_tie_break"],
        marker="s",
        linewidth=2.0,
        label="Score tie-break",
    )
    ax.set_xlabel("Dimensions")
    ax.set_ylabel("Mean selected anti-hubs still tied")
    ax.set_title("Anti-hub tie reduction by dimensionality")
    ax.set_xticks(dimension_summary["n_features"].tolist())
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(tie_output_dir / "score_sg_tie_break_by_dimension.png", dpi=220)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether ScoreSG support graphs are connected after "
            "anti-hub missing-edge reinforcement for k values from k_max to k_min."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarking/missing_edges/results"),
        help="Directory where missing-edge connectivity outputs are written.",
    )
    parser.add_argument(
        "--tie-output-dir",
        type=Path,
        default=Path("benchmarking/unties/results"),
        help="Directory where anti-hub tie-breaking diagnostics are written.",
    )
    parser.add_argument(
        "--preset",
        choices=["quick", "paper"],
        default="quick",
        help="quick is suitable for smoke tests; paper expands the synthetic grid.",
    )
    parser.add_argument(
        "--sample-sizes",
        type=parse_int_list,
        help=(
            "Comma-separated synthetic sample sizes. The paper preset defaults to "
            "1k, 5k, 10k, 25k, 50k, 100k, 200k, 500k, and 1M."
        ),
    )
    parser.add_argument(
        "--dimensions",
        type=parse_int_list,
        help="Comma-separated synthetic dimensions. The paper preset includes 128.",
    )
    parser.add_argument("--seeds", type=parse_int_list, default=[42])
    parser.add_argument(
        "--distributions",
        type=parse_str_list,
        default=list(DEFAULT_DISTRIBUTIONS),
    )
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-synthetic", action="store_true")
    parser.add_argument("--no-real", action="store_true")
    parser.add_argument(
        "--real-datasets",
        type=parse_str_list,
        default=list(DEFAULT_REAL_DATASETS),
        help="Comma-separated built-in sklearn datasets.",
    )
    parser.add_argument(
        "--no-builtin-real",
        action="store_true",
        help="Skip built-in sklearn real datasets while keeping optional real CSVs.",
    )
    parser.add_argument(
        "--real-csv",
        action="append",
        type=parse_named_path,
        default=[],
        metavar="NAME=PATH",
        help=(
            "Optional local real dataset CSV. Non-numeric columns are ignored. "
            "Example: --real-csv beans=/path/to/Dry_Bean_Dataset.csv"
        ),
    )
    parser.add_argument(
        "--include-separated-synthetic",
        action="store_true",
        help="Also generate separated-cluster versions of each synthetic dataset.",
    )
    parser.add_argument(
        "--no-separated-synthetic",
        action="store_true",
        help="Disable separated synthetic datasets for the paper preset.",
    )
    parser.add_argument(
        "--only-separated-synthetic",
        action="store_true",
        help="Generate only separated synthetic datasets, skipping iid synthetic data.",
    )
    parser.add_argument(
        "--separated-clusters",
        type=int,
        default=6,
        help="Number of clusters in separated synthetic variants.",
    )
    parser.add_argument(
        "--separated-cluster-scale",
        type=float,
        default=0.65,
        help="Within-cluster scale for separated synthetic variants.",
    )
    parser.add_argument(
        "--separated-cluster-separation",
        type=float,
        default=8.0,
        help="Between-center separation for separated synthetic variants.",
    )
    parser.add_argument(
        "--csv-drop-column",
        action="append",
        type=parse_drop_column,
        default=[],
        metavar="DATASET:COLUMN",
        help="Column to drop from a named CSV before selecting numeric features.",
    )
    parser.add_argument(
        "--normalize-synthetic",
        action="store_true",
        help="Apply median imputation and StandardScaler to synthetic datasets.",
    )
    parser.add_argument(
        "--no-normalize-real",
        action="store_true",
        help="Disable StandardScaler for built-in and CSV real datasets.",
    )
    parser.add_argument(
        "--approx-n-trees",
        type=int,
        default=None,
        help="Optional PyNNDescent n_trees override.",
    )
    parser.add_argument(
        "--approx-n-iters",
        type=int,
        default=None,
        help="Optional PyNNDescent n_iters override.",
    )
    parser.add_argument(
        "--max-real-samples",
        type=int,
        default=None,
        help="Optional deterministic cap for large real datasets.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write CSV files only and skip matplotlib figure generation.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    sample_sizes = args.sample_sizes
    dimensions = args.dimensions
    if sample_sizes is None:
        sample_sizes = [1000] if args.preset == "quick" else list(PAPER_SAMPLE_SIZES)
    if dimensions is None:
        dimensions = [2, 10] if args.preset == "quick" else list(PAPER_DIMENSIONS)
    include_separated = args.include_separated_synthetic or (
        args.preset == "paper" and not args.no_separated_synthetic
    )
    if args.only_separated_synthetic and not include_separated:
        parser.error("--only-separated-synthetic requires separated synthetic datasets")

    approx_knn_kwargs: dict[str, object] = {}
    if args.approx_n_trees is not None:
        approx_knn_kwargs["n_trees"] = args.approx_n_trees
    if args.approx_n_iters is not None:
        approx_knn_kwargs["n_iters"] = args.approx_n_iters

    datasets: list[DatasetSpec] = []
    if not args.no_synthetic:
        datasets.extend(
            iter_synthetic_datasets(
                distributions=args.distributions,
                sample_sizes=sample_sizes,
                dimensions=dimensions,
                seeds=args.seeds,
                normalize=args.normalize_synthetic,
                include_separated=include_separated,
                only_separated=args.only_separated_synthetic,
                separated_clusters=args.separated_clusters,
                separated_cluster_scale=args.separated_cluster_scale,
                separated_cluster_separation=args.separated_cluster_separation,
            )
        )

    normalize_real = not args.no_normalize_real
    if not args.no_real and not args.no_builtin_real:
        for name in args.real_datasets:
            spec = load_builtin_real_dataset(name)
            if (
                args.max_real_samples is not None
                and spec.X.shape[0] > args.max_real_samples
            ):
                spec = DatasetSpec(
                    name=f"{spec.name}-first{args.max_real_samples}",
                    family=spec.family,
                    distribution=spec.distribution,
                    structure=spec.structure,
                    X=spec.X[: args.max_real_samples],
                    normalized=spec.normalized,
                    seed=spec.seed,
                )
            datasets.append(spec)

    drop_columns_by_dataset: dict[str, list[str]] = {}
    for dataset_name, column in args.csv_drop_column:
        drop_columns_by_dataset.setdefault(dataset_name, []).append(column)

    for name, path in args.real_csv:
        spec = load_csv_real_dataset(
            name,
            path,
            drop_columns=drop_columns_by_dataset.get(name, []),
            normalize=normalize_real,
        )
        if (
            args.max_real_samples is not None
            and spec.X.shape[0] > args.max_real_samples
        ):
            spec = DatasetSpec(
                name=f"{spec.name}-first{args.max_real_samples}",
                family=spec.family,
                distribution=spec.distribution,
                structure=spec.structure,
                X=spec.X[: args.max_real_samples],
                normalized=spec.normalized,
                seed=spec.seed,
            )
        datasets.append(spec)

    if not datasets:
        raise SystemExit("no datasets selected")

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []

    for spec in datasets:
        try:
            detail, summary = run_dataset(
                spec,
                k_min=args.k_min,
                k_max=args.k_max,
                random_state=args.random_state,
                approx_knn_kwargs=approx_knn_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Dataset failed: %s", spec.name)
            failures.append((spec.name, repr(exc)))
            continue
        detail_rows.extend(detail)
        summary_rows.extend(summary)

    for dataset_name, error in failures:
        summary_rows.append(
            {
                "dataset": dataset_name,
                "family": "failed",
                "distribution": "failed",
                "structure": "failed",
                "variant": "failed",
                "n_samples": math.nan,
                "n_features": math.nan,
                "seed": math.nan,
                "normalized": math.nan,
                "k_min": args.k_min,
                "k_max": args.k_max,
                "n_k_values": 0,
                "connected_all_k": False,
                "connected_rate": 0.0,
                "n_components": math.nan,
                "largest_component_size": math.nan,
                "smallest_component_size": math.nan,
                "n_edges": math.nan,
                "reinforcement_vertex_count": math.nan,
                "anti_hub_count": math.nan,
                "cutoff_in_degree": math.nan,
                "candidate_tie_count": math.nan,
                "selected_from_ties_degree_only": math.nan,
                "selected_from_ties_score_tie_break": math.nan,
                "score_tie_candidate_count": math.nan,
                "build_seconds": math.nan,
                "connectivity_check_seconds": math.nan,
                "error": error,
            }
        )

    write_outputs(
        detail_rows=detail_rows,
        summary_rows=summary_rows,
        output_dir=args.output_dir,
        tie_output_dir=args.tie_output_dir,
        generate_plots=not args.no_plots,
    )
    LOGGER.info("Wrote connectivity results to %s", args.output_dir)
    LOGGER.info("Wrote tie diagnostics to %s", args.tie_output_dir)


if __name__ == "__main__":
    main()
