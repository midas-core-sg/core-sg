from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import hdbscan
import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_blobs,
)
from sklearn.metrics import adjusted_rand_score

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - exercised only without numba.
    njit = None
    prange = range

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.missing_edges.missing_edges_connectivity import (  # noqa: E402
    DEFAULT_DISTRIBUTIONS,
    DEFAULT_REAL_DATASETS,
    make_synthetic_distribution,
    parse_int_list,
    parse_named_path,
    parse_str_list,
    preprocess_features,
)
from core_sg import CoreSG  # noqa: E402
from core_sg.core_sg import mst_from_core_sg  # noqa: E402
from core_sg.edges import build_knng_vectors  # noqa: E402
from core_sg.hdbscan_adapter import mst_to_single_linkage_tree  # noqa: E402
from core_sg.reweight import sort_core_sg  # noqa: E402
from core_sg.score_sg import (  # noqa: E402
    _compute_exact_edge_distances,
    build_approximate_core_k_list,
    build_approximate_knn_graph,
    build_selected_clique,
)

LOGGER = logging.getLogger("benchmarking.quality_comparison")

DEFAULT_SAMPLE_SIZES = (1000, 5000, 10000, 25000, 50000)
DEFAULT_RUNTIME_SAMPLE_SIZES = (5000, 10000, 20000, 30000, 40000, 50000)
DEFAULT_DIMENSIONS = (2, 10, 20, 32, 64, 128)
DEFAULT_METHODS = (
    "hdbscan_generic",
    "optimized_hdbscan",
    "score_sg",
    "score_sg_random",
)
METHOD_DISPLAY_NAMES = {
    "hdbscan_generic": "HDBSCAN",
    "optimized_hdbscan": "Optimized HDBSCAN",
    "score_sg": "ScoreSG",
    "score_sg_random": "ScoreSG Random",
}
HAI_DEFINITION = "exact_single_linkage_pairwise_smallest_common_cluster_size_agreement"
MAX_PYTHON_HAI_SAMPLES = 50_000


@dataclass(frozen=True)
class QualityDataset:
    name: str
    benchmark_group: str
    family: str
    distribution: str
    structure: str
    X: np.ndarray
    y_true: np.ndarray | None
    normalized: bool
    seed: int | None
    n_clusters_true: int | None = None


@dataclass(frozen=True)
class Batch:
    name: str
    output_dir: Path
    dataset: QualityDataset


@dataclass
class MethodResult:
    method: str
    labels: np.ndarray | None
    mst: np.ndarray | None
    single_linkage_tree: np.ndarray | None
    fit_seconds: float
    extract_seconds: float
    status: str
    error: str = ""


if njit is not None:

    @njit(cache=True)
    def _rmq_lca(
        left_leaf: int,
        right_leaf: int,
        first_occurrence: np.ndarray,
        euler: np.ndarray,
        depth: np.ndarray,
        sparse_table: np.ndarray,
        log2: np.ndarray,
    ) -> int:
        left = first_occurrence[left_leaf]
        right = first_occurrence[right_leaf]
        if left > right:
            tmp = left
            left = right
            right = tmp
        length = right - left + 1
        level = log2[length]
        offset = 1 << level
        pos_a = sparse_table[level, left]
        pos_b = sparse_table[level, right - offset + 1]
        if depth[pos_a] <= depth[pos_b]:
            return euler[pos_a]
        return euler[pos_b]

    @njit(cache=True, parallel=True, fastmath=True)
    def _exact_hai_from_lca_structures(
        first_a: np.ndarray,
        euler_a: np.ndarray,
        depth_a: np.ndarray,
        sparse_a: np.ndarray,
        cluster_size_a: np.ndarray,
        first_b: np.ndarray,
        euler_b: np.ndarray,
        depth_b: np.ndarray,
        sparse_b: np.ndarray,
        cluster_size_b: np.ndarray,
        log2: np.ndarray,
        n_samples: int,
    ) -> float:
        total = 0.0
        normalizer = float(n_samples)
        for i in prange(n_samples - 1):
            local = 0.0
            for j in range(i + 1, n_samples):
                lca_a = _rmq_lca(i, j, first_a, euler_a, depth_a, sparse_a, log2)
                lca_b = _rmq_lca(i, j, first_b, euler_b, depth_b, sparse_b, log2)
                distance_a = cluster_size_a[lca_a] / normalizer
                distance_b = cluster_size_b[lca_b] / normalizer
                diff = distance_a - distance_b
                if diff < 0.0:
                    diff = -diff
                local += diff
            total += local
        return 1.0 - (2.0 * total / (normalizer * normalizer))


def _rmq_lca_python(
    left_leaf: int,
    right_leaf: int,
    first_occurrence: np.ndarray,
    euler: np.ndarray,
    depth: np.ndarray,
    sparse_table: np.ndarray,
    log2: np.ndarray,
) -> int:
    left = int(first_occurrence[left_leaf])
    right = int(first_occurrence[right_leaf])
    if left > right:
        left, right = right, left
    length = right - left + 1
    level = int(log2[length])
    offset = 1 << level
    pos_a = int(sparse_table[level, left])
    pos_b = int(sparse_table[level, right - offset + 1])
    return int(euler[pos_a] if depth[pos_a] <= depth[pos_b] else euler[pos_b])


def _exact_hai_from_lca_structures_python(
    first_a: np.ndarray,
    euler_a: np.ndarray,
    depth_a: np.ndarray,
    sparse_a: np.ndarray,
    cluster_size_a: np.ndarray,
    first_b: np.ndarray,
    euler_b: np.ndarray,
    depth_b: np.ndarray,
    sparse_b: np.ndarray,
    cluster_size_b: np.ndarray,
    log2: np.ndarray,
    n_samples: int,
) -> float:
    total = 0.0
    normalizer = float(n_samples)
    for i in range(n_samples - 1):
        for j in range(i + 1, n_samples):
            lca_a = _rmq_lca_python(i, j, first_a, euler_a, depth_a, sparse_a, log2)
            lca_b = _rmq_lca_python(i, j, first_b, euler_b, depth_b, sparse_b, log2)
            total += abs(
                (cluster_size_a[lca_a] / normalizer)
                - (cluster_size_b[lca_b] / normalizer)
            )
    return 1.0 - (2.0 * total / (normalizer * normalizer))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run ARI and hierarchy-agreement comparisons one dataset "
            "configuration at a time. HDBSCAN with algorithm='generic' is the "
            "exact reference for ARI-vs-reference and HAI."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarking/quality/results"),
        help="Directory for merged CSV files and per-configuration run folders.",
    )
    parser.add_argument(
        "--benchmark-groups",
        type=parse_str_list,
        default=["runtime", "synthetic", "real"],
        help="Comma-separated subset of: runtime,synthetic,real.",
    )
    parser.add_argument(
        "--methods",
        type=parse_str_list,
        default=list(DEFAULT_METHODS),
        help=(
            "Comma-separated subset of: hdbscan_generic,optimized_hdbscan,"
            "score_sg,score_sg_random."
        ),
    )
    parser.add_argument(
        "--sample-sizes",
        type=parse_int_list,
        default=list(DEFAULT_SAMPLE_SIZES),
        help="Synthetic missing-edge/unties sample sizes. Keep <= 50000.",
    )
    parser.add_argument(
        "--runtime-sample-sizes",
        type=parse_int_list,
        default=list(DEFAULT_RUNTIME_SAMPLE_SIZES),
        help="Runtime-benchmark sample sizes. Keep <= 50000 for exact HDBSCAN.",
    )
    parser.add_argument(
        "--dimensions",
        type=parse_int_list,
        default=list(DEFAULT_DIMENSIONS),
        help="Synthetic diagnostic dimensions.",
    )
    parser.add_argument(
        "--runtime-dimensions",
        type=parse_int_list,
        default=[20],
        help="Runtime-benchmark dimensions.",
    )
    parser.add_argument("--runtime-centers", type=int, default=10)
    parser.add_argument(
        "--distributions",
        type=parse_str_list,
        default=list(DEFAULT_DISTRIBUTIONS),
    )
    parser.add_argument("--seeds", type=parse_int_list, default=[42])
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=50)
    parser.add_argument("--synthetic-normalize", action="store_true")
    parser.add_argument("--no-separated", action="store_true")
    parser.add_argument("--separated-clusters", type=int, default=6)
    parser.add_argument("--separated-cluster-scale", type=float, default=0.65)
    parser.add_argument("--separated-cluster-separation", type=float, default=8.0)
    parser.add_argument(
        "--real-datasets",
        type=parse_str_list,
        default=list(DEFAULT_REAL_DATASETS),
    )
    parser.add_argument(
        "--no-builtin-real",
        action="store_true",
        help="Skip built-in real datasets and use only datasets passed with --real-csv.",
    )
    parser.add_argument(
        "--real-csv",
        action="append",
        type=parse_named_path,
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--csv-drop-column",
        action="append",
        default=[],
        metavar="DATASET:COLUMN",
        help="Drop a column before selecting numeric features from a CSV dataset.",
    )
    parser.add_argument(
        "--csv-target-column",
        action="append",
        default=[],
        metavar="DATASET:COLUMN",
        help="Optional target-label column for ARI-vs-true on CSV datasets.",
    )
    parser.add_argument("--max-real-samples", type=int, default=None)
    parser.add_argument(
        "--approx-knn-kwargs-json",
        default="{}",
        help="JSON object forwarded to PyNNDescent for ScoreSG variants.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def validate_args(args: argparse.Namespace) -> None:
    allowed_groups = {"runtime", "synthetic", "real"}
    unknown_groups = set(args.benchmark_groups) - allowed_groups
    if unknown_groups:
        raise ValueError(f"Unknown benchmark groups: {sorted(unknown_groups)}")

    unknown_methods = set(args.methods) - set(DEFAULT_METHODS)
    if unknown_methods:
        raise ValueError(f"Unknown methods: {sorted(unknown_methods)}")

    if "hdbscan_generic" not in args.methods:
        raise ValueError(
            "hdbscan_generic is required because it is the exact comparison reference."
        )
    if args.k_max < 2 or args.k_min < 2 or args.k_min > args.k_max:
        raise ValueError("Require 2 <= k_min <= k_max.")
    if max(args.sample_sizes + args.runtime_sample_sizes) > 50000:
        raise ValueError(
            "Exact HDBSCAN generic is memory-intensive; this script caps N at 50000."
        )
    if args.separated_clusters < 1:
        raise ValueError("--separated-clusters must be positive.")
    if args.max_real_samples is not None and args.max_real_samples < 2:
        raise ValueError("--max-real-samples must be at least 2 when provided.")


def parse_column_mapping(values: Iterable[str]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for raw in values:
        if ":" not in raw:
            raise argparse.ArgumentTypeError("expected DATASET:COLUMN")
        dataset, column = raw.split(":", 1)
        dataset = dataset.strip()
        column = column.strip()
        if not dataset or not column:
            raise argparse.ArgumentTypeError("expected DATASET:COLUMN")
        mapping.setdefault(dataset, []).append(column)
    return mapping


def slugify(value: str) -> str:
    allowed = []
    for char in value.lower():
        if char.isalnum():
            allowed.append(char)
        elif char in {"-", "_", "."}:
            allowed.append("-")
        else:
            allowed.append("-")
    slug = "".join(allowed).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "dataset"


def make_separated_dataset_with_labels(
    distribution: str,
    *,
    n_samples: int,
    n_features: int,
    seed: int,
    n_clusters: int,
    separation: float,
    cluster_scale: float,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.arange(n_samples, dtype=np.int64) % n_clusters
    rng.shuffle(labels)

    centers = np.zeros((n_clusters, n_features), dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, num=n_clusters, endpoint=False)
    centers[:, 0] = separation * np.cos(angles)
    if n_features > 1:
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
    X = centers[labels] + cluster_scale * noise
    X = preprocess_features(X, normalize=normalize)
    return X, labels


def iter_runtime_datasets(args: argparse.Namespace) -> Iterable[QualityDataset]:
    for seed in args.seeds:
        for n_samples in args.runtime_sample_sizes:
            for n_features in args.runtime_dimensions:
                X, labels = make_blobs(
                    n_samples=n_samples,
                    n_features=n_features,
                    centers=args.runtime_centers,
                    random_state=seed,
                )
                X = np.ascontiguousarray(X, dtype=np.float64)
                yield QualityDataset(
                    name=(
                        f"runtime-blobs-n{n_samples}-d{n_features}"
                        f"-c{args.runtime_centers}-seed{seed}"
                    ),
                    benchmark_group="runtime",
                    family="synthetic",
                    distribution="gaussian_blobs",
                    structure="separated_clusters",
                    X=X,
                    y_true=np.asarray(labels, dtype=np.int64),
                    normalized=False,
                    seed=seed,
                    n_clusters_true=args.runtime_centers,
                )


def iter_synthetic_quality_datasets(
    args: argparse.Namespace,
) -> Iterable[QualityDataset]:
    for distribution in args.distributions:
        for seed in args.seeds:
            for n_samples in args.sample_sizes:
                for n_features in args.dimensions:
                    X = make_synthetic_distribution(
                        distribution,
                        n_samples=n_samples,
                        n_features=n_features,
                        seed=seed,
                    )
                    X = preprocess_features(X, normalize=args.synthetic_normalize)
                    yield QualityDataset(
                        name=f"{distribution}-iid-n{n_samples}-d{n_features}-seed{seed}",
                        benchmark_group="synthetic",
                        family="synthetic",
                        distribution=distribution,
                        structure="iid",
                        X=X,
                        y_true=None,
                        normalized=args.synthetic_normalize,
                        seed=seed,
                    )

                    if not args.no_separated:
                        separated, labels = make_separated_dataset_with_labels(
                            distribution,
                            n_samples=n_samples,
                            n_features=n_features,
                            seed=seed,
                            n_clusters=args.separated_clusters,
                            separation=args.separated_cluster_separation,
                            cluster_scale=args.separated_cluster_scale,
                            normalize=args.synthetic_normalize,
                        )
                        yield QualityDataset(
                            name=(
                                f"{distribution}-separated-n{n_samples}"
                                f"-d{n_features}-seed{seed}"
                            ),
                            benchmark_group="synthetic",
                            family="synthetic",
                            distribution=distribution,
                            structure="separated_clusters",
                            X=separated,
                            y_true=labels,
                            normalized=args.synthetic_normalize,
                            seed=seed,
                            n_clusters_true=args.separated_clusters,
                        )


def load_builtin_real_dataset(name: str) -> QualityDataset:
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
    return QualityDataset(
        name=name,
        benchmark_group="real",
        family="real",
        distribution="real_builtin",
        structure="real",
        X=X,
        y_true=np.asarray(data.target, dtype=np.int64),
        normalized=True,
        seed=None,
        n_clusters_true=int(np.unique(data.target).shape[0]),
    )


def with_dataset_seed(dataset: QualityDataset, seed: int) -> QualityDataset:
    return QualityDataset(
        **{
            **asdict(dataset),
            "name": f"{dataset.name}-seed{seed}",
            "seed": seed,
        }
    )


def load_csv_real_dataset(
    name: str,
    path: Path,
    *,
    drop_columns: Iterable[str],
    target_columns: Iterable[str],
    max_samples: int | None,
) -> QualityDataset:
    frame = pd.read_csv(path)
    target = None
    for column in target_columns:
        if column in frame.columns:
            target = frame[column].to_numpy()
            frame = frame.drop(columns=[column])

    for column in drop_columns:
        if column in frame.columns:
            frame = frame.drop(columns=[column])

    numeric = frame.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError(f"{path} has no numeric feature columns after preprocessing")

    if max_samples is not None and numeric.shape[0] > max_samples:
        numeric = numeric.iloc[:max_samples]
        if target is not None:
            target = target[:max_samples]

    X = preprocess_features(numeric.to_numpy(), normalize=True)
    y_true = (
        None
        if target is None
        else pd.Series(target).astype("category").cat.codes.to_numpy()
    )
    return QualityDataset(
        name=name,
        benchmark_group="real",
        family="real",
        distribution="real_csv",
        structure="real",
        X=X,
        y_true=y_true,
        normalized=True,
        seed=None,
        n_clusters_true=None if y_true is None else int(np.unique(y_true).shape[0]),
    )


def iter_real_datasets(args: argparse.Namespace) -> Iterable[QualityDataset]:
    if not args.no_builtin_real:
        for name in args.real_datasets:
            dataset = load_builtin_real_dataset(name)
            if (
                args.max_real_samples is not None
                and dataset.X.shape[0] > args.max_real_samples
            ):
                dataset = QualityDataset(
                    **{
                        **asdict(dataset),
                        "X": dataset.X[: args.max_real_samples],
                        "y_true": (
                            None
                            if dataset.y_true is None
                            else dataset.y_true[: args.max_real_samples]
                        ),
                    }
                )
            for seed in args.seeds:
                yield with_dataset_seed(dataset, seed)

    drop_mapping = parse_column_mapping(args.csv_drop_column)
    target_mapping = parse_column_mapping(args.csv_target_column)
    for name, path in args.real_csv:
        dataset = load_csv_real_dataset(
            name,
            path,
            drop_columns=drop_mapping.get(name, []),
            target_columns=target_mapping.get(name, []),
            max_samples=args.max_real_samples,
        )
        for seed in args.seeds:
            yield with_dataset_seed(dataset, seed)


def iter_datasets(args: argparse.Namespace) -> Iterable[QualityDataset]:
    if "runtime" in args.benchmark_groups:
        yield from iter_runtime_datasets(args)
    if "synthetic" in args.benchmark_groups:
        yield from iter_synthetic_quality_datasets(args)
    if "real" in args.benchmark_groups:
        yield from iter_real_datasets(args)


def make_batches(args: argparse.Namespace) -> list[Batch]:
    run_root = args.output_dir / "_runs"
    batches = [
        Batch(
            name=slugify(dataset.name),
            output_dir=run_root / slugify(dataset.name),
            dataset=dataset,
        )
        for dataset in iter_datasets(args)
    ]
    if args.limit is not None:
        batches = batches[: args.limit]
    return batches


def fit_hdbscan(X: np.ndarray, *, k: int, algorithm: str) -> MethodResult:
    start = perf_counter()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=k,
        min_samples=k,
        metric="euclidean",
        algorithm=algorithm,
        approx_min_span_tree=False,
        gen_min_span_tree=True,
        match_reference_implementation=True,
    ).fit(X)
    elapsed = perf_counter() - start
    return MethodResult(
        method="hdbscan_generic" if algorithm == "generic" else "optimized_hdbscan",
        labels=np.asarray(clusterer.labels_, dtype=np.int64),
        mst=np.asarray(clusterer._min_spanning_tree, dtype=np.float64),
        single_linkage_tree=np.asarray(
            clusterer._single_linkage_tree, dtype=np.float64
        ),
        fit_seconds=elapsed,
        extract_seconds=0.0,
        status="ok",
    )


def build_score_sg_random_core(
    X: np.ndarray,
    *,
    k_max: int,
    random_state: int,
    approx_knn_kwargs: dict[str, object],
) -> CoreSG:
    neighbor_indices, neighbor_distances = build_approximate_knn_graph(
        X,
        k_max,
        metric="euclidean",
        p=2,
        random_state=random_state,
        approx_knn_kwargs=approx_knn_kwargs,
    )
    core_k_list = build_approximate_core_k_list(neighbor_distances, k_max)

    source = np.repeat(np.arange(X.shape[0], dtype=np.int64), k_max)
    target = neighbor_indices.reshape(-1)
    exact_distances = _compute_exact_edge_distances(
        X,
        source,
        target,
        metric="euclidean",
        p=2,
    ).reshape(X.shape[0], k_max)
    metric_edges, knn_support = build_knng_vectors(
        neighbor_indices,
        exact_distances,
        knng_size=X.shape[0],
        k_max=k_max,
    )

    rng = np.random.default_rng(random_state)
    selected_size = max(1, int(math.floor(math.sqrt(X.shape[0]))))
    selected = np.sort(
        rng.choice(X.shape[0], size=min(selected_size, X.shape[0]), replace=False)
    ).astype(np.int64)
    clique_metric_edges, clique_support_edges = build_selected_clique(
        X,
        selected,
        metric="euclidean",
        p=2,
    )
    if clique_metric_edges.shape[0] > 0:
        metric_edges = np.vstack([metric_edges, clique_metric_edges])
        support_graph = np.vstack([knn_support, clique_support_edges])
    else:
        support_graph = knn_support

    core = CoreSG(
        metric="euclidean",
        p=2,
        algorithm="score-sg",
        no_noise=False,
        random_state=random_state,
        approx_knn_kwargs=approx_knn_kwargs,
        match_reference_implementation=True,
    )
    core.n_samples_ = X.shape[0]
    core.k_max_ = k_max
    core._raw_data_ = X
    core.support_graph_ = sort_core_sg(support_graph)
    core.metric_edges_ = metric_edges
    core.core_distances_ = core_k_list
    core._tree_to_labels_data_ = X
    core.anti_hubs_ = selected
    core._extract_score_sg_k_max_outputs()
    ensure_reusable_core_metadata(core, X, k_max=k_max)
    return core


def ensure_reusable_core_metadata(core: CoreSG, X: np.ndarray, *, k_max: int) -> None:
    core.n_samples_ = X.shape[0]
    core.k_max_ = k_max
    if getattr(core, "_raw_data_", None) is None:
        core._raw_data_ = X


def build_reusable_method(
    method: str,
    X: np.ndarray,
    *,
    k_max: int,
    random_state: int,
    approx_knn_kwargs: dict[str, object],
) -> tuple[CoreSG | None, float, str, str]:
    start = perf_counter()
    try:
        if method == "score_sg":
            core = CoreSG(
                metric="euclidean",
                p=2,
                algorithm="score-sg",
                no_noise=False,
                random_state=random_state,
                approx_knn_kwargs=approx_knn_kwargs,
                match_reference_implementation=True,
            )
            core.fit(X, k_max=k_max)
            ensure_reusable_core_metadata(core, X, k_max=k_max)
        elif method == "score_sg_random":
            core = build_score_sg_random_core(
                X,
                k_max=k_max,
                random_state=random_state,
                approx_knn_kwargs=approx_knn_kwargs,
            )
        else:
            raise ValueError(f"Not a reusable method: {method}")
        return core, perf_counter() - start, "ok", ""
    except Exception as exc:  # noqa: BLE001 - benchmark must record failures.
        tb = traceback.extract_tb(exc.__traceback__)
        location = ""
        if tb:
            frame = tb[-1]
            location = f" at {Path(frame.filename).name}:{frame.lineno}"
        return (
            None,
            perf_counter() - start,
            "failed",
            f"{type(exc).__name__}{location}: {exc}",
        )


def extract_reusable_result(
    core: CoreSG | None,
    *,
    method: str,
    k: int,
    k_max: int,
    fit_seconds: float,
    build_status: str,
    build_error: str,
) -> MethodResult:
    if core is None:
        return MethodResult(
            method=method,
            labels=None,
            mst=None,
            single_linkage_tree=None,
            fit_seconds=fit_seconds,
            extract_seconds=0.0,
            status=build_status,
            error=build_error,
        )

    try:
        start = perf_counter()
        core.extract_hierarchy_from_core_sg(k)
        extract_seconds = perf_counter() - start
        mst = reusable_mst_array(
            core,
            k=k,
            k_max=k_max,
        )
        single_linkage_tree = reusable_single_linkage_array(
            core,
            mst=mst,
            k=k,
            k_max=k_max,
        )
        return MethodResult(
            method=method,
            labels=np.asarray(core.labels_, dtype=np.int64),
            mst=np.asarray(mst, dtype=np.float64),
            single_linkage_tree=np.asarray(single_linkage_tree, dtype=np.float64),
            fit_seconds=fit_seconds,
            extract_seconds=extract_seconds,
            status="ok",
        )
    except Exception as exc:  # noqa: BLE001 - benchmark must record failures.
        tb = traceback.extract_tb(exc.__traceback__)
        location = ""
        if tb:
            frame = tb[-1]
            location = f" at {Path(frame.filename).name}:{frame.lineno}"
        return MethodResult(
            method=method,
            labels=None,
            mst=None,
            single_linkage_tree=None,
            fit_seconds=fit_seconds,
            extract_seconds=0.0,
            status="failed",
            error=f"{type(exc).__name__}{location}: {exc}",
        )


def reusable_mst_array(
    core: CoreSG,
    *,
    k: int,
    k_max: int,
) -> np.ndarray:
    value = getattr(core, "_min_spanning_tree_array_", None)
    if value is not None:
        return value

    if k == k_max:
        value = getattr(core, "_min_spanning_tree_k_max_array_", None)
        if value is not None:
            return value

    support_graph = getattr(core, "support_graph_", None)
    metric_edges = getattr(core, "metric_edges_", None)
    core_distances = getattr(core, "core_distances_", None)
    n_samples = getattr(core, "n_samples_", None)
    if (
        support_graph is not None
        and metric_edges is not None
        and core_distances is not None
        and n_samples is not None
    ):
        return mst_from_core_sg(
            core_sg=support_graph,
            metric_edges=metric_edges,
            core_k_list=core_distances,
            n_nodes=int(n_samples),
            k=k,
        )

    raise AttributeError(
        "CoreSG did not expose or allow reconstruction of the MST after "
        "hierarchy extraction."
    )


def reusable_single_linkage_array(
    core: CoreSG,
    *,
    mst: np.ndarray,
    k: int,
    k_max: int,
) -> np.ndarray:
    value = getattr(core, "_single_linkage_tree_array_", None)
    if value is not None:
        return value

    if k == k_max:
        value = getattr(core, "_single_linkage_tree_k_max_array_", None)
        if value is not None:
            return value

    return mst_to_single_linkage_tree(mst)


def build_hierarchy_lca_structure(
    single_linkage_tree: np.ndarray,
    *,
    n_samples: int,
) -> dict[str, np.ndarray]:
    tree = np.asarray(single_linkage_tree, dtype=np.float64)
    expected_rows = n_samples - 1
    if tree.shape[0] != expected_rows or tree.shape[1] < 4:
        raise ValueError(
            "single_linkage_tree must have shape (n_samples - 1, at least 4)."
        )

    n_nodes = 2 * n_samples - 1
    left_child = np.full(n_nodes, -1, dtype=np.int64)
    right_child = np.full(n_nodes, -1, dtype=np.int64)
    cluster_size = np.ones(n_nodes, dtype=np.float64)

    for row_idx, row in enumerate(tree):
        node = n_samples + row_idx
        left = int(row[0])
        right = int(row[1])
        left_child[node] = left
        right_child[node] = right
        cluster_size[node] = float(row[3])

    root = n_nodes - 1
    euler_list: list[int] = []
    depth_list: list[int] = []
    first_occurrence = np.full(n_nodes, -1, dtype=np.int64)
    stack: list[tuple[int, int, int]] = [(root, 0, 0)]

    while stack:
        node, state, depth_value = stack.pop()
        if first_occurrence[node] == -1:
            first_occurrence[node] = len(euler_list)

        left = left_child[node]
        right = right_child[node]
        if left == -1 and right == -1:
            euler_list.append(node)
            depth_list.append(depth_value)
            continue

        if state == 0:
            euler_list.append(node)
            depth_list.append(depth_value)
            stack.append((node, 1, depth_value))
            stack.append((left, 0, depth_value + 1))
        elif state == 1:
            euler_list.append(node)
            depth_list.append(depth_value)
            stack.append((node, 2, depth_value))
            stack.append((right, 0, depth_value + 1))
        else:
            euler_list.append(node)
            depth_list.append(depth_value)

    euler = np.asarray(euler_list, dtype=np.int64)
    depth = np.asarray(depth_list, dtype=np.int64)
    first_leaf = first_occurrence[:n_samples].copy()
    if np.any(first_leaf < 0):
        raise ValueError("single_linkage_tree does not contain all leaves.")

    log2 = np.zeros(euler.shape[0] + 1, dtype=np.int64)
    for idx in range(2, log2.shape[0]):
        log2[idx] = log2[idx // 2] + 1

    levels = int(log2[euler.shape[0]]) + 1
    sparse_table = np.empty((levels, euler.shape[0]), dtype=np.int64)
    sparse_table[0] = np.arange(euler.shape[0], dtype=np.int64)
    for level in range(1, levels):
        span = 1 << level
        half = span >> 1
        limit = euler.shape[0] - span + 1
        previous = sparse_table[level - 1]
        current = sparse_table[level]
        for idx in range(limit):
            pos_a = previous[idx]
            pos_b = previous[idx + half]
            current[idx] = pos_a if depth[pos_a] <= depth[pos_b] else pos_b
        if limit < euler.shape[0]:
            current[limit:] = current[limit - 1]

    return {
        "first": first_leaf,
        "euler": euler,
        "depth": depth,
        "sparse": sparse_table,
        "cluster_size": cluster_size,
        "log2": log2,
    }


def exact_hai(
    method_single_linkage_tree: np.ndarray | None,
    reference_single_linkage_tree: np.ndarray | None,
    *,
    n_samples: int,
) -> float:
    """Compute exact HAI from two single-linkage hierarchy trees.

    HAI compares all point pairs using the size of the smallest single-linkage
    cluster containing each pair, normalized by the dataset size. The MST edge
    set is intentionally not used here; MST diagnostics are reported separately.
    """
    if method_single_linkage_tree is None or reference_single_linkage_tree is None:
        return float("nan")

    method_structure = build_hierarchy_lca_structure(
        method_single_linkage_tree,
        n_samples=n_samples,
    )
    reference_structure = build_hierarchy_lca_structure(
        reference_single_linkage_tree,
        n_samples=n_samples,
    )
    if njit is None:
        if n_samples > MAX_PYTHON_HAI_SAMPLES:
            raise ImportError(
                "Exact HAI without numba is only enabled for small runs "
                f"(n_samples <= {MAX_PYTHON_HAI_SAMPLES}). Install numba or "
                "use a smaller dataset for the all-pairs hierarchy comparison."
            )
        return float(
            _exact_hai_from_lca_structures_python(
                method_structure["first"],
                method_structure["euler"],
                method_structure["depth"],
                method_structure["sparse"],
                method_structure["cluster_size"],
                reference_structure["first"],
                reference_structure["euler"],
                reference_structure["depth"],
                reference_structure["sparse"],
                reference_structure["cluster_size"],
                method_structure["log2"],
                n_samples,
            )
        )

    return float(
        _exact_hai_from_lca_structures(  # type: ignore[name-defined]
            method_structure["first"],
            method_structure["euler"],
            method_structure["depth"],
            method_structure["sparse"],
            method_structure["cluster_size"],
            reference_structure["first"],
            reference_structure["euler"],
            reference_structure["depth"],
            reference_structure["sparse"],
            reference_structure["cluster_size"],
            method_structure["log2"],
            n_samples,
        )
    )


def edge_weight_map(mst: np.ndarray | None) -> dict[tuple[int, int], float]:
    if mst is None:
        return {}
    result: dict[tuple[int, int], float] = {}
    for row in np.asarray(mst, dtype=np.float64):
        left = int(row[0])
        right = int(row[1])
        if left == right:
            continue
        edge = (min(left, right), max(left, right))
        result[edge] = float(row[2])
    return result


def mst_edge_diagnostics(
    method_mst: np.ndarray | None,
    reference_mst: np.ndarray | None,
) -> dict[str, object]:
    method_edges = edge_weight_map(method_mst)
    reference_edges = edge_weight_map(reference_mst)
    method_set = set(method_edges)
    reference_set = set(reference_edges)

    if not method_set or not reference_set:
        return {
            "mst_edge_jaccard": np.nan,
            "mst_common_edges": 0,
            "mst_method_edges": len(method_set),
            "mst_reference_edges": len(reference_set),
            "mst_weight_mae_common": np.nan,
            "mst_weight_rmse_common": np.nan,
        }

    common = method_set & reference_set
    union = method_set | reference_set
    diffs = np.array(
        [method_edges[edge] - reference_edges[edge] for edge in common],
        dtype=np.float64,
    )
    return {
        "mst_edge_jaccard": len(common) / len(union),
        "mst_common_edges": len(common),
        "mst_method_edges": len(method_set),
        "mst_reference_edges": len(reference_set),
        "mst_weight_mae_common": (
            float(np.mean(np.abs(diffs))) if diffs.shape[0] else np.nan
        ),
        "mst_weight_rmse_common": (
            float(np.sqrt(np.mean(diffs**2))) if diffs.shape[0] else np.nan
        ),
    }


def safe_ari(labels: np.ndarray | None, reference: np.ndarray | None) -> float:
    if labels is None or reference is None:
        return float("nan")
    return float(adjusted_rand_score(reference, labels))


def n_clusters(labels: np.ndarray | None) -> int | float:
    if labels is None:
        return float("nan")
    unique = set(np.asarray(labels, dtype=np.int64).tolist())
    return len(unique) - int(-1 in unique)


def dataset_config_name(dataset: QualityDataset) -> str:
    if dataset.seed is None:
        return dataset.name
    seed_suffix = f"-seed{dataset.seed}"
    if dataset.name.endswith(seed_suffix):
        return dataset.name[: -len(seed_suffix)]
    return dataset.name


def base_row(dataset: QualityDataset, *, k: int, k_max: int) -> dict[str, object]:
    return {
        "benchmark_group": dataset.benchmark_group,
        "dataset": dataset.name,
        "dataset_config": dataset_config_name(dataset),
        "family": dataset.family,
        "distribution": dataset.distribution,
        "structure": dataset.structure,
        "n_samples": int(dataset.X.shape[0]),
        "n_features": int(dataset.X.shape[1]),
        "n_clusters_true": dataset.n_clusters_true,
        "seed": dataset.seed,
        "normalized": dataset.normalized,
        "k": k,
        "k_max": k_max,
        "reference_method": "HDBSCAN",
        "hai_definition": HAI_DEFINITION,
    }


def result_row(
    dataset: QualityDataset,
    *,
    k: int,
    k_max: int,
    result: MethodResult,
    reference: MethodResult | None,
) -> dict[str, object]:
    mst_comparison = (
        mst_edge_diagnostics(result.mst, reference.mst)
        if reference is not None and reference.status == "ok"
        else mst_edge_diagnostics(result.mst, None)
    )
    if reference is not None and reference.status == "ok" and result.status == "ok":
        if result.method == reference.method:
            hai = 1.0
        else:
            hai = exact_hai(
                result.single_linkage_tree,
                reference.single_linkage_tree,
                n_samples=dataset.X.shape[0],
            )
    else:
        hai = float("nan")

    total_seconds = result.fit_seconds + result.extract_seconds
    if result.method in {"score_sg", "score_sg_random"}:
        workflow_seconds = result.extract_seconds
        if k == k_max:
            workflow_seconds += result.fit_seconds
    else:
        workflow_seconds = total_seconds

    row = {
        **base_row(dataset, k=k, k_max=k_max),
        "method": METHOD_DISPLAY_NAMES[result.method],
        "method_key": result.method,
        "status": result.status,
        "error": result.error,
        "fit_seconds": result.fit_seconds,
        "extract_seconds": result.extract_seconds,
        "total_seconds": total_seconds,
        "workflow_seconds_contribution": workflow_seconds,
        "n_clusters_found": n_clusters(result.labels),
        "ari_vs_hdbscan_generic": safe_ari(
            result.labels,
            None if reference is None else reference.labels,
        ),
        "ari_vs_true": safe_ari(result.labels, dataset.y_true),
        "hai": hai,
    }
    row.update(mst_comparison)
    return row


def run_batch(
    batch: Batch,
    *,
    methods: list[str],
    k_min: int,
    k_max: int,
    random_state: int,
    approx_knn_kwargs: dict[str, object],
) -> list[dict[str, object]]:
    dataset = batch.dataset
    effective_k_max = min(k_max, dataset.X.shape[0] - 1)
    if effective_k_max < k_min:
        raise ValueError(f"{dataset.name} has too few samples for k_min={k_min}.")

    LOGGER.info(
        "Starting quality batch | dataset=%s | n=%s | d=%s | k=%s..%s",
        dataset.name,
        dataset.X.shape[0],
        dataset.X.shape[1],
        k_min,
        effective_k_max,
    )

    method_random_state = dataset.seed if dataset.seed is not None else random_state
    reusable: dict[str, tuple[CoreSG | None, float, str, str]] = {}
    for method in methods:
        if method in {"score_sg", "score_sg_random"}:
            reusable[method] = build_reusable_method(
                method,
                dataset.X,
                k_max=effective_k_max,
                random_state=method_random_state,
                approx_knn_kwargs=approx_knn_kwargs,
            )
            LOGGER.info(
                "Built %s | dataset=%s | status=%s",
                METHOD_DISPLAY_NAMES[method],
                dataset.name,
                reusable[method][2],
            )

    rows: list[dict[str, object]] = []
    for k in range(effective_k_max, k_min - 1, -1):
        LOGGER.info("Running reference HDBSCAN | dataset=%s | k=%s", dataset.name, k)
        reference = fit_hdbscan(dataset.X, k=k, algorithm="generic")

        for method in methods:
            if method == "hdbscan_generic":
                result = reference
            elif method == "optimized_hdbscan":
                LOGGER.info(
                    "Running Optimized HDBSCAN | dataset=%s | k=%s", dataset.name, k
                )
                result = fit_hdbscan(dataset.X, k=k, algorithm="prims_kdtree")
            elif method in {"score_sg", "score_sg_random"}:
                core, fit_seconds, status, error = reusable[method]
                result = extract_reusable_result(
                    core,
                    method=method,
                    k=k,
                    k_max=effective_k_max,
                    fit_seconds=fit_seconds,
                    build_status=status,
                    build_error=error,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            rows.append(
                result_row(
                    dataset,
                    k=k,
                    k_max=effective_k_max,
                    result=result,
                    reference=reference,
                )
            )

    return rows


def summarize_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if "dataset_config" not in frame.columns:
        frame["dataset_config"] = frame["dataset"]

    group_columns = [
        "benchmark_group",
        "dataset_config",
        "family",
        "distribution",
        "structure",
        "n_samples",
        "n_features",
        "method",
        "method_key",
    ]
    per_seed = (
        frame.groupby([*group_columns, "dataset", "seed"], dropna=False)
        .agg(
            k_values=("k", "count"),
            distinct_k_values=("k", "nunique"),
            successful_k_values=("status", lambda values: int((values == "ok").sum())),
            failed_k_values=("status", lambda values: int((values != "ok").sum())),
            mean_ari_vs_hdbscan_generic=("ari_vs_hdbscan_generic", "mean"),
            min_ari_vs_hdbscan_generic=("ari_vs_hdbscan_generic", "min"),
            mean_ari_vs_true=("ari_vs_true", "mean"),
            min_ari_vs_true=("ari_vs_true", "min"),
            mean_hai=("hai", "mean"),
            min_hai=("hai", "min"),
            mean_total_seconds=("total_seconds", "mean"),
            cumulative_workflow_seconds=("workflow_seconds_contribution", "sum"),
            mean_n_clusters_found=("n_clusters_found", "mean"),
        )
        .reset_index()
    )

    def seed_count(values):
        return int(pd.Series(values).dropna().nunique())

    def std(values):
        return float(pd.Series(values).std(ddof=1))

    grouped = per_seed.groupby(group_columns, dropna=False)
    return grouped.agg(
        k_values=("k_values", "sum"),
        distinct_k_values=("distinct_k_values", "max"),
        seed_values=("seed", seed_count),
        dataset_replicates=("dataset", "nunique"),
        successful_k_values=("successful_k_values", "sum"),
        failed_k_values=("failed_k_values", "sum"),
        mean_ari_vs_hdbscan_generic=("mean_ari_vs_hdbscan_generic", "mean"),
        std_ari_vs_hdbscan_generic=("mean_ari_vs_hdbscan_generic", std),
        min_ari_vs_hdbscan_generic=("min_ari_vs_hdbscan_generic", "min"),
        mean_ari_vs_true=("mean_ari_vs_true", "mean"),
        std_ari_vs_true=("mean_ari_vs_true", std),
        min_ari_vs_true=("min_ari_vs_true", "min"),
        mean_hai=("mean_hai", "mean"),
        std_hai=("mean_hai", std),
        min_hai=("min_hai", "min"),
        mean_total_seconds=("mean_total_seconds", "mean"),
        std_total_seconds=("mean_total_seconds", std),
        cumulative_workflow_seconds=("cumulative_workflow_seconds", "sum"),
        mean_cumulative_workflow_seconds=("cumulative_workflow_seconds", "mean"),
        std_cumulative_workflow_seconds=("cumulative_workflow_seconds", std),
        mean_workflow_seconds=("cumulative_workflow_seconds", "mean"),
        std_workflow_seconds=("cumulative_workflow_seconds", std),
        mean_n_clusters_found=("mean_n_clusters_found", "mean"),
        std_n_clusters_found=("mean_n_clusters_found", std),
    ).reset_index()


def write_batch_outputs(batch: Batch, rows: list[dict[str, object]]) -> None:
    batch.output_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    summary = summarize_rows(rows)
    detail.to_csv(batch.output_dir / "quality_comparison_by_k.csv", index=False)
    summary.to_csv(batch.output_dir / "quality_comparison_summary.csv", index=False)


def merge_outputs(output_dir: Path) -> dict[str, int]:
    run_root = output_dir / "_runs"
    detail_frames = []
    for path in sorted(run_root.glob("*/quality_comparison_by_k.csv")):
        detail_frames.append(pd.read_csv(path))

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_rows = 0
    summary_rows = 0
    if detail_frames:
        detail = pd.concat(detail_frames, ignore_index=True)
        if "dataset_config" not in detail.columns:
            detail["dataset_config"] = detail["dataset"]
        detail.to_csv(output_dir / "quality_comparison_by_k.csv", index=False)
        detail_rows = int(detail.shape[0])
        summary = summarize_rows(detail.to_dict("records"))
        summary.to_csv(output_dir / "quality_comparison_summary.csv", index=False)
        summary_rows = int(summary.shape[0])
    return {"detail_rows": detail_rows, "summary_rows": summary_rows}


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(args.log_level)
    validate_args(args)
    approx_knn_kwargs = json.loads(args.approx_knn_kwargs_json)
    if not isinstance(approx_knn_kwargs, dict):
        raise ValueError("--approx-knn-kwargs-json must decode to a JSON object.")

    batches = make_batches(args)
    LOGGER.info("Prepared %s one-by-one quality batches.", len(batches))
    completed = 0
    failed = 0
    skipped = 0

    for index, batch in enumerate(batches, start=1):
        done_path = batch.output_dir / "done.json"
        failed_path = batch.output_dir / "failed.json"
        if args.resume and done_path.exists():
            skipped += 1
            LOGGER.info(
                "Skipping completed batch %s/%s: %s", index, len(batches), batch.name
            )
            continue

        batch.output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Running batch %s/%s: %s", index, len(batches), batch.name)
        start = perf_counter()
        try:
            rows = run_batch(
                batch,
                methods=list(args.methods),
                k_min=args.k_min,
                k_max=args.k_max,
                random_state=args.random_state,
                approx_knn_kwargs=approx_knn_kwargs,
            )
            write_batch_outputs(batch, rows)
            elapsed = perf_counter() - start
            if failed_path.exists():
                failed_path.unlink()
            write_json(
                done_path,
                {
                    "batch": batch.name,
                    "elapsed_seconds": elapsed,
                    "rows": len(rows),
                    "dataset": {
                        "name": batch.dataset.name,
                        "n_samples": int(batch.dataset.X.shape[0]),
                        "n_features": int(batch.dataset.X.shape[1]),
                    },
                },
            )
            completed += 1
            LOGGER.info("Completed batch %s in %.2fs", batch.name, elapsed)
        except Exception as exc:  # noqa: BLE001 - batch runner records failures.
            elapsed = perf_counter() - start
            failed += 1
            write_json(
                failed_path,
                {
                    "batch": batch.name,
                    "elapsed_seconds": elapsed,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            LOGGER.exception("Failed batch %s", batch.name)
            if args.stop_on_error:
                raise
        finally:
            merge_info = merge_outputs(args.output_dir)
            write_json(
                args.output_dir / "quality_comparison_manifest.json",
                {
                    "completed_batches_this_run": completed,
                    "failed_batches_this_run": failed,
                    "skipped_batches_this_run": skipped,
                    "prepared_batches": len(batches),
                    "methods": list(args.methods),
                    "benchmark_groups": list(args.benchmark_groups),
                    "k_min": args.k_min,
                    "k_max": args.k_max,
                    "hai_definition": HAI_DEFINITION,
                    **merge_info,
                },
            )

    LOGGER.info(
        "Quality comparison finished | completed=%s | failed=%s | skipped=%s",
        completed,
        failed,
        skipped,
    )


if __name__ == "__main__":
    main()
