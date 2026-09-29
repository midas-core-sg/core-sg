from __future__ import annotations

import argparse
import csv
import logging
import re
import statistics
import sys
from pathlib import Path
from time import perf_counter

import hdbscan
from sklearn.datasets import make_blobs

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core_sg import CoreSGClusterer  # noqa: E402

LOGGER = logging.getLogger("benchmarking.script")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark HDBSCAN and CoreSG variants for k values up to k_max "
            "and save execution times to a CSV file."
        )
    )
    parser.add_argument(
        "--label",
        required=True,
        help=(
            "Logical label for this benchmark run. If it contains 'nocython', "
            "the CoreSG variants are written as NoCython in the CSV."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarking/run_time/results"),
        help="Directory where the CSV file will be saved.",
    )
    parser.add_argument("--n-samples", type=int, default=15000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--centers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--k-max",
        type=int,
        default=50,
        help="Maximum k used in the benchmark. The script runs k from 2 to k_max.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=30,
        help="Number of executions used to compute the average time for each variant.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity for benchmark progress messages.",
    )
    return parser.parse_args()


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned or "benchmark"


def core_sg_prefix(label: str) -> str:
    return "CoreSGNoCython" if "nocython" in label.lower() else "CoreSG"


def make_dataset(
    n_samples: int,
    n_features: int,
    centers: int,
    seed: int,
):
    print(
        f"Generating synthetic dataset: n={n_samples}, d={n_features}, "
        f"centers={centers}, seed={seed}"
    )
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        random_state=seed,
    )
    return X


def run_hdbscan_variant(
    *,
    X,
    ks: list[int],
    algorithm: str,
    match_reference_implementation: bool,
    label: str,
    repetitions: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    method_name = (
        f"HDBSCAN_algorithm_{algorithm}_"
        f"matchRef_{str(match_reference_implementation).lower()}"
    )
    LOGGER.info("Starting variant: %s", method_name)

    for k in ks:
        LOGGER.info(
            "Running %s for k=%s with %s repetitions", method_name, k, repetitions
        )
        fit_samples: list[float] = []
        n_clusters_found = None

        for repetition in range(1, repetitions + 1):
            start = perf_counter()
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=k,
                min_samples=k,
                metric="euclidean",
                algorithm=algorithm,
                approx_min_span_tree=False,
                gen_min_span_tree=True,
                match_reference_implementation=match_reference_implementation,
            ).fit(X)
            elapsed = perf_counter() - start
            fit_samples.append(elapsed)
            n_clusters_found = len(set(clusterer.labels_)) - int(
                -1 in clusterer.labels_
            )
            LOGGER.info(
                "Finished %s | k=%s | repetition=%s/%s | fit=%.6fs",
                method_name,
                k,
                repetition,
                repetitions,
                elapsed,
            )

        rows.append(
            {
                "label": label,
                "method": method_name,
                "family": "HDBSCAN",
                "algorithm": algorithm,
                "fit_match_reference_implementation": match_reference_implementation,
                "extract_match_reference_implementation": "",
                "approx_min_span_tree": False,
                "k": k,
                "fit_seconds": statistics.mean(fit_samples),
                "extract_seconds": "",
                "total_seconds": statistics.mean(fit_samples),
                "fit_seconds_std": statistics.pstdev(fit_samples)
                if len(fit_samples) > 1
                else 0.0,
                "extract_seconds_std": "",
                "total_seconds_std": statistics.pstdev(fit_samples)
                if len(fit_samples) > 1
                else 0.0,
                "repetitions": repetitions,
                "n_clusters_found": n_clusters_found,
            }
        )
        LOGGER.info(
            "Completed %s | k=%s | mean_fit=%.6fs | std_fit=%.6fs",
            method_name,
            k,
            statistics.mean(fit_samples),
            statistics.pstdev(fit_samples) if len(fit_samples) > 1 else 0.0,
        )

    return rows


def run_core_sg_variant(
    *,
    X,
    ks: list[int],
    k_max: int,
    fit_match_reference_implementation: bool,
    extract_match_reference_implementation: bool,
    label: str,
    repetitions: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    prefix = core_sg_prefix(label)
    method_name = (
        f"{prefix}_fitMatchRef_{str(fit_match_reference_implementation).lower()}_"
        f"extractMatchRef_{str(extract_match_reference_implementation).lower()}"
    )
    LOGGER.info("Starting variant: %s", method_name)
    fit_samples: list[float] = []
    extract_samples_by_k: dict[int, list[float]] = {k: [] for k in ks}
    n_clusters_by_k: dict[int, int] = {}

    for repetition in range(1, repetitions + 1):
        clusterer = CoreSGClusterer(
            k_max=k_max,
            metric="euclidean",
            p=2,
            verbose=0,
            no_noise=False,
            match_reference_implementation=fit_match_reference_implementation,
        )

        fit_start = perf_counter()
        clusterer.fit(X, k=k_max)
        fit_elapsed = perf_counter() - fit_start
        fit_samples.append(fit_elapsed)
        core = clusterer.core_sg_
        LOGGER.info(
            "Finished %s | fit repetition=%s/%s | fit=%.6fs",
            method_name,
            repetition,
            repetitions,
            fit_elapsed,
        )

        for k in ks:
            core.hdbscan_kwargs["match_reference_implementation"] = (
                extract_match_reference_implementation
            )

            extract_start = perf_counter()
            core.extract_hierarchy_from_core_sg(k)
            extract_elapsed = perf_counter() - extract_start
            extract_samples_by_k[k].append(extract_elapsed)
            n_clusters_by_k[k] = len(set(core.labels_)) - int(-1 in core.labels_)
            LOGGER.info(
                "Finished %s | k=%s | repetition=%s/%s | extract=%.6fs",
                method_name,
                k,
                repetition,
                repetitions,
                extract_elapsed,
            )

    fit_mean = statistics.mean(fit_samples)
    fit_std = statistics.pstdev(fit_samples) if len(fit_samples) > 1 else 0.0

    for k in ks:
        extract_mean = statistics.mean(extract_samples_by_k[k])
        extract_std = (
            statistics.pstdev(extract_samples_by_k[k])
            if len(extract_samples_by_k[k]) > 1
            else 0.0
        )

        rows.append(
            {
                "label": label,
                "method": method_name,
                "family": prefix,
                "algorithm": "core_sg",
                "fit_match_reference_implementation": fit_match_reference_implementation,
                "extract_match_reference_implementation": extract_match_reference_implementation,
                "approx_min_span_tree": False,
                "k": k,
                "fit_seconds": fit_mean,
                "extract_seconds": extract_mean,
                "total_seconds": fit_mean + extract_mean,
                "fit_seconds_std": fit_std,
                "extract_seconds_std": extract_std,
                "total_seconds_std": (fit_std**2 + extract_std**2) ** 0.5,
                "repetitions": repetitions,
                "n_clusters_found": n_clusters_by_k[k],
            }
        )
        LOGGER.info(
            "Completed %s | k=%s | mean_fit=%.6fs | mean_extract=%.6fs | total=%.6fs",
            method_name,
            k,
            fit_mean,
            extract_mean,
            fit_mean + extract_mean,
        )

    return rows


def write_csv(
    output_path: Path, rows: list[dict[str, object]], metadata: dict[str, object]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "method",
        "family",
        "algorithm",
        "fit_match_reference_implementation",
        "extract_match_reference_implementation",
        "approx_min_span_tree",
        "k",
        "fit_seconds",
        "extract_seconds",
        "total_seconds",
        "fit_seconds_std",
        "extract_seconds_std",
        "total_seconds_std",
        "repetitions",
        "n_clusters_found",
        "n_samples",
        "n_features",
        "centers",
        "seed",
        "k_max",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **metadata})


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.k_max < 2:
        raise ValueError("--k-max must be at least 2.")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1.")

    LOGGER.info(
        "Benchmark started | label=%s | k_max=%s | repetitions=%s",
        args.label,
        args.k_max,
        args.repetitions,
    )

    X = make_dataset(
        n_samples=args.n_samples,
        n_features=args.n_features,
        centers=args.centers,
        seed=args.seed,
    )
    ks = list(range(args.k_max, 1, -1))

    rows: list[dict[str, object]] = []

    rows.extend(
        run_hdbscan_variant(
            X=X,
            ks=ks,
            algorithm="generic",
            match_reference_implementation=True,
            label=args.label,
            repetitions=args.repetitions,
        )
    )
    rows.extend(
        run_hdbscan_variant(
            X=X,
            ks=ks,
            algorithm="generic",
            match_reference_implementation=False,
            label=args.label,
            repetitions=args.repetitions,
        )
    )
    rows.extend(
        run_core_sg_variant(
            X=X,
            ks=ks,
            k_max=args.k_max,
            fit_match_reference_implementation=True,
            extract_match_reference_implementation=True,
            label=args.label,
            repetitions=args.repetitions,
        )
    )
    rows.extend(
        run_core_sg_variant(
            X=X,
            ks=ks,
            k_max=args.k_max,
            fit_match_reference_implementation=True,
            extract_match_reference_implementation=False,
            label=args.label,
            repetitions=args.repetitions,
        )
    )
    rows.extend(
        run_hdbscan_variant(
            X=X,
            ks=ks,
            algorithm="best",
            match_reference_implementation=True,
            label=args.label,
            repetitions=args.repetitions,
        )
    )
    rows.extend(
        run_hdbscan_variant(
            X=X,
            ks=ks,
            algorithm="best",
            match_reference_implementation=False,
            label=args.label,
            repetitions=args.repetitions,
        )
    )

    output_path = (
        args.output_dir / f"{sanitize_filename(args.label)}_{args.n_samples}.csv"
    )
    metadata = {
        "n_samples": args.n_samples,
        "n_features": args.n_features,
        "centers": args.centers,
        "seed": args.seed,
        "k_max": args.k_max,
    }
    write_csv(output_path, rows, metadata)
    LOGGER.info("Saved benchmark results to: %s", output_path)


if __name__ == "__main__":
    main()


def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        temp = arr[min_idx]
        arr[min_idx] = arr[i]
        arr[i] = temp
    return arr
