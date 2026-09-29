from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.missing_edges.missing_edges_connectivity import (  # noqa: E402
    DEFAULT_DISTRIBUTIONS,
    DEFAULT_REAL_DATASETS,
    PAPER_DIMENSIONS,
    PAPER_SAMPLE_SIZES,
    parse_int_list,
    parse_named_path,
    parse_str_list,
    write_outputs,
)


@dataclass(frozen=True)
class Batch:
    name: str
    output_dir: Path
    tie_output_dir: Path
    command: list[str]


def parse_threads(value: str) -> int:
    threads = int(value)
    if threads <= 0:
        raise argparse.ArgumentTypeError("threads must be positive")
    return threads


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the large missing-edge grid one configuration at a time and "
            "merge the resulting CSV files."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarking/missing_edges/results_large"),
        help="Final merged missing-edge output directory.",
    )
    parser.add_argument(
        "--tie-output-dir",
        type=Path,
        default=Path("benchmarking/unties/results_large"),
        help="Final merged tie-diagnostic output directory.",
    )
    parser.add_argument(
        "--sample-sizes",
        type=parse_int_list,
        default=list(PAPER_SAMPLE_SIZES),
    )
    parser.add_argument(
        "--dimensions",
        type=parse_int_list,
        default=list(PAPER_DIMENSIONS),
    )
    parser.add_argument(
        "--distributions",
        type=parse_str_list,
        default=list(DEFAULT_DISTRIBUTIONS),
    )
    parser.add_argument("--seeds", type=parse_int_list, default=[42])
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threads", type=parse_threads, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--no-real", action="store_true")
    parser.add_argument(
        "--only-separated-synthetic",
        action="store_true",
        help="Forwarded to missing_edges_connectivity.py.",
    )
    parser.add_argument(
        "--separated-clusters",
        type=int,
        default=6,
        help="Forwarded to missing_edges_connectivity.py.",
    )
    parser.add_argument(
        "--separated-cluster-scale",
        type=float,
        default=0.65,
        help="Forwarded to missing_edges_connectivity.py.",
    )
    parser.add_argument(
        "--separated-cluster-separation",
        type=float,
        default=8.0,
        help="Forwarded to missing_edges_connectivity.py.",
    )
    parser.add_argument(
        "--real-datasets",
        type=parse_str_list,
        default=list(DEFAULT_REAL_DATASETS),
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
        help="Forwarded to missing_edges_connectivity.py.",
    )
    parser.add_argument("--max-real-samples", type=int, default=None)
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate final merged plots after all CSV files are combined.",
    )
    return parser


def batch_slug(*parts: object) -> str:
    return "-".join(str(part).replace("_", "-") for part in parts)


def base_command(
    args: argparse.Namespace, output_dir: Path, tie_output_dir: Path
) -> list[str]:
    command = [
        sys.executable,
        "benchmarking/missing_edges/missing_edges_connectivity.py",
        "--preset",
        "paper",
        "--output-dir",
        str(output_dir),
        "--tie-output-dir",
        str(tie_output_dir),
        "--k-min",
        str(args.k_min),
        "--k-max",
        str(args.k_max),
        "--random-state",
        str(args.random_state),
        "--seeds",
        ",".join(str(seed) for seed in args.seeds),
        "--separated-clusters",
        str(args.separated_clusters),
        "--separated-cluster-scale",
        str(args.separated_cluster_scale),
        "--separated-cluster-separation",
        str(args.separated_cluster_separation),
        "--no-plots",
        "--log-level",
        "INFO",
    ]
    if args.only_separated_synthetic:
        command.append("--only-separated-synthetic")
    return command


def iter_batches(args: argparse.Namespace) -> list[Batch]:
    run_root = args.output_dir / "_runs"
    tie_run_root = args.tie_output_dir / "_runs"
    batches: list[Batch] = []

    for distribution in args.distributions:
        for n_samples in args.sample_sizes:
            for n_features in args.dimensions:
                slug = batch_slug(
                    "synthetic", distribution, f"n{n_samples}", f"d{n_features}"
                )
                output_dir = run_root / slug
                tie_output_dir = tie_run_root / slug
                command = base_command(args, output_dir, tie_output_dir)
                command.extend(
                    [
                        "--sample-sizes",
                        str(n_samples),
                        "--dimensions",
                        str(n_features),
                        "--distributions",
                        distribution,
                        "--no-real",
                    ]
                )
                batches.append(Batch(slug, output_dir, tie_output_dir, command))

    if not args.no_real:
        for name in args.real_datasets:
            slug = batch_slug("real", name)
            output_dir = run_root / slug
            tie_output_dir = tie_run_root / slug
            command = base_command(args, output_dir, tie_output_dir)
            command.extend(["--no-synthetic", "--real-datasets", name])
            if args.max_real_samples is not None:
                command.extend(["--max-real-samples", str(args.max_real_samples)])
            batches.append(Batch(slug, output_dir, tie_output_dir, command))

        for name, path in args.real_csv:
            slug = batch_slug("real-csv", name)
            output_dir = run_root / slug
            tie_output_dir = tie_run_root / slug
            command = base_command(args, output_dir, tie_output_dir)
            command.extend(
                [
                    "--no-synthetic",
                    "--no-builtin-real",
                    "--real-csv",
                    f"{name}={path}",
                ]
            )
            for drop_column in args.csv_drop_column:
                command.extend(["--csv-drop-column", drop_column])
            if args.max_real_samples is not None:
                command.extend(["--max-real-samples", str(args.max_real_samples)])
            batches.append(Batch(slug, output_dir, tie_output_dir, command))

    if args.limit is not None:
        batches = batches[: args.limit]
    return batches


def limited_thread_env(threads: int) -> dict[str, str]:
    env = os.environ.copy()
    value = str(threads)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        env[key] = value
    return env


def run_batch(batch: Batch, *, args: argparse.Namespace, env: dict[str, str]) -> bool:
    batch.output_dir.mkdir(parents=True, exist_ok=True)
    batch.tie_output_dir.mkdir(parents=True, exist_ok=True)
    done_path = batch.output_dir / "done.json"
    failed_path = batch.output_dir / "failed.json"
    log_path = batch.output_dir / "run.log"

    required_outputs = (
        batch.output_dir / "missing_edges_connectivity_by_k.csv",
        batch.output_dir / "missing_edges_connectivity_summary.csv",
    )
    has_complete_outputs = all(
        path.exists() and path.stat().st_size > 0 for path in required_outputs
    )
    if args.resume and done_path.exists() and has_complete_outputs:
        print(f"[skip] {batch.name}", flush=True)
        return True

    print(f"[run] {batch.name}", flush=True)
    start = perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            batch.command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = perf_counter() - start

    payload = {
        "batch": batch.name,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "command": batch.command,
        "log": str(log_path),
    }
    if completed.returncode == 0:
        done_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if failed_path.exists():
            failed_path.unlink()
        print(f"[done] {batch.name} ({elapsed:.1f}s)", flush=True)
        return True

    failed_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[fail] {batch.name} ({elapsed:.1f}s); see {log_path}", flush=True)
    return False


def read_existing_csv(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def combine_outputs(args: argparse.Namespace, batches: list[Batch]) -> None:
    detail_paths = [
        batch.output_dir / "missing_edges_connectivity_by_k.csv" for batch in batches
    ]
    summary_paths = [
        batch.output_dir / "missing_edges_connectivity_summary.csv" for batch in batches
    ]
    detail = read_existing_csv(detail_paths)
    summary = read_existing_csv(summary_paths)

    write_outputs(
        detail_rows=detail.to_dict(orient="records"),
        summary_rows=summary.to_dict(orient="records"),
        output_dir=args.output_dir,
        tie_output_dir=args.tie_output_dir,
        generate_plots=args.plots,
    )

    failure_records = []
    for batch in batches:
        failed_path = batch.output_dir / "failed.json"
        if failed_path.exists():
            failure_records.append(json.loads(failed_path.read_text(encoding="utf-8")))
    if failure_records:
        pd.DataFrame(failure_records).to_csv(
            args.output_dir / "large_grid_run_failures.csv",
            index=False,
        )
    else:
        stale_failures = args.output_dir / "large_grid_run_failures.csv"
        if stale_failures.exists():
            stale_failures.unlink()

    manifest = {
        "batches": [asdict(batch) | {"command": batch.command} for batch in batches],
        "completed_batches": int(
            sum((batch.output_dir / "done.json").exists() for batch in batches)
        ),
        "failed_batches": int(
            sum((batch.output_dir / "failed.json").exists() for batch in batches)
        ),
        "merged_detail_rows": int(detail.shape[0]),
        "merged_summary_rows": int(summary.shape[0]),
    }
    (args.output_dir / "large_grid_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tie_output_dir.mkdir(parents=True, exist_ok=True)
    batches = iter_batches(args)
    env = limited_thread_env(args.threads)

    print(
        f"Running {len(batches)} batches one at a time with {args.threads} thread(s).",
        flush=True,
    )
    all_ok = True
    for index, batch in enumerate(batches, start=1):
        print(f"[{index}/{len(batches)}]", flush=True)
        ok = run_batch(batch, args=args, env=env)
        all_ok = all_ok and ok
        combine_outputs(args, batches)
        if not ok and args.stop_on_failure:
            raise SystemExit(1)

    combine_outputs(args, batches)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
