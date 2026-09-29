from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.missing_edges.missing_edges_connectivity import (  # noqa: E402
    DEFAULT_DISTRIBUTIONS,
    make_separated_cluster_distribution,
    parse_str_list,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot 2D separated-cluster synthetic datasets for inspection."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarking/missing_edges/figures/separated_datasets_d2.png"),
    )
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-features", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--separation", type=float, default=8.0)
    parser.add_argument("--cluster-scale", type=float, default=0.65)
    parser.add_argument(
        "--distributions",
        type=parse_str_list,
        default=list(DEFAULT_DISTRIBUTIONS),
    )
    parser.add_argument("--point-size", type=float, default=1.8)
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.n_features != 2:
        raise ValueError("This inspection plot is intended for D=2 only.")

    n_distributions = len(args.distributions)
    n_cols = 3
    n_rows = int(np.ceil(n_distributions / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.6 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for ax, distribution in zip(axes.ravel(), args.distributions):
        X = make_separated_cluster_distribution(
            distribution,
            n_samples=args.n_samples,
            n_features=args.n_features,
            seed=args.seed,
            n_clusters=args.n_clusters,
            separation=args.separation,
            cluster_scale=args.cluster_scale,
        )
        ax.scatter(
            X[:, 0],
            X[:, 1],
            s=args.point_size,
            alpha=args.alpha,
            linewidths=0,
            color="#1f2937",
        )
        ax.set_title(distribution.replace("_", " "))
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.35, alpha=0.35)

    for ax in axes.ravel()[n_distributions:]:
        ax.axis("off")

    fig.suptitle(
        (
            "Separated synthetic datasets, D=2 "
            f"(N={args.n_samples}, clusters={args.n_clusters}, "
            f"separation={args.separation}, scale={args.cluster_scale})"
        ),
        fontsize=14,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
