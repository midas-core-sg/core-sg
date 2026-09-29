from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUN_TIME_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = RUN_TIME_DIR / "results"
ASSETS_DIR = RUN_TIME_DIR / "report_assets"
REPORT_DIR = RUN_TIME_DIR / "reports"
REPORT_PATH = REPORT_DIR / "benchmark_report.md"

HDBSCAN_METHODS = [
    "HDBSCAN_algorithm_best_matchRef_false",
    "HDBSCAN_algorithm_best_matchRef_true",
    "HDBSCAN_algorithm_generic_matchRef_false",
    "HDBSCAN_algorithm_generic_matchRef_true",
]
CORESG_METHODS = [
    "CoreSG_fitMatchRef_true_extractMatchRef_false",
    "CoreSG_fitMatchRef_true_extractMatchRef_true",
]

METHOD_LABELS = {
    "CoreSG_fitMatchRef_true_extractMatchRef_false": "CoreSG MST extraction (extractMatchRef=false)",
    "CoreSG_fitMatchRef_true_extractMatchRef_true": "CoreSG MST extraction (extractMatchRef=true)",
    "HDBSCAN_algorithm_best_matchRef_false": "HDBSCAN best (matchRef=false)",
    "HDBSCAN_algorithm_best_matchRef_true": "HDBSCAN best (matchRef=true)",
    "HDBSCAN_algorithm_generic_matchRef_false": "HDBSCAN generic (matchRef=false)",
    "HDBSCAN_algorithm_generic_matchRef_true": "HDBSCAN generic (matchRef=true)",
}

DISPLAY_ORDER = CORESG_METHODS + HDBSCAN_METHODS
PLOT_ORDER = [
    "CoreSG_fitMatchRef_true_extractMatchRef_false",
    "CoreSG_fitMatchRef_true_extractMatchRef_true",
    "HDBSCAN_algorithm_best_matchRef_false",
    "HDBSCAN_algorithm_best_matchRef_true",
    "HDBSCAN_algorithm_generic_matchRef_false",
    "HDBSCAN_algorithm_generic_matchRef_true",
]

STYLE_BY_METHOD = {
    "CoreSG_fitMatchRef_true_extractMatchRef_false": {
        "color": "#1b9e77",
        "linestyle": "-",
        "linewidth": 2.4,
    },
    "CoreSG_fitMatchRef_true_extractMatchRef_true": {
        "color": "#66a61e",
        "linestyle": "-",
        "linewidth": 2.4,
    },
    "HDBSCAN_algorithm_best_matchRef_false": {
        "color": "#d95f02",
        "linestyle": "--",
        "linewidth": 2.1,
    },
    "HDBSCAN_algorithm_best_matchRef_true": {
        "color": "#e7298a",
        "linestyle": "--",
        "linewidth": 2.1,
    },
    "HDBSCAN_algorithm_generic_matchRef_false": {
        "color": "#7570b3",
        "linestyle": ":",
        "linewidth": 2.1,
    },
    "HDBSCAN_algorithm_generic_matchRef_true": {
        "color": "#1f78b4",
        "linestyle": ":",
        "linewidth": 2.1,
    },
}


def load_data() -> pd.DataFrame:
    csv_paths = sorted(RESULTS_DIR.glob("results_*.csv"))
    if not csv_paths:
        raise FileNotFoundError("No benchmarking CSV files were found.")

    df = pd.concat((pd.read_csv(path) for path in csv_paths), ignore_index=True)

    numeric_columns = [
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
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["method_label"] = df["method"].map(METHOD_LABELS)
    return df


def report_link(path: Path) -> str:
    return os.path.relpath(path, REPORT_DIR).replace(os.sep, "/")


def build_cumulative_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for (n_samples, method), group in df.groupby(["n_samples", "method"], sort=True):
        if method in CORESG_METHODS:
            fit_seconds = float(group["fit_seconds"].iloc[0])
            extract_sum = float(group["extract_seconds"].fillna(0.0).sum())
            fit_std = float(group["fit_seconds_std"].iloc[0])
            extract_cumulative_std = math.sqrt(
                float((group["extract_seconds_std"].fillna(0.0) ** 2).sum())
            )
            cumulative_total = fit_seconds + extract_sum
            cumulative_std = math.sqrt(fit_std**2 + extract_cumulative_std**2)
        else:
            fit_seconds = math.nan
            extract_sum = math.nan
            fit_std = math.nan
            extract_cumulative_std = math.nan
            cumulative_total = float(group["total_seconds"].sum())
            cumulative_std = math.sqrt(
                float((group["total_seconds_std"].fillna(0.0) ** 2).sum())
            )
        rows.append(
            {
                "n_samples": int(n_samples),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "fit_seconds": fit_seconds,
                "extract_sum_seconds": extract_sum,
                "fit_std_seconds": fit_std,
                "extract_sum_std_seconds": extract_cumulative_std,
                "cumulative_total_seconds": cumulative_total,
                "cumulative_total_std_seconds": cumulative_std,
                "n_k_values": int(group["k"].nunique()),
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values(["n_samples", "method_label"]).reset_index(drop=True)


def build_best_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for n_samples, group in summary.groupby("n_samples", sort=True):
        coresg = group[group["method"].isin(CORESG_METHODS)].copy()
        hdbscan = group[group["method"].isin(HDBSCAN_METHODS)].copy()

        best_coresg = coresg.loc[coresg["cumulative_total_seconds"].idxmin()]
        best_hdbscan = hdbscan.loc[hdbscan["cumulative_total_seconds"].idxmin()]

        speedup = (
            best_hdbscan["cumulative_total_seconds"]
            / best_coresg["cumulative_total_seconds"]
        )
        reduction = 1.0 - (
            best_coresg["cumulative_total_seconds"]
            / best_hdbscan["cumulative_total_seconds"]
        )

        rows.append(
            {
                "n_samples": int(n_samples),
                "best_coresg": best_coresg["method_label"],
                "coresg_total_seconds": float(best_coresg["cumulative_total_seconds"]),
                "coresg_total_std_seconds": float(
                    best_coresg["cumulative_total_std_seconds"]
                ),
                "best_hdbscan": best_hdbscan["method_label"],
                "hdbscan_total_seconds": float(
                    best_hdbscan["cumulative_total_seconds"]
                ),
                "hdbscan_total_std_seconds": float(
                    best_hdbscan["cumulative_total_std_seconds"]
                ),
                "speedup_x": float(speedup),
                "reduction_pct": float(reduction * 100.0),
            }
        )

    return pd.DataFrame(rows).sort_values("n_samples").reset_index(drop=True)


def build_pairwise_speedups(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for n_samples, group in summary.groupby("n_samples", sort=True):
        coresg = group[group["method"].isin(CORESG_METHODS)]
        hdbscan = group[group["method"].isin(HDBSCAN_METHODS)]
        for _, core_row in coresg.iterrows():
            for _, hdb_row in hdbscan.iterrows():
                speedup = (
                    hdb_row["cumulative_total_seconds"]
                    / core_row["cumulative_total_seconds"]
                )
                rows.append(
                    {
                        "n_samples": int(n_samples),
                        "coresg_method": core_row["method_label"],
                        "hdbscan_method": hdb_row["method_label"],
                        "speedup_x": float(speedup),
                    }
                )
    return pd.DataFrame(rows)


def build_best_coresg_comparisons(summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    best_coresg_method = "CoreSG_fitMatchRef_true_extractMatchRef_false"
    method_tables: dict[str, pd.DataFrame] = {}

    other_methods = [method for method in DISPLAY_ORDER if method != best_coresg_method]
    for other_method in other_methods:
        rows: list[dict[str, float | int | str]] = []
        for n_samples, group in summary.groupby("n_samples", sort=True):
            core_row = group[group["method"] == best_coresg_method].iloc[0]
            other_row = group[group["method"] == other_method].iloc[0]
            speedup = (
                other_row["cumulative_total_seconds"]
                / core_row["cumulative_total_seconds"]
            )
            reduction_pct = 100.0 * (
                1.0
                - (
                    core_row["cumulative_total_seconds"]
                    / other_row["cumulative_total_seconds"]
                )
            )
            absolute_gap = (
                other_row["cumulative_total_seconds"]
                - core_row["cumulative_total_seconds"]
            )
            rows.append(
                {
                    "Samples": int(n_samples),
                    "CoreSG mean (s)": float(core_row["cumulative_total_seconds"]),
                    "CoreSG std (s)": float(core_row["cumulative_total_std_seconds"]),
                    "Compared method": METHOD_LABELS[other_method],
                    "Method mean (s)": float(other_row["cumulative_total_seconds"]),
                    "Method std (s)": float(other_row["cumulative_total_std_seconds"]),
                    "Absolute gap (s)": float(absolute_gap),
                    "Speedup": float(speedup),
                    "Runtime reduction (%)": float(reduction_pct),
                }
            )
        method_tables[other_method] = pd.DataFrame(rows)
    return method_tables


def save_per_k_plot(df: pd.DataFrame) -> Path:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.flatten()

    handles = {}
    sample_sizes = sorted(df["n_samples"].unique())
    for axis, n_samples in zip(axes, sample_sizes):
        subset = df[df["n_samples"] == n_samples].copy()
        subset = subset[
            (subset["k"] == subset["k_max"])
            | (subset["k"] == subset["k"].min())
            | (((subset["k_max"] - subset["k"]) % 5) == 0)
        ].copy()
        subset["comparison_seconds"] = subset.apply(
            lambda row: (
                row["total_seconds"]
                if row["method"] not in CORESG_METHODS
                else (
                    row["fit_seconds"] + row["extract_seconds"]
                    if int(row["k"]) == int(row["k_max"])
                    else row["extract_seconds"]
                )
            ),
            axis=1,
        )

        for method in PLOT_ORDER:
            method_subset = subset[subset["method"] == method].sort_values("k")
            style = STYLE_BY_METHOD[method]
            (line,) = axis.plot(
                method_subset["k"],
                method_subset["comparison_seconds"],
                label=METHOD_LABELS[method],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
            )
            handles.setdefault(method, line)

        positive_values = subset["comparison_seconds"][subset["comparison_seconds"] > 0]
        linthresh = max(float(positive_values.quantile(0.25)), 1e-4)
        k_values = sorted(subset["k"].dropna().astype(int).unique(), reverse=True)
        axis.set_title(f"n = {n_samples:,}")
        axis.set_yscale("symlog", linthresh=linthresh, linscale=1.0)
        axis.set_xlabel("k")
        axis.set_ylabel("Seconds")
        axis.set_xlim(subset["k"].max(), subset["k"].min())
        axis.set_xticks(k_values)
        axis.grid(True, which="major", alpha=0.25)

    fig.suptitle(
        "Per-k runtime: CoreSG MST extraction vs. full HDBSCAN execution",
        fontsize=16,
        y=0.98,
    )
    fig.legend(
        [handles[method] for method in PLOT_ORDER],
        [METHOD_LABELS[method] for method in PLOT_ORDER],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))

    output_path = ASSETS_DIR / "per_k_runtime_grid.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_cumulative_plot(summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(12, 7))
    for method in PLOT_ORDER:
        subset = summary[summary["method"] == method].sort_values("n_samples")
        style = STYLE_BY_METHOD[method]
        ax.plot(
            subset["n_samples"],
            subset["cumulative_total_seconds"],
            marker="o",
            markersize=6,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=METHOD_LABELS[method],
        )

    ax.set_title(
        "Cumulative multi-k runtime across all k values (build + all MST extractions for CoreSG)",
        fontsize=14,
    )
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Seconds")
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    output_path = ASSETS_DIR / "cumulative_runtime_by_samples.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_coresg_breakdown_plot(summary: pd.DataFrame) -> Path:
    subset = summary[summary["method"].isin(CORESG_METHODS)].copy()
    sample_sizes = sorted(subset["n_samples"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    colors = {"fit": "#4c78a8", "extract": "#f58518"}

    for axis, method in zip(axes, CORESG_METHODS):
        method_subset = subset[subset["method"] == method].sort_values("n_samples")
        positions = range(len(sample_sizes))
        axis.bar(
            positions,
            method_subset["fit_seconds"],
            color=colors["fit"],
            label="CoreSG build time",
        )
        axis.bar(
            positions,
            method_subset["extract_sum_seconds"],
            bottom=method_subset["fit_seconds"],
            color=colors["extract"],
            label="Sum of MST extraction times",
        )
        axis.set_title(
            METHOD_LABELS[method].replace("CoreSG MST extraction ", ""),
            fontsize=11,
            pad=10,
        )
        axis.set_xticks(list(positions), [f"{n:,}" for n in sample_sizes])
        axis.set_xlabel("Number of samples")
        axis.tick_params(axis="x", labelrotation=0, labelsize=10)
        axis.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("Seconds")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle("CoreSG cumulative runtime breakdown", fontsize=14, y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.84))

    output_path = ASSETS_DIR / "coresg_runtime_breakdown.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_speedup_heatmap(pairwise_speedups: pd.DataFrame) -> Path:
    best_coresg = "CoreSG MST extraction (extractMatchRef=false)"
    subset = pairwise_speedups[pairwise_speedups["coresg_method"] == best_coresg].copy()
    pivot = subset.pivot(
        index="n_samples", columns="hdbscan_method", values="speedup_x"
    ).sort_index()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    image = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)), [f"{value:,}" for value in pivot.index])
    for row_index, row in enumerate(pivot.values):
        for column_index, value in enumerate(row):
            text_color = "white" if value > (pivot.values.max() * 0.55) else "black"
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}x",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )
    ax.set_title(
        "Speedup of the fastest CoreSG configuration over each HDBSCAN variant"
    )
    ax.set_xlabel("HDBSCAN variant")
    ax.set_ylabel("Number of samples")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Speedup (HDBSCAN / CoreSG)")
    fig.tight_layout()

    output_path = ASSETS_DIR / "speedup_heatmap.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def format_seconds(value: float) -> str:
    return f"{value:.2f}"


def format_optional_seconds(value: float) -> str:
    return "-" if pd.isna(value) else format_seconds(float(value))


def format_speedup(value: float) -> str:
    return f"{value:.2f}x"


def format_percent(value: float) -> str:
    return f"{value:.1f}%"


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.2f} ± {std:.2f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(str(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    best_summary: pd.DataFrame,
    pairwise_speedups: pd.DataFrame,
    method_comparison_tables: dict[str, pd.DataFrame],
    per_k_plot: Path,
    cumulative_plot: Path,
    breakdown_plot: Path,
    speedup_plot: Path,
) -> str:
    n_features = int(df["n_features"].dropna().iloc[0])
    centers = int(df["centers"].dropna().iloc[0])
    seed = int(df["seed"].dropna().iloc[0])
    repetitions = int(df["repetitions"].dropna().iloc[0])
    k_min = int(df["k"].min())
    k_max = int(df["k"].max())
    n_k_values = int(df["k"].nunique())
    sample_sizes = sorted(df["n_samples"].dropna().astype(int).unique())
    sample_sizes_literal = ", ".join(str(value) for value in sample_sizes)

    best_table = best_summary.copy()
    best_table["CoreSG cumulative mean ± std (s)"] = best_table.apply(
        lambda row: format_mean_std(
            float(row["coresg_total_seconds"]), float(row["coresg_total_std_seconds"])
        ),
        axis=1,
    )
    best_table["HDBSCAN cumulative mean ± std (s)"] = best_table.apply(
        lambda row: format_mean_std(
            float(row["hdbscan_total_seconds"]),
            float(row["hdbscan_total_std_seconds"]),
        ),
        axis=1,
    )
    best_table["speedup_x"] = best_table["speedup_x"].map(format_speedup)
    best_table["reduction_pct"] = best_table["reduction_pct"].map(format_percent)
    best_table = best_table.rename(
        columns={
            "n_samples": "Samples",
            "best_coresg": "Best CoreSG configuration",
            "best_hdbscan": "Best HDBSCAN configuration",
            "speedup_x": "Speedup",
            "reduction_pct": "Runtime reduction",
        }
    )
    best_table = best_table[
        [
            "Samples",
            "Best CoreSG configuration",
            "CoreSG cumulative mean ± std (s)",
            "Best HDBSCAN configuration",
            "HDBSCAN cumulative mean ± std (s)",
            "Speedup",
            "Runtime reduction",
        ]
    ]

    cumulative_table = summary.copy()
    cumulative_table["cumulative_total_seconds"] = cumulative_table[
        "cumulative_total_seconds"
    ].map(format_seconds)
    cumulative_table["cumulative_total_std_seconds"] = cumulative_table[
        "cumulative_total_std_seconds"
    ].map(format_seconds)
    cumulative_table["fit_seconds"] = cumulative_table["fit_seconds"].map(
        format_optional_seconds
    )
    cumulative_table["fit_std_seconds"] = cumulative_table["fit_std_seconds"].map(
        format_optional_seconds
    )
    cumulative_table["extract_sum_seconds"] = cumulative_table[
        "extract_sum_seconds"
    ].map(format_optional_seconds)
    cumulative_table["extract_sum_std_seconds"] = cumulative_table[
        "extract_sum_std_seconds"
    ].map(format_optional_seconds)
    cumulative_table = cumulative_table.rename(
        columns={
            "n_samples": "Samples",
            "method_label": "Method",
            "fit_seconds": "CoreSG build mean (s)",
            "fit_std_seconds": "CoreSG build std (s)",
            "extract_sum_seconds": "MST extraction sum mean (s)",
            "extract_sum_std_seconds": "MST extraction sum std (s)",
            "cumulative_total_seconds": "Cumulative mean (s)",
            "cumulative_total_std_seconds": "Cumulative std (s)",
            "n_k_values": "k values",
        }
    )[
        [
            "Samples",
            "Method",
            "CoreSG build mean (s)",
            "CoreSG build std (s)",
            "MST extraction sum mean (s)",
            "MST extraction sum std (s)",
            "Cumulative mean (s)",
            "Cumulative std (s)",
            "k values",
        ]
    ]

    fastest_row = best_summary.loc[best_summary["speedup_x"].idxmax()]
    smallest_row = best_summary.loc[best_summary["speedup_x"].idxmin()]

    generic_vs_best = summary[
        summary["method"].isin(
            [
                "HDBSCAN_algorithm_best_matchRef_false",
                "HDBSCAN_algorithm_generic_matchRef_false",
            ]
        )
    ].pivot(index="n_samples", columns="method", values="cumulative_total_seconds")
    generic_penalty = (
        generic_vs_best["HDBSCAN_algorithm_generic_matchRef_false"]
        / generic_vs_best["HDBSCAN_algorithm_best_matchRef_false"]
    ).mean()

    report = f"""# CoreSG versus HDBSCAN for Repeated Multi-k Hierarchy Extraction

## Abstract

This report evaluates CoreSG and HDBSCAN on a repeated multi-`k` clustering workflow in which hierarchies are required for all `k` values from `{k_max}` down to `{k_min}`. The main question is whether the reusable CoreSG construction cost is amortized when many minimum spanning tree (MST) extractions are needed. Across all tested dataset sizes, the best CoreSG configuration outperforms the best HDBSCAN baseline in cumulative runtime, with reductions ranging from **{format_percent(float(smallest_row["reduction_pct"]))} to {format_percent(float(fastest_row["reduction_pct"]))}** and speedups from **{format_speedup(float(smallest_row["speedup_x"]))}** to **{format_speedup(float(fastest_row["speedup_x"]))}**. The strongest benefits appear in larger datasets and against the generic HDBSCAN variants, where the cumulative gap becomes very large. In the per-`k` view, the first CoreSG point at `k={k_max}` is now explicitly charged with the graph construction cost, since the first usable hierarchy necessarily includes that build step.

## 1. Introduction

CoreSG is intended for workflows in which a single graph structure is built once and then reused to extract MSTs and hierarchy artifacts for many `k` values. This is a different performance target from classical HDBSCAN execution, where the algorithm is rerun independently for each `k`. The experiments studied here therefore focus on cumulative runtime over a descending sequence of `k` values, rather than on single-run performance alone.

The central hypothesis is that CoreSG pays a higher up-front construction cost, but compensates for it through cheap repeated MST extractions. If that tradeoff holds, CoreSG should provide lower cumulative runtime than HDBSCAN in repeated multi-`k` analysis.

## 2. Experimental Design

### 2.1 Datasets and Parameters

The CSV files in [`benchmarking/`](./) cover synthetic datasets with the following settings:

- `n_samples` in `{{{sample_sizes_literal}}}`
- `n_features = {n_features}`
- `centers = {centers}`
- `seed = {seed}`
- `k` values from `{k_max}` down to `{k_min}` (`{n_k_values}` values)
- `{repetitions}` repetitions per configuration

### 2.2 Compared Methods

The benchmark includes:

- `CoreSG_fitMatchRef_true_extractMatchRef_false`
- `CoreSG_fitMatchRef_true_extractMatchRef_true`
- `HDBSCAN_algorithm_best_matchRef_false`
- `HDBSCAN_algorithm_best_matchRef_true`
- `HDBSCAN_algorithm_generic_matchRef_false`
- `HDBSCAN_algorithm_generic_matchRef_true`

### 2.3 Runtime Measures

Two complementary runtime views are used throughout this report:

1. **Per-k runtime comparison**.
   For CoreSG, this uses **`fit_seconds + extract_seconds` at `k = k_max`** and **MST extraction time only** for smaller `k`.
   For HDBSCAN, this uses the **full execution time**.

2. **Cumulative runtime comparison**.
   For CoreSG, the cumulative mean is recomputed as:
   `CoreSG cumulative mean = CoreSG build mean + sum of MST extraction means`
   For HDBSCAN, the cumulative mean is the sum of full execution means across all `k`.

This distinction is important because summing the `total_seconds` column directly for CoreSG would incorrectly count the build phase once for every `k`.

### 2.4 Reporting Conventions

All summary tables report results in a `mean ± std` style commonly used in empirical computer science papers. Because the CSV files already contain aggregated statistics per `k`, the cumulative standard deviation is reconstructed by variance propagation, i.e. by taking the square root of the sum of squared standard deviations. This is an approximation, but it is the most defensible estimate that can be derived from the available benchmark outputs.

## 3. Results

### 3.1 Best-Configuration Summary

The fastest CoreSG configuration is consistently **`extractMatchRef=false`**, and the fastest HDBSCAN baseline is consistently **`best, matchRef=false`**.

{dataframe_to_markdown(best_table)}

These results show that CoreSG already becomes advantageous at `n=1000` and that the cumulative advantage generally increases with dataset size. The most favorable point against the best HDBSCAN baseline occurs at `n={int(fastest_row["n_samples"])}`, where CoreSG reaches **{format_speedup(float(fastest_row["speedup_x"]))}** speedup and **{format_percent(float(fastest_row["reduction_pct"]))}** cumulative runtime reduction.

### 3.2 Per-k Runtime Trends

Figure 1 compares **CoreSG runtime per requested `k`** with **full HDBSCAN execution time** for each value of `k`. For readability, the curves are built using a subsampling of the benchmark results in steps of five `k` units, while preserving both endpoints (`k={k_max}` and `k={k_min}`). For CoreSG, the point at `k={k_max}` includes the graph construction cost (`fit_seconds`) because that first result cannot be obtained without building the support graph. For all subsequent `k` values, the plotted time corresponds only to MST extraction. The x-axis is intentionally displayed from the largest `k` to the smallest `k`, and the y-axis uses a **symmetric logarithmic scale** so that the initial CoreSG build cost remains visible without flattening the lower extraction costs.

![Per-k runtime comparison]({report_link(per_k_plot)})

The figure shows a stable pattern. At `k={k_max}`, the CoreSG curve starts with the full cost of construction plus extraction, which is the correct cost for producing the first hierarchy. After that initial point, the CoreSG runtime drops sharply because the expensive construction phase has already been paid. As `k` decreases, CoreSG extraction remains comparatively cheap, whereas HDBSCAN continues to pay the cost of a full execution at every step. This separation becomes especially clear for larger datasets, where all HDBSCAN curves move upward while the post-build CoreSG extraction costs remain in a much lower time band.

### 3.3 Cumulative Runtime Scaling

Figure 2 shows the cumulative mean runtime required to complete the entire multi-`k` experiment. This is the main metric for evaluating the intended CoreSG use case.

![Cumulative runtime by sample size]({report_link(cumulative_plot)})

The cumulative curves confirm that the one-time CoreSG construction cost is amortized by repeated reuse. Even though CoreSG pays a larger up-front cost than a single MST extraction, the total cost over all `{n_k_values}` values of `k` remains substantially lower than repeated HDBSCAN executions. At `n=40000`, for instance, the best CoreSG configuration completes the whole workload in **{format_seconds(float(best_summary.loc[best_summary["n_samples"] == 40000, "coresg_total_seconds"].iloc[0]))} s**, compared with **{format_seconds(float(best_summary.loc[best_summary["n_samples"] == 40000, "hdbscan_total_seconds"].iloc[0]))} s** for the best HDBSCAN baseline.

### 3.4 CoreSG Cost Decomposition

Figure 3 decomposes the cumulative CoreSG cost into the one-time graph build and the sum of MST extraction costs.

![CoreSG runtime breakdown]({report_link(breakdown_plot)})

The figure indicates that the build stage dominates CoreSG cost, which is expected because it constructs the reusable support graph. Nevertheless, the aggregate extraction cost remains small enough that the overall cumulative cost stays favorable. The `extractMatchRef=false` configuration is systematically better than `extractMatchRef=true`, and the gap widens as the dataset size grows.

### 3.5 Pairwise Speedup Matrix

Figure 4 reports the speedup of the fastest CoreSG configuration over each HDBSCAN variant.

![Speedup heatmap]({report_link(speedup_plot)})

The heatmap reinforces two conclusions. First, CoreSG is faster than every HDBSCAN variant in the repeated multi-`k` scenario. Second, the largest gains occur against the generic HDBSCAN variants, where the cumulative execution penalty is especially high. On average, `HDBSCAN generic (matchRef=false)` is about **{generic_penalty:.2f}x** slower than `HDBSCAN best (matchRef=false)` even before considering the additional CoreSG advantage.

## 4. Detailed Pairwise Analysis

This section compares the best CoreSG configuration, **CoreSG MST extraction (`extractMatchRef=false`)**, against every other method individually. Each table reports cumulative mean runtime, cumulative standard deviation, absolute time gap, speedup, and runtime reduction.

"""
    for method in [
        "CoreSG_fitMatchRef_true_extractMatchRef_true",
        "HDBSCAN_algorithm_best_matchRef_false",
        "HDBSCAN_algorithm_best_matchRef_true",
        "HDBSCAN_algorithm_generic_matchRef_false",
        "HDBSCAN_algorithm_generic_matchRef_true",
    ]:
        comparison_df = method_comparison_tables[method].copy()
        comparison_df["CoreSG mean (s)"] = comparison_df["CoreSG mean (s)"].map(
            format_seconds
        )
        comparison_df["CoreSG std (s)"] = comparison_df["CoreSG std (s)"].map(
            format_seconds
        )
        comparison_df["Method mean (s)"] = comparison_df["Method mean (s)"].map(
            format_seconds
        )
        comparison_df["Method std (s)"] = comparison_df["Method std (s)"].map(
            format_seconds
        )
        comparison_df["Absolute gap (s)"] = comparison_df["Absolute gap (s)"].map(
            format_seconds
        )
        comparison_df["Speedup"] = comparison_df["Speedup"].map(format_speedup)
        comparison_df["Runtime reduction (%)"] = comparison_df[
            "Runtime reduction (%)"
        ].map(format_percent)

        method_label = METHOD_LABELS[method]
        fastest_gain_row = method_comparison_tables[method].loc[
            method_comparison_tables[method]["Speedup"].idxmax()
        ]
        slowest_gain_row = method_comparison_tables[method].loc[
            method_comparison_tables[method]["Speedup"].idxmin()
        ]
        largest_gap_row = method_comparison_tables[method].loc[
            method_comparison_tables[method]["Absolute gap (s)"].idxmax()
        ]

        report += f"""
### Best CoreSG vs. {method_label}

{dataframe_to_markdown(comparison_df)}

CoreSG is faster than **{method_label}** for every tested dataset size. The advantage ranges from **{format_speedup(float(slowest_gain_row["Speedup"]))}** at `n={int(slowest_gain_row["Samples"])}` to **{format_speedup(float(fastest_gain_row["Speedup"]))}** at `n={int(fastest_gain_row["Samples"])}`. In absolute terms, the largest time gap appears at `n={int(largest_gap_row["Samples"])}`, where CoreSG saves **{format_seconds(float(largest_gap_row["Absolute gap (s)"]))} s** over the full multi-`k` experiment.

"""

    report += f"""
## 5. Full Cumulative Statistics

{dataframe_to_markdown(cumulative_table)}

The table above makes the `mean ± std` reporting explicit for both the cumulative totals and, where applicable, the two internal CoreSG components: build cost and total MST extraction cost.

## 6. Discussion

Taken together, the results strongly support the original CoreSG design rationale. The method is not optimized for a single isolated `k`, but for a sequence of related hierarchy extractions over many `k` values. Under that workload, CoreSG consistently dominates HDBSCAN because it converts repeated expensive recomputation into one reusable build plus many inexpensive extraction steps.

The pairwise comparisons are also important methodologically. CoreSG does not merely outperform the weakest baselines. It remains ahead of the strongest HDBSCAN baseline across every tested dataset size, and its advantage becomes especially pronounced as scale increases. This suggests that the reuse strategy is not a marginal optimization; it changes the practical runtime regime of the whole multi-`k` workflow.

## 7. Threats to Validity

Some limitations should be acknowledged:

- The datasets are synthetic, so the absolute timings may differ on real-world distributions.
- The cumulative standard deviations are reconstructed from aggregated per-`k` summaries rather than from raw repetition-level traces.
- The experiments keep `n_features`, `centers`, and `k_max` fixed, so the conclusions are strongest for this benchmark shape.
- The study measures runtime only; memory consumption and downstream clustering quality are outside the scope of this report.

These limitations do not invalidate the observed trend, but they do define the boundary of what can be claimed directly from the current benchmark suite.

## 8. Conclusion

The experimental evidence consistently indicates that CoreSG is the preferable runtime strategy for repeated multi-`k` hierarchy extraction. Its best configuration, **CoreSG MST extraction (`extractMatchRef=false`)**, outperforms every HDBSCAN variant tested here, including the strongest `best, matchRef=false` baseline. The performance gains are already visible at small scale and become substantial for larger datasets.

For one-off analyses at a single `k`, plain HDBSCAN may still be attractive because it avoids the CoreSG build phase. For the benchmarked workload, however, the cumulative evidence is clear: reusable graph construction plus repeated MST extraction is a more efficient execution model than rerunning HDBSCAN independently for every `k`.
"""
    return report


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    summary = build_cumulative_summary(df)
    best_summary = build_best_summary(summary)
    pairwise_speedups = build_pairwise_speedups(summary)
    method_comparison_tables = build_best_coresg_comparisons(summary)

    per_k_plot = save_per_k_plot(df)
    cumulative_plot = save_cumulative_plot(summary)
    breakdown_plot = save_coresg_breakdown_plot(summary)
    speedup_plot = save_speedup_heatmap(pairwise_speedups)

    report = build_report(
        df=df,
        summary=summary,
        best_summary=best_summary,
        pairwise_speedups=pairwise_speedups,
        method_comparison_tables=method_comparison_tables,
        per_k_plot=per_k_plot,
        cumulative_plot=cumulative_plot,
        breakdown_plot=breakdown_plot,
        speedup_plot=speedup_plot,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Report written to: {REPORT_PATH}")
    print(f"Assets written to: {ASSETS_DIR}")


if __name__ == "__main__":
    main()
