from __future__ import annotations

import csv
import shutil
from pathlib import Path
import math

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import pandas as pd
except Exception:  # pragma: no cover - fallback for broken local scientific envs
    plt = None
    pd = None


ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = ROOT / "docs"
OUT = DOCS_ROOT / "source" / "_static" / "images" / "benchmarks"
ASSETS = ROOT / "benchmarking" / "report_assets"
RESULTS = ROOT / "benchmarking" / "results"


def copy_existing_assets() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in ASSETS.glob("*.png"):
        shutil.copy2(path, OUT / path.name)


def load_results() -> pd.DataFrame:
    if pd is None:
        return None
    frames = []
    for path in sorted(RESULTS.glob("*.csv")):
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            continue
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def generated_summary(df) -> None:
    if df is None or df.empty or plt is None:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (1000, 420), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 980, 400), outline="#24546a", width=3)
        draw.text(
            (55, 80),
            "Benchmark figures are copied from benchmarking/report_assets.",
            fill="#20333d",
        )
        draw.text(
            (55, 125),
            "Regenerate in a full docs environment for CSV-derived plots.",
            fill="#20333d",
        )
        image.save(OUT / "generated_runtime_summary.png")
        return
    sample_col = _first_existing(
        list(df.columns), ["n_samples", "samples", "sample_size"]
    )
    method_col = _first_existing(list(df.columns), ["method", "configuration", "name"])
    time_col = _first_existing(
        list(df.columns),
        ["total_seconds_mean", "mean_seconds", "total_seconds", "fit_seconds_mean"],
    )
    if sample_col is None or method_col is None or time_col is None:
        return
    grouped = df.groupby([sample_col, method_col], as_index=False)[time_col].mean()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for method, part in grouped.groupby(method_col):
        label = str(method)
        if len(label) > 38:
            label = label[:35] + "..."
        ax.plot(part[sample_col], part[time_col], marker="o", label=label)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Mean runtime (s)")
    ax.set_title("Benchmark result summary")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.savefig(OUT / "generated_runtime_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _method_label(method: str) -> str:
    labels = {
        "ScoreSG_extractMatchRef_false": "ScoreSG, optimized extraction",
        "ScoreSG_extractMatchRef_true": "ScoreSG, reference extraction",
        "CoreSG_fitMatchRef_true_extractMatchRef_false": "Exact CoreSG, optimized extraction",
        "CoreSG_fitMatchRef_true_extractMatchRef_true": "Exact CoreSG, reference extraction",
        "HDBSCAN_algorithm_best_matchRef_false": "HDBSCAN best, optimized",
        "HDBSCAN_algorithm_generic_matchRef_false": "HDBSCAN generic, optimized",
    }
    return labels.get(method, method)


def cumulative_results(df):
    if df is None or df.empty or pd is None:
        return None
    required = {
        "n_samples",
        "method",
        "family",
        "k",
        "fit_seconds",
        "extract_seconds",
        "total_seconds",
    }
    if not required.issubset(df.columns):
        return None

    rows = []
    for (n_samples, method, family), part in df.groupby(
        ["n_samples", "method", "family"]
    ):
        part = part.sort_values("k", ascending=False)
        if str(family) in {"CoreSG", "ScoreSG"} or str(method).startswith(
            ("CoreSG", "ScoreSG")
        ):
            fit_seconds = float(part["fit_seconds"].fillna(0).iloc[0])
            extract_seconds = float(part["extract_seconds"].fillna(0).sum())
            total_seconds = fit_seconds + extract_seconds
        else:
            fit_seconds = math.nan
            extract_seconds = math.nan
            total_seconds = float(part["total_seconds"].fillna(0).sum())
        rows.append(
            {
                "n_samples": int(n_samples),
                "method": method,
                "family": family,
                "fit_seconds": fit_seconds,
                "extract_seconds": extract_seconds,
                "total_seconds": total_seconds,
            }
        )
    return pd.DataFrame(rows)


def generated_score_sg_figures(df) -> None:
    cumulative = cumulative_results(df)
    if cumulative is None or cumulative.empty or plt is None:
        return

    common_methods = [
        "ScoreSG_extractMatchRef_false",
        "CoreSG_fitMatchRef_true_extractMatchRef_false",
        "HDBSCAN_algorithm_best_matchRef_false",
        "HDBSCAN_algorithm_generic_matchRef_false",
    ]
    common_ns = [5000, 10000, 20000, 30000, 40000, 50000]
    common = cumulative[
        cumulative["method"].isin(common_methods)
        & cumulative["n_samples"].isin(common_ns)
    ].copy()
    if not common.empty:
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        for method in common_methods:
            part = common[common["method"] == method].sort_values("n_samples")
            if part.empty:
                continue
            ax.plot(
                part["n_samples"],
                part["total_seconds"],
                marker="o",
                linewidth=2.1,
                label=_method_label(method),
            )
        ax.set_xlabel("Samples")
        ax.set_ylabel("Cumulative runtime for all k values (s)")
        ax.set_title("ScoreSG vs exact CoreSG and HDBSCAN")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.28)
        ax.legend(fontsize=8)
        fig.savefig(OUT / "score_sg_common_runtime.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    score_methods = ["ScoreSG_extractMatchRef_false", "ScoreSG_extractMatchRef_true"]
    score = cumulative[cumulative["method"].isin(score_methods)].copy()
    if not score.empty:
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        for method in score_methods:
            part = score[score["method"] == method].sort_values("n_samples")
            if part.empty:
                continue
            ax.plot(
                part["n_samples"],
                part["total_seconds"],
                marker="o",
                linewidth=2.1,
                label=_method_label(method),
            )
        ax.set_xlabel("Samples")
        ax.set_ylabel("Cumulative runtime for all k values (s)")
        ax.set_title("ScoreSG cumulative runtime scaling")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.savefig(OUT / "score_sg_extended_runtime.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def _load_csv_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path in sorted(RESULTS.glob("*.csv")):
        try:
            with path.open(newline="") as handle:
                records.extend(csv.DictReader(handle))
        except Exception:
            continue
    return records


def _cumulative_records() -> list[dict[str, float | str | int]]:
    grouped: dict[tuple[int, str, str], list[dict[str, str]]] = {}
    for row in _load_csv_records():
        try:
            key = (int(row["n_samples"]), row["method"], row["family"])
        except Exception:
            continue
        grouped.setdefault(key, []).append(row)

    cumulative: list[dict[str, float | str | int]] = []
    for (n_samples, method, family), rows in grouped.items():
        rows.sort(key=lambda item: int(item["k"]), reverse=True)
        if family in {"CoreSG", "ScoreSG"} or method.startswith(("CoreSG", "ScoreSG")):
            fit_seconds = float(rows[0].get("fit_seconds") or 0.0)
            extract_seconds = sum(
                float(row.get("extract_seconds") or 0.0) for row in rows
            )
            total_seconds = fit_seconds + extract_seconds
        else:
            total_seconds = sum(float(row.get("total_seconds") or 0.0) for row in rows)
        cumulative.append(
            {
                "n_samples": n_samples,
                "method": method,
                "total_seconds": total_seconds,
            }
        )
    return cumulative


def _draw_pil_line_chart(
    path: Path,
    title: str,
    series: list[tuple[str, list[tuple[int, float]]]],
    *,
    log_y: bool = False,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1200, 720
    left, top, right, bottom = 110, 80, 820, 610
    colors = ["#1b6ca8", "#2d8a42", "#c56a00", "#8b2f97", "#d33f49"]
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        label_font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        title_font = label_font = small_font = None

    points = [(x, y) for _, values in series for x, y in values]
    if not points:
        return
    min_x = min(x for x, _ in points)
    max_x = max(x for x, _ in points)
    raw_min_y = min(y for _, y in points if y > 0)
    raw_max_y = max(y for _, y in points)
    min_y = math.log10(raw_min_y) if log_y else 0.0
    max_y = math.log10(raw_max_y) if log_y else raw_max_y

    def sx(value: int) -> float:
        return left + (value - min_x) / (max_x - min_x) * (right - left)

    def sy(value: float) -> float:
        y_value = math.log10(value) if log_y else value
        return bottom - (y_value - min_y) / (max_y - min_y) * (bottom - top)

    draw.text((left, 28), title, fill="#17242c", font=title_font)
    draw.line((left, bottom, right, bottom), fill="#263238", width=2)
    draw.line((left, top, left, bottom), fill="#263238", width=2)
    draw.text((left + 250, height - 62), "Samples", fill="#263238", font=label_font)
    draw.text((20, top + 205), "Runtime (s)", fill="#263238", font=label_font)

    for tick in [min_x, 50000, 100000, max_x]:
        if tick < min_x or tick > max_x:
            continue
        x = sx(tick)
        draw.line((x, bottom, x, bottom + 7), fill="#263238", width=1)
        draw.text((x - 34, bottom + 14), f"{tick:,}", fill="#263238", font=small_font)

    y_ticks = (
        [10, 100, 1000, 10000]
        if log_y
        else [0, raw_max_y / 4, raw_max_y / 2, raw_max_y * 3 / 4, raw_max_y]
    )
    for tick in y_ticks:
        if tick <= 0 or tick > raw_max_y:
            continue
        y = sy(tick)
        draw.line((left - 7, y, left, y), fill="#263238", width=1)
        draw.line((left, y, right, y), fill="#d8dee3", width=1)
        draw.text((20, y - 9), f"{tick:,.0f}", fill="#263238", font=small_font)

    for idx, (label, values) in enumerate(series):
        color = colors[idx % len(colors)]
        coords = [(sx(x), sy(y)) for x, y in values]
        if len(coords) > 1:
            draw.line(coords, fill=color, width=4)
        for x, y in coords:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        legend_y = top + idx * 34
        draw.line((870, legend_y + 10, 910, legend_y + 10), fill=color, width=4)
        draw.text((922, legend_y), label, fill="#17242c", font=small_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def generated_score_sg_fallback_figures() -> None:
    cumulative = _cumulative_records()
    if not cumulative:
        return

    def values_for(method: str, samples: list[int]) -> list[tuple[int, float]]:
        values = []
        for sample in samples:
            for row in cumulative:
                if row["method"] == method and row["n_samples"] == sample:
                    values.append((sample, float(row["total_seconds"])))
                    break
        return values

    common_samples = [5000, 10000, 20000, 30000, 40000, 50000]
    common_series = [
        (
            _method_label("ScoreSG_extractMatchRef_false"),
            values_for("ScoreSG_extractMatchRef_false", common_samples),
        ),
        (
            _method_label("CoreSG_fitMatchRef_true_extractMatchRef_false"),
            values_for("CoreSG_fitMatchRef_true_extractMatchRef_false", common_samples),
        ),
        (
            _method_label("HDBSCAN_algorithm_best_matchRef_false"),
            values_for("HDBSCAN_algorithm_best_matchRef_false", common_samples),
        ),
        (
            _method_label("HDBSCAN_algorithm_generic_matchRef_false"),
            values_for("HDBSCAN_algorithm_generic_matchRef_false", common_samples),
        ),
    ]
    _draw_pil_line_chart(
        OUT / "score_sg_common_runtime.png",
        "ScoreSG vs exact CoreSG and HDBSCAN",
        common_series,
        log_y=True,
    )

    score_samples = [
        5000,
        10000,
        20000,
        30000,
        40000,
        50000,
        60000,
        70000,
        80000,
        90000,
        100000,
        200000,
    ]
    score_series = [
        (
            _method_label("ScoreSG_extractMatchRef_false"),
            values_for("ScoreSG_extractMatchRef_false", score_samples),
        ),
        (
            _method_label("ScoreSG_extractMatchRef_true"),
            values_for("ScoreSG_extractMatchRef_true", score_samples),
        ),
    ]
    _draw_pil_line_chart(
        OUT / "score_sg_extended_runtime.png",
        "ScoreSG cumulative runtime scaling",
        score_series,
        log_y=False,
    )


def main() -> None:
    copy_existing_assets()
    df = load_results()
    generated_summary(df)
    generated_score_sg_figures(df)
    if (
        not (OUT / "score_sg_common_runtime.png").exists()
        or not (OUT / "score_sg_extended_runtime.png").exists()
    ):
        generated_score_sg_fallback_figures()


if __name__ == "__main__":
    main()
