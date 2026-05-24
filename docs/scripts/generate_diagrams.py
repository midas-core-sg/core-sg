from __future__ import annotations

from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
except Exception:  # pragma: no cover - fallback for broken local scientific envs
    plt = None
    Circle = FancyArrowPatch = Rectangle = None


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "source" / "_static" / "images" / "theory"


def _box(ax, xy, text, width=2.4, height=0.75, color="#d7eef8"):
    x, y = xy
    rect = Rectangle(
        (x, y), width, height, facecolor=color, edgecolor="#24546a", linewidth=1.5
    )
    ax.add_patch(rect)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9)
    return rect


def _arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="->", mutation_scale=12, linewidth=1.4, color="#334"
        )
    )


def pipeline_overview():
    labels = [
        "X",
        "pairwise\ndistances",
        "kNN\ngraph",
        "Core-SG\nsupport",
        "reweight(k)",
        "MST(k)",
        "hierarchy",
        "labels",
    ]
    fig, ax = plt.subplots(figsize=(13, 2.4))
    ax.axis("off")
    for i, label in enumerate(labels):
        _box(
            ax,
            (i * 1.65, 0.65),
            label,
            width=1.35,
            height=0.7,
            color="#e7f2d9" if i in {0, 7} else "#d7eef8",
        )
        if i:
            _arrow(ax, ((i - 1) * 1.65 + 1.35, 1.0), (i * 1.65, 1.0))
    ax.set_xlim(-0.2, 13.0)
    ax.set_ylim(0, 2)
    fig.savefig(OUT / "pipeline_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def multi_k_reuse():
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axis("off")
    _box(ax, (0.3, 1.45), "fit k_max=30", width=2.2, color="#e7f2d9")
    for i, k in enumerate([25, 20, 15]):
        y = 2.45 - i
        _box(ax, (4.1, y), f"extract k={k}", width=2.2, color="#d7eef8")
        _arrow(ax, (2.5, 1.82), (4.1, y + 0.38))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 3.5)
    fig.savefig(OUT / "multi_k_reuse.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def clusterer_reuse():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axis("off")
    _box(ax, (0.2, 3.3), "CoreSGClusterer(k_max=30)", width=3.1, color="#e7f2d9")
    rows = [
        ("fit(X, k=25)", "create core_sg_\nCoreSG.fit(...)\nextract k=25"),
        ("fit(X, k=20)", "reuse core_sg_\nextract k=20"),
        ("fit(X, k=15)", "reuse core_sg_\nextract k=15"),
    ]
    for i, (left, right) in enumerate(rows):
        y = 2.4 - i * 1.15
        _box(ax, (0.3, y), left, width=2.0, color="#d7eef8")
        _box(ax, (4.0, y - 0.1), right, width=3.1, height=0.95, color="#f8e6c8")
        _arrow(ax, (2.3, y + 0.38), (4.0, y + 0.38))
    ax.set_xlim(0, 7.8)
    ax.set_ylim(0, 4.3)
    fig.savefig(OUT / "clusterer_reuse.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def support_graph_vs_mst():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    positions = [(0.5, 1.8), (1.6, 2.4), (2.6, 1.7), (1.9, 0.7), (0.7, 0.5)]
    support_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 3), (1, 3), (1, 4)]
    mst_edges = [(4, 0), (0, 1), (1, 2), (2, 3)]
    for ax, title, edges in zip(
        axes, ["Support graph", "Extracted MST"], [support_edges, mst_edges]
    ):
        ax.axis("off")
        ax.set_title(title)
        for u, v in edges:
            ax.plot(
                [positions[u][0], positions[v][0]],
                [positions[u][1], positions[v][1]],
                color="#59717c",
                linewidth=1.6,
            )
        for x, y in positions:
            ax.add_patch(
                Circle(
                    (x, y),
                    0.13,
                    facecolor="#e7f2d9",
                    edgecolor="#24546a",
                    linewidth=1.2,
                )
            )
        ax.set_xlim(0, 3.1)
        ax.set_ylim(0.1, 2.8)
    fig.savefig(OUT / "support_graph_vs_mst.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def hdbscan_adapter_boundary():
    labels = [
        "CoreSG",
        "hdbscan_adapter",
        "filtered\n_tree_to_labels\nkwargs",
        "HDBSCAN\nprivate APIs",
    ]
    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.axis("off")
    for i, label in enumerate(labels):
        _box(
            ax,
            (i * 2.2, 0.8),
            label,
            width=1.8,
            color="#f8e6c8" if i == 1 else "#d7eef8",
        )
        if i:
            _arrow(ax, ((i - 1) * 2.2 + 1.8, 1.18), (i * 2.2, 1.18))
    ax.set_xlim(-0.2, 8.8)
    ax.set_ylim(0.2, 2.2)
    fig.savefig(OUT / "hdbscan_adapter_boundary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def score_sg_pipeline():
    labels = [
        "X",
        "PyNNDescent",
        "approximate\nkNN graph",
        "anti-hub\nselection",
        "support\ngraph",
        "MST /\nhierarchy",
    ]
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.axis("off")
    for i, label in enumerate(labels):
        _box(
            ax,
            (i * 1.75, 0.85),
            label,
            width=1.45,
            color="#e7f2d9" if i == 0 else "#d7eef8",
        )
        if i:
            _arrow(ax, ((i - 1) * 1.75 + 1.45, 1.2), (i * 1.75, 1.2))
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(0.2, 2.3)
    fig.savefig(OUT / "score_sg_pipeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fallback_png(name: str, lines: list[str]) -> None:
    from PIL import Image, ImageDraw

    width = 1200
    height = 320
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, width - 20, height - 20), outline="#24546a", width=3)
    y = 70
    for line in lines:
        draw.text((50, y), line, fill="#20333d")
        y += 38
    image.save(OUT / name)


def fallback_all() -> None:
    diagrams = {
        "pipeline_overview.png": [
            "X -> pairwise distances -> kNN graph -> Core-SG support",
            "-> reweight(k) -> MST(k) -> hierarchy -> labels",
        ],
        "multi_k_reuse.png": [
            "fit k_max=30",
            "-> extract k=25",
            "-> extract k=20",
            "-> extract k=15",
        ],
        "clusterer_reuse.png": [
            "CoreSGClusterer(k_max=30)",
            "fit(X, k=25): create core_sg_, fit, extract",
            "fit(X, k=20): reuse core_sg_, extract",
            "fit(X, k=15): reuse core_sg_, extract",
        ],
        "support_graph_vs_mst.png": [
            "Support graph: many reusable support edges",
            "Extracted MST: n - 1 selected edges for a target k",
        ],
        "hdbscan_adapter_boundary.png": [
            "CoreSG -> hdbscan_adapter",
            "-> filtered _tree_to_labels kwargs",
            "-> HDBSCAN private APIs",
        ],
        "score_sg_pipeline.png": [
            "X -> PyNNDescent -> approximate kNN graph",
            "-> anti-hub selection -> support graph -> MST / hierarchy",
        ],
    }
    for name, lines in diagrams.items():
        fallback_png(name, lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if plt is None:
        fallback_all()
        return
    pipeline_overview()
    multi_k_reuse()
    clusterer_reuse()
    support_graph_vs_mst()
    hdbscan_adapter_boundary()
    score_sg_pipeline()


if __name__ == "__main__":
    main()
