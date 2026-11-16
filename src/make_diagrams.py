"""Buat diagram pipeline untuk laporan (gaya brutalist, konsisten dengan UI)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

PAPER = "#f2efe4"
INK = "#0a0a0a"
LIME = "#c8ff00"
ORANGE = "#ffb000"
BLUE = "#2f5bff"
PINK = "#ff4fa3"
WHITE = "#fffdf5"

STEPS = [
    ("01  DATASET", "13.169 tweet · 12 label\nIbrohim & Budi (2019)", LIME),
    ("02  PRAPROSES", "case folding · URL/mention\nslang 2026 · dedup", WHITE),
    ("03  SPLIT", "train 10.262 · val 1.283\ntest 1.283 (stratified)", ORANGE),
    ("04  FITUR", "TF-IDF word 1–2 gram\n+ char_wb 3–5 gram", WHITE),
    ("05  MODEL", "Multinomial NB · LogReg\nLinear SVM", BLUE),
    ("06  EVALUASI", "macro-F1 · confusion matrix\n5-fold CV · error analysis", WHITE),
    ("07  DEMO", "FastAPI + HTML\nlabel · skor · token", PINK),
]


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 12.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16.4)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    box_height = 1.55
    gap = 0.58
    y = 15.5 - box_height
    centers = []

    for index, (title, detail, color) in enumerate(STEPS):
        ax.add_patch(
            Rectangle(
                (1.0, y),
                8.0,
                box_height,
                facecolor=color,
                edgecolor=INK,
                linewidth=4,
                zorder=2,
            )
        )
        ax.add_patch(
            Rectangle(
                (1.22, y - 0.14),
                7.56,
                box_height,
                facecolor="none",
                edgecolor=INK,
                linewidth=1.2,
                linestyle=(0, (3, 3)),
                zorder=3,
            )
        )
        text_color = WHITE if color in (BLUE,) else INK
        ax.text(
            1.35,
            y + box_height - 0.42,
            title,
            fontsize=14,
            fontweight="black",
            family="DejaVu Sans",
            color=text_color,
            va="center",
            zorder=4,
        )
        ax.text(
            1.35,
            y + 0.42,
            detail,
            fontsize=9.2,
            family="DejaVu Sans Mono",
            color=text_color,
            va="center",
            linespacing=1.5,
            zorder=4,
        )
        centers.append(y)
        y -= box_height + gap

    for upper, lower in zip(centers, centers[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (5.0, upper - 0.02),
                (5.0, lower + box_height + 0.02),
                arrowstyle="-|>",
                mutation_scale=20,
                linewidth=3.5,
                color=INK,
                zorder=5,
            )
        )

    ax.text(
        5.0,
        16.05,
        "PIPELINE PENELITIAN",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="black",
        family="DejaVu Sans",
        color=INK,
    )
    ax.text(
        5.0,
        0.55,
        "setiap tahap: konsep → implementasi → hasil → analisis",
        ha="center",
        va="center",
        fontsize=9.5,
        family="DejaVu Sans Mono",
        color=INK,
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(RESULTS_DIR / "pipeline_diagram.png", dpi=200, facecolor=PAPER)
    plt.close(fig)
    print(f"Diagram tersimpan di {RESULTS_DIR / 'pipeline_diagram.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
