"""Bangun dua flowchart wajib:

1. `results/pipeline_umum.png` — pipeline proyek dalam 12 tahap, dari
   identifikasi masalah sampai kesimpulan, lengkap dengan keputusan.
2. `results/algoritma_linear_svm.png` — mekanisme internal metode yang dipilih:
   teks -> praproses -> TF-IDF -> skor linear -> argmax -> label.

Gaya visual mengikuti tema brutalist laporan agar konsisten.

Jalankan:

    python src/build_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

PAPER = "#F4F1E8"
INK = "#14140F"
LIME = "#C8FF00"
ORANGE = "#FFB000"
PINK = "#FF4FA3"
BLUE = "#2F5BFF"
WHITE = "#FFFDF5"
MUTED = "#5B5B52"


def box(ax, x, y, w, h, title, detail="", face=WHITE, edge=INK, title_size=12,
        detail_size=9, title_color=INK, detail_color=INK):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="square,pad=0",
        linewidth=2.4, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.22, y + h - 0.26, title,
        fontsize=title_size, fontweight="bold", color=title_color,
        family="DejaVu Sans", va="top", zorder=3,
    )
    if detail:
        ax.text(
            x + 0.22, y + h - 0.62, detail,
            fontsize=detail_size, color=detail_color,
            family="DejaVu Sans Mono", va="top", linespacing=1.35, zorder=3,
        )


def arrow(ax, x1, y1, x2, y2, color=INK, style="-|>", rad=0.0, width=2.0):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, mutation_scale=18,
            linewidth=width, color=color, zorder=4,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0, shrinkB=0,
        )
    )


def build_pipeline() -> Path:
    stages = [
        ("01  PROBLEM IDENTIFICATION",
         "Moderasi manual tidak skalabel\nJenis masalah: klasifikasi teks 3 kelas", LIME),
        ("02  DATA ACQUISITION",
         "GitHub Ibrohim & Budi (2019)\n13.169 tweet · 12 kolom label", WHITE),
        ("03  DATA UNDERSTANDING & QUALITY",
         "Cek tipe, missing, duplikat, rentang\n341 duplikat · 0 missing value", WHITE),
        ("04  EXPLORATORY DATA ANALYSIS",
         "Statistik, boxplot, distribusi kelas\nNormal 44,4% · Abusive 13,3% · Hate 42,2%", WHITE),
        ("05  DATA PREPROCESSING",
         "Case folding · slang · netralisasi tokoh\ndedup: 13.169 -> 12.822", WHITE),
        ("06  TRANSFORMATION & FEATURE ENGINEERING",
         "TF-IDF word 1-2 gram + char_wb 3-5\n15.315 + 34.593 = 49.908 fitur", ORANGE),
        ("07  DATA PREPARATION",
         "X = text_final, y = label\nSplit 80/10/10 stratified", WHITE),
        ("08  AI METHOD DESIGN",
         "NB · LR · Linear SVM\nDipilih dari macro-F1 validation", BLUE),
        ("09  MODEL IMPLEMENTATION",
         "scikit-learn Pipeline\nFeatureUnion + LinearSVC(C=0,1)", WHITE),
        ("10  TESTING & EXPERIMENTATION",
         "Grid hyperparameter di validation\n5-fold CV · test dipakai sekali", WHITE),
        ("11  EVALUATION & ERROR ANALYSIS",
         "macro-F1 0,8893 · akurasi 89,01%\nUji 2026: 83,3%", WHITE),
        ("12  FINAL OUTPUT & CONCLUSION",
         "Model .joblib · API FastAPI\ndemo web · laporan · slide", PINK),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 18.4), dpi=170)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 18.4)
    ax.axis("off")

    ax.text(4.5, 18.25, "PIPELINE PENGEMBANGAN AI — 12 TAHAP",
            ha="center", va="top", fontsize=15, fontweight="bold", color=INK)

    top = 16.65
    height = 1.3
    gap = 0.15
    width = 7.6
    x = 0.7
    y = top
    positions = []
    for title, detail, face in stages:
        box(ax, x, y, width, height, title, detail, face=face)
        positions.append((y, y + height))
        y -= height + gap

    for i in range(len(positions) - 1):
        y_from = positions[i][0]
        y_to = positions[i + 1][1]
        arrow(ax, x + width / 2, y_from, x + width / 2, y_to)

    # Loop umpan balik: evaluasi -> desain metode
    eval_top = positions[10][1]
    design_bottom = positions[7][0]
    lane = x + width + 0.55
    arrow(ax, x + width, eval_top - 0.45, lane, eval_top - 0.45,
          color=ORANGE, style="-", width=1.8)
    arrow(ax, lane, eval_top - 0.45, lane, design_bottom + 0.45,
          color=ORANGE, style="-", width=1.8)
    arrow(ax, lane, design_bottom + 0.45, x + width,
          design_bottom + 0.45, color=ORANGE)
    ax.text(lane + 0.12, (eval_top + design_bottom) / 2, "ulangi\njika\nbelum\nlayak",
            fontsize=8, color=ORANGE, family="DejaVu Sans Mono", va="center")

    ax.text(4.5, 0.08, "setiap tahap: konsep -> implementasi -> hasil -> analisis",
            ha="center", va="bottom", fontsize=9, color=MUTED, family="DejaVu Sans Mono")

    path = RESULTS / "pipeline_umum.png"
    fig.savefig(path, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return path


def build_algorithm() -> Path:
    fig, ax = plt.subplots(figsize=(10.2, 12.4), dpi=170)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 12.4)
    ax.axis("off")

    ax.text(5.1, 12.15, "MEKANISME INTERNAL: TF-IDF + LINEAR SVM",
            ha="center", va="top", fontsize=15, fontweight="bold", color=INK)
    ax.text(5.1, 11.75, "bagaimana satu teks berubah menjadi satu label",
            ha="center", va="top", fontsize=9.5, color=MUTED,
            family="DejaVu Sans Mono")

    left, width = 0.7, 8.8

    box(ax, left, 10.3, width, 1.0, "1. TEKS MASUK",
        '"Kamu goblok banget sih"   ->   1 kalimat, 4 token', face=WHITE)
    arrow(ax, 5.1, 10.3, 5.1, 9.95)

    box(ax, left, 8.95, width, 1.0, "2. PRAPROSES",
        "case folding · normalisasi slang · hapus mention/URL · netralisasi tokoh", face=WHITE)
    arrow(ax, 5.1, 8.95, 5.1, 8.6)

    box(ax, left, 7.6, width, 1.0, "3. TOKEN",
        "kamu | goblok | banget | sih", face=WHITE)

    # Cabang ke dua vektor TF-IDF
    arrow(ax, 3.0, 7.6, 3.0, 7.05)
    arrow(ax, 7.2, 7.6, 7.2, 7.05)
    box(ax, left, 5.7, 4.2, 1.35, "4a. TF-IDF WORD 1-2 GRAM",
        "tf(t,d) x idf(t)\nidf = ln(N/df) + 1\n15.315 fitur", face=LIME, title_size=11)
    box(ax, 5.3, 5.7, 4.2, 1.35, "4b. TF-IDF CHAR_WB 3-5",
        '"gob" "obl" "blo" "lok"\ntahan typo & ejaan\n34.593 fitur', face=LIME, title_size=11)

    # Gabung
    arrow(ax, 3.0, 5.7, 3.0, 5.5)
    arrow(ax, 7.2, 5.7, 7.2, 5.5)
    arrow(ax, 3.0, 5.5, 7.2, 5.5, style="-", width=1.8)
    arrow(ax, 5.1, 5.5, 5.1, 5.3)
    box(ax, left, 4.35, width, 0.95, "4c. GABUNG (FeatureUnion)",
        "vektor x berdimensi 49.908, sangat sparse", face=ORANGE)

    arrow(ax, 5.1, 4.35, 5.1, 4.0)
    box(ax, left, 3.0, width, 1.0, "5. SKOR LINEAR PER KELAS",
        "score_c = w_c . x + b_c      kelas = argmax_c score_c", face=WHITE)

    arrow(ax, 5.1, 3.0, 5.1, 2.65)
    box(ax, left, 1.55, width, 1.1, "6. KEPUTUSAN MARGIN MAKSIMUM",
        "SVM memilih hyperplane dengan margin terlebar; C = 0,1\nmengatur toleransi terhadap kesalahan klasifikasi.",
        face=BLUE)

    arrow(ax, 5.1, 1.55, 5.1, 1.4)
    box(ax, left, 0.1, 4.15, 1.3, "7. OUTPUT",
        "label + confidence\nNormal 0,111\nKebencian/Abusif 0,889", face=PINK)
    box(ax, 5.4, 0.1, 4.1, 1.3, "8. PENJELASAN TOKEN",
        "kontribusi = x_i * w_i\nkata pendorong vs penahan", face=WHITE)

    path = RESULTS / "algoritma_linear_svm.png"
    fig.savefig(path, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return path


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for path in (build_pipeline(), build_algorithm()):
        print(f"Diagram tersimpan: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
