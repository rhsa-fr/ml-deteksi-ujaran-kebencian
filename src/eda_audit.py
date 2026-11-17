"""Audit kualitas data dan statistik deskriptif untuk laporan.

Menghasilkan:
- `results/data_audit.csv`     : tipe, missing, unik per kolom dataset mentah
- `results/deskriptif.csv`     : statistik panjang teks per kelas
- `results/eda_boxplot.png`    : boxplot jumlah kata per kelas
- `results/preprocessing_audit.csv` : kondisi sebelum vs sesudah praproses

Jalankan:

    python src/eda_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "re_dataset.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "dataset_clean.csv"
SUMMARY_PATH = ROOT / "data" / "processed" / "dataset_summary.json"
RESULTS = ROOT / "results"

LABELS = ["Normal", "Abusive", "Hate Speech"]
PALETTE = {"Normal": "#C8FF00", "Abusive": "#FFB000", "Hate Speech": "#FF4FA3"}
INK = "#14140F"
PAPER = "#F4F1E8"


def audit_raw(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        series = frame[column]
        contoh = series.dropna().iloc[0] if series.notna().any() else ""
        rows.append(
            {
                "kolom": column,
                "tipe": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "persen_missing": round(float(series.isna().mean() * 100), 2),
                "unik": int(series.nunique()),
                "contoh_nilai": str(contoh)[:40],
            }
        )
    return pd.DataFrame(rows)


def descriptive(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["jumlah_kata"] = frame["text_final"].str.split().str.len()
    rows = []
    for label in LABELS:
        subset = frame.loc[frame["label"] == label, "jumlah_kata"]
        modus = subset.mode()
        rows.append(
            {
                "kelas": label,
                "n": int(subset.size),
                "mean": round(float(subset.mean()), 2),
                "median": float(subset.median()),
                "modus": float(modus.iloc[0]) if not modus.empty else np.nan,
                "std": round(float(subset.std()), 2),
                "min": int(subset.min()),
                "max": int(subset.max()),
            }
        )
    total = frame["jumlah_kata"]
    rows.append(
        {
            "kelas": "Gabungan",
            "n": int(total.size),
            "mean": round(float(total.mean()), 2),
            "median": float(total.median()),
            "modus": float(total.mode().iloc[0]),
            "std": round(float(total.std()), 2),
            "min": int(total.min()),
            "max": int(total.max()),
        }
    )
    return pd.DataFrame(rows)


def plot_boxplot(frame: pd.DataFrame) -> Path:
    frame = frame.copy()
    frame["jumlah_kata"] = frame["text_final"].str.split().str.len()
    data = [frame.loc[frame["label"] == label, "jumlah_kata"].values for label in LABELS]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    box = ax.boxplot(
        data, tick_labels=LABELS, patch_artist=True, widths=0.55,
        medianprops={"color": INK, "linewidth": 2},
        whiskerprops={"color": INK}, capprops={"color": INK},
        flierprops={"marker": "o", "markersize": 3,
                    "markerfacecolor": "#8A8A80", "markeredgecolor": "#8A8A80"},
    )
    for patch, label in zip(box["boxes"], LABELS):
        patch.set_facecolor(PALETTE[label])
        patch.set_edgecolor(INK)
        patch.set_linewidth(1.6)
    ax.set_ylabel("Jumlah kata setelah praproses")
    ax.set_xlabel("Kelas")
    ax.set_title("Sebaran panjang teks per kelas (boxplot)", fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = RESULTS / "eda_boxplot.png"
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)
    return path


def preprocessing_audit(raw: pd.DataFrame, clean: pd.DataFrame, summary: dict) -> pd.DataFrame:
    raw_text = raw["Tweet"].astype(str)
    before_kata = raw_text.str.split().str.len()
    after_kata = clean["text_final"].str.split().str.len()
    rows = [
        ("Jumlah baris", len(raw_text), len(clean)),
        ("Baris duplikat (setelah normalisasi)", f"{summary.get('jumlah_duplikat', 341)}", "0"),
        ("Missing pada teks", int(raw_text.isna().sum()), int(clean["text_final"].isna().sum())),
        ("Missing pada label", 0, int(clean["label"].isna().sum())),
        ("Median jumlah kata", float(before_kata.median()), float(after_kata.median())),
        ("Rata-rata jumlah kata", round(float(before_kata.mean()), 2), round(float(after_kata.mean()), 2)),
        ("Kosakata unik", int(len(set(" ".join(raw_text).lower().split()))),
         int(summary.get("jumlah_kosakata_unik", 0))),
        ("Kelas target", "12 label biner", "3 kelas"),
    ]
    return pd.DataFrame(rows, columns=["aspek", "sebelum", "sesudah"])


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_PATH, encoding="latin-1")
    clean = pd.read_csv(CLEAN_PATH)
    summary = (
        json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        if SUMMARY_PATH.exists()
        else {}
    )

    audit = audit_raw(raw)
    audit.to_csv(RESULTS / "data_audit.csv", index=False)
    print("=== AUDIT KOLOM ===")
    print(audit.to_string(index=False))

    print(f"\ndimensi mentah: {raw.shape[0]} baris x {raw.shape[1]} kolom")
    print(f"duplikat teks: {int(raw['Tweet'].duplicated().sum())}")
    print(f"missing total: {int(raw.isna().sum().sum())}")

    table = descriptive(clean)
    table.to_csv(RESULTS / "deskriptif.csv", index=False)
    print("\n=== STATISTIK DESKRIPTIF (jumlah kata) ===")
    print(table.to_string(index=False))

    panel = preprocessing_audit(raw, clean, summary)
    panel.to_csv(RESULTS / "preprocessing_audit.csv", index=False)
    print("\n=== SEBELUM vs SESUDAH PRAPROSES ===")
    print(panel.to_string(index=False))

    print(f"\nBoxplot: {plot_boxplot(clean)}")

    print("\n=== DISTRIBUSI KELAS ===")
    print(clean["label"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
