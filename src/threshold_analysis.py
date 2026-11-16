"""Analisis ambang (threshold) untuk indikasi Hate Speech.

Model SVM menghasilkan decision score, lalu softmax dipakai sebagai skor
relatif. Skrip ini menyapu ambang skor Hate Speech pada test set dan
melaporkan precision/recall/F1 untuk keputusan biner "indikasi hate".
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "dataset_clean.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
RESULTS_DIR = ROOT / "results"
LABELS = ["Normal", "Abusive", "Hate Speech"]
RANDOM_STATE = 42


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.atleast_2d(values)
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def main() -> int:
    frame = pd.read_csv(DATA_PATH)
    X = frame["text_final"]
    y = frame["label"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )

    model = joblib.load(MODEL_PATH)
    features = model.named_steps["features"].transform(X_test)
    classifier = model.named_steps["clf"]
    classes = [str(label) for label in classifier.classes_]
    scores = softmax(classifier.decision_function(features))
    hate_index = classes.index("Hate Speech")
    hate_scores = scores[:, hate_index]
    y_true = (y_test == "Hate Speech").astype(int).to_numpy()

    rows = []
    for threshold in np.round(np.arange(0.20, 0.91, 0.05), 2):
        y_pred = (hate_scores >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        rows.append(
            {
                "threshold": float(threshold),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "flagged": int(y_pred.sum()),
                "support_hate": int(y_true.sum()),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "hate_threshold_analysis.csv", index=False)

    best = table.loc[table["f1"].idxmax()]
    argmax_pred = (np.argmax(scores, axis=1) == hate_index).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, argmax_pred, average="binary", zero_division=0
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(table["threshold"], table["precision"], marker="o", label="Precision")
    ax.plot(table["threshold"], table["recall"], marker="s", label="Recall")
    ax.plot(table["threshold"], table["f1"], marker="^", label="F1")
    ax.axvline(best["threshold"], color="#a63d40", linestyle="--", label=f"F1 terbaik ({best['threshold']:.2f})")
    ax.set_xlabel("Ambang skor Hate Speech")
    ax.set_ylabel("Skor")
    ax.set_ylim(0, 1.02)
    ax.set_title("Kurva ambang indikasi Hate Speech (test 2019)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "hate_threshold_curve.png", dpi=200)
    plt.close(fig)

    payload = {
        "catatan": "Skor SVM bukan probabilitas terkalibrasi; ambang bersifat relatif pada dataset ini.",
        "support_hate": int(y_true.sum()),
        "support_non_hate": int(len(y_true) - y_true.sum()),
        "argmax_baseline": {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
        },
        "best_f1_threshold": {
            "threshold": float(best["threshold"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
        },
        "tabel": rows,
    }
    (RESULTS_DIR / "hate_threshold.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"Support hate: {int(y_true.sum())} / {len(y_true)} "
        f"({y_true.mean() * 100:.1f}%)"
    )
    print(table.to_string(index=False))
    print(
        f"\nArgmax baseline: P={precision:.4f} R={recall:.4f} F1={f1:.4f}"
    )
    print(
        f"Ambang F1 terbaik: {best['threshold']:.2f} "
        f"(P={best['precision']:.4f} R={best['recall']:.4f} F1={best['f1']:.4f})"
    )
    print(f"Tabel: {RESULTS_DIR / 'hate_threshold_analysis.csv'}")
    print(f"Grafik: {RESULTS_DIR / 'hate_threshold_curve.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
