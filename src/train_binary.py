"""Latih model final biner: Normal vs Kebencian/Abusif.

Skema biner dipilih setelah uji generalisasi 2026 menunjukkan model 3 kelas
gagal membedakan hinaan biasa (Abusive) dari serangan kelompok (Hate Speech):
akurasi pada kalimat bergaya 2026 hanya 58,3%. Dengan skema biner, model yang
sama mencapai 87,5% pada uji tersebut.

Semua setelan lain dipertahankan identik dengan `src/train.py` supaya
perbandingan tetap adil:
- sumber data dan pembersihan sama (`data/processed/dataset_clean.csv`),
- split 80/10/10 dengan `random_state=42`,
- tuning hyperparameter pada validation,
- model final dilatih pada train+val lalu diuji sekali pada test,
- fitur TF-IDF word 1-2 gram + char_wb 3-5 gram.

Artefak 3 kelas tetap tersimpan sebagai `metrics_3class.json` dan
`best_model_3class.joblib` untuk keperluan laporan.

Jalankan:

    python src/train_binary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import make_pipeline, tune_on_validation  # noqa: E402

F1_SCORING = "f1_macro"

DATA_PATH = ROOT / "data" / "processed" / "dataset_clean.csv"
SUMMARY_PATH = ROOT / "data" / "processed" / "dataset_summary.json"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
CONFIG_PATH = ROOT / "models" / "preprocess_config.json"
RESULTS_DIR = ROOT / "results"
METRICS_PATH = RESULTS_DIR / "metrics.json"
PROBE_PATH = RESULTS_DIR / "probe_2026.csv"

LABELS = ["Normal", "Kebencian/Abusif"]
RANDOM_STATE = 42
USE_STEMMING = False

PALETTE = ["#2f6f4e", "#a63d40"]


def load_binary_dataset() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH)
    frame = frame.dropna(subset=["text_final", "label"])
    frame = frame[frame["text_final"].str.strip().str.len() > 0]
    frame = frame.reset_index(drop=True)
    frame["label"] = frame["label"].map(
        lambda value: "Normal" if value == "Normal" else "Kebencian/Abusif"
    )
    return frame


def metrics_for(y_true, y_pred) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    per_class = {
        label: {
            "precision": round(float(precision[i]), 6),
            "recall": round(float(recall[i]), 6),
            "f1": round(float(f1[i]), 6),
            "support": int(support[i]),
        }
        for i, label in enumerate(LABELS)
    }
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 6),
        "per_class": per_class,
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
    }


def plot_confusion(y_true, y_pred) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.imshow(matrix, cmap="Blues")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(
                j, i, f"{matrix[i, j]}",
                ha="center", va="center", fontsize=13,
                color="white" if matrix[i, j] > matrix.max() / 2 else "#1a1a1a",
            )
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion matrix — biner (test)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def top_features(pipeline, top_k: int = 25) -> pd.DataFrame:
    classifier = pipeline.named_steps["clf"]
    word_vectorizer = pipeline.named_steps["features"].transformer_list[0][1]
    names = np.array(word_vectorizer.get_feature_names_out())
    coefficients = classifier.coef_
    rows = []
    for index, label in enumerate(classifier.classes_):
        weights = coefficients[0] if coefficients.shape[0] == 1 else coefficients[index]
        word_weights = weights[: len(names)]
        order = np.argsort(word_weights)[::-1][:top_k]
        for rank, position in enumerate(order, start=1):
            rows.append(
                {
                    "model": "svm",
                    "class": str(label),
                    "rank": rank,
                    "feature": names[position],
                    "weight": round(float(word_weights[position]), 6),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    frame = load_binary_dataset()
    print(f"Total data: {len(frame):,}")
    print("Distribusi:", frame["label"].value_counts().to_dict())

    X = frame["text_final"]
    y = frame["label"]

    # Stratifikasi memakai kode numerik, bukan nama kelas. Kalau memakai nama,
    # urutan kelas ikut berubah saat label diganti nama dan split jadi berbeda.
    label_codes = {label: index for index, label in enumerate(LABELS)}
    y_codes = y.map(label_codes)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y_codes, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        stratify=y_temp.map(label_codes), random_state=RANDOM_STATE
    )
    print(f"Split: train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}")
    print(f"  test: {y_test.value_counts().to_dict()}")

    print("Tuning hyperparameter di validation...")
    best_params, tuning = tune_on_validation(X_train, y_train, X_val, y_val)
    print("Skor validation:", json.dumps(tuning["best"], indent=2))
    best_family = max(tuning["best"], key=tuning["best"].get)
    print(f"Keluarga terbaik: {best_family} {best_params[best_family]}")

    cv_pipeline = make_pipeline(best_family, best_params[best_family])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        cv_pipeline, X_train, y_train, cv=cv, scoring=F1_SCORING, n_jobs=1
    )
    print(f"CV macro-F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])

    comparison = []
    pipelines = {}
    for family in ("nb", "lr", "svm"):
        pipeline = make_pipeline(family, best_params[family])
        pipeline.fit(X_full, y_full)
        pred = pipeline.predict(X_test)
        metrics = metrics_for(y_test, pred)
        pipelines[family] = pipeline
        comparison.append(
            {
                "model": family,
                "params": best_params[family],
                "val_macro_f1": round(float(tuning["best"][family]), 6),
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["macro_f1"],
                "test_weighted_f1": metrics["weighted_f1"],
            }
        )
        print(
            f"  {family}: val={tuning['best'][family]:.4f} "
            f"test_acc={metrics['accuracy']:.4f} test_f1={metrics['macro_f1']:.4f}"
        )

    final = pipelines[best_family]
    predictions = final.predict(X_test)
    best_metrics = metrics_for(y_test, predictions)

    print(f"\nModel terpilih: {best_family}")
    print(f"Accuracy={best_metrics['accuracy']:.4f} macro-F1={best_metrics['macro_f1']:.4f}")
    print(classification_report(y_test, predictions, labels=LABELS, digits=4, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=LABELS))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = (
        json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        if SUMMARY_PATH.exists()
        else {}
    )

    payload = {
        "label_scheme": "binary",
        "label_mapping": {
            "Normal": "Normal",
            "Abusive": "Kebencian/Abusif",
            "Hate Speech": "Kebencian/Abusif",
        },
        "dataset": summary,
        "split": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
            "random_state": RANDOM_STATE,
        },
        "tuning": tuning,
        "cross_validation": {
            "mean_macro_f1": round(float(cv_scores.mean()), 6),
            "std_macro_f1": round(float(cv_scores.std()), 6),
            "folds": 5,
        },
        "comparison": comparison,
        "best_model": best_family,
        "selected_on": "validation macro-F1",
        "use_stemming": USE_STEMMING,
        "best_metrics": best_metrics,
        "label_order": LABELS,
        "test_size": int(len(y_test)),
        "misclassified": int((predictions != y_test).sum()),
    }
    METRICS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    joblib.dump(final, MODEL_PATH)
    CONFIG_PATH.write_text(
        json.dumps({"use_stemming": USE_STEMMING, "label_scheme": "binary"}, indent=2),
        encoding="utf-8",
    )

    (RESULTS_DIR / "classification_report.txt").write_text(
        classification_report(y_test, predictions, labels=LABELS, digits=4, zero_division=0),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "text_raw": frame.loc[X_test.index, "text_raw"].values,
            "text_final": X_test.values,
            "label": y_test.values,
            "prediksi": predictions,
            "confidence": np.abs(final.decision_function(X_test)),
            "benar": predictions == y_test.values,
        }
    ).to_csv(RESULTS_DIR / "test_predictions.csv", index=False, encoding="utf-8")

    plot_confusion(y_test, predictions)
    top_features(final).to_csv(RESULTS_DIR / "top_features.csv", index=False)

    # ---- Uji generalisasi 2026 -------------------------------------------
    from src.inference import prepare_text
    from src.preprocess import load_slang_map

    slang = load_slang_map()
    probes = [
        ("Normal", "Program makan bergizi gratis sudah jalan di sekolah anak saya, lumayan membantu"),
        ("Normal", "Efisiensi anggaran perlu, tapi jangan sampai layanan kesehatan ikut kena"),
        ("Normal", "Judi online memang harus diberantas, banyak yang jadi korban"),
        ("Normal", "Harga cabai naik lagi padahal baru awal tahun"),
        ("Normal", "Aku baru coba update aplikasinya, tampilannya lebih rapi sekarang"),
        ("Normal", "Timnas main malam ini, siapa yang mau nonton bareng"),
        ("Normal", "Banyak pinjol ilegal sekarang, hati-hati sebelum pinjam"),
        ("Normal", "Cuaca panas banget, jangan lupa minum air"),
        ("Kebencian/Abusif", "Dasar tolol, baca beritanya sampai selesai dulu baru komentar"),
        ("Kebencian/Abusif", "Kamu goblok banget sih, sudah dijelaskan masih nanya"),
        ("Kebencian/Abusif", "Bangsat, pesananku belum sampai juga"),
        ("Kebencian/Abusif", "Dia itu bego, nggak ngerti situasi sama sekali"),
        ("Kebencian/Abusif", "Muka lo jelek banget, kelakuan lo lebih jelek"),
        ("Kebencian/Abusif", "Kampret, janji doang tapi nggak pernah ditepatin"),
        ("Kebencian/Abusif", "Otak lo di mana, kok bisa mikir kayak gitu"),
        ("Kebencian/Abusif", "Reseller sampah, nipu terus"),
        ("Kebencian/Abusif", "Dasar cebong, semua yang pro pemerintah pasti korup"),
        ("Kebencian/Abusif", "Orang kristen memang nggak pantas jadi pemimpin di negara ini"),
        ("Kebencian/Abusif", "Cina asing, pulang saja ke negaranya"),
        ("Kebencian/Abusif", "Perempuan nggak cocok jadi pemimpin, terlalu emosional"),
        ("Kebencian/Abusif", "Dasar kampret, pendukung itu nggak layak hidup di sini"),
        ("Kebencian/Abusif", "Penyandang disabilitas nggak usah ikut pemilu, bikin repot"),
        ("Kebencian/Abusif", "Dasar kafir, pantas saja negaranya hancur"),
        ("Kebencian/Abusif", "Anak Papua memang biadab, susah diatur"),
    ]
    prepared = [prepare_text(text, slang, USE_STEMMING) for _, text in probes]
    probe_pred = final.predict(prepared)
    gold = [label for label, _ in probes]
    probe_accuracy = float(accuracy_score(gold, probe_pred))
    print(f"\nProbe 2026: {int((probe_pred == np.array(gold)).sum())}/{len(gold)} "
          f"= {probe_accuracy * 100:.1f}%")

    pd.DataFrame(
        {
            "teks": [text for _, text in probes],
            "aktual": gold,
            "prediksi": probe_pred,
            "benar": np.array(probe_pred) == np.array(gold),
        }
    ).to_csv(PROBE_PATH, index=False, encoding="utf-8")

    payload["probe_2026"] = {
        "n": len(gold),
        "accuracy": round(probe_accuracy, 4),
        "correct": int((probe_pred == np.array(gold)).sum()),
        "note": "Probe manual bergaya 2026, bukan test set resmi.",
    }
    METRICS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nModel   : {MODEL_PATH}")
    print(f"Metrik  : {METRICS_PATH}")
    print(f"Probe   : {PROBE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
