"""Latih dan evaluasi model klasifikasi ujaran kebencian.

Alur:
1. Baca dataset bersih hasil `preprocess.py`.
2. Split stratified 80/10/10 (train/val/test).
3. Tuning hyperparameter di validation set untuk tiga keluarga model:
   Multinomial Naive Bayes, Regresi Logistik, dan Linear SVM.
4. Cross-validation 5-fold pada train untuk model terbaik.
5. Latih ulang model terpilih pada train+val, evaluasi sekali di test.
6. Simpan metrik, confusion matrix, error analysis, dan model.
"""

from __future__ import annotations

import json
import platform
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "dataset_clean.csv"
SUMMARY_PATH = ROOT / "data" / "processed" / "dataset_summary.json"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

LABELS = ["Normal", "Abusive", "Hate Speech"]
RANDOM_STATE = 42
# Pipeline utama tidak memakai stemming (lihat catatan di preprocess.py).
USE_STEMMING = False

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
PALETTE = ["#2f6f4e", "#c47f2c", "#a63d40"]


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} belum ada. Jalankan `python src/preprocess.py` dahulu."
        )
    frame = pd.read_csv(DATA_PATH)
    frame = frame.dropna(subset=["text_final", "label"])
    frame = frame[frame["text_final"].str.strip().str.len() > 0]
    return frame.reset_index(drop=True)


def make_vectorizer() -> FeatureUnion:
    """TF-IDF word 1-2 gram + char_wb 3-5 gram."""
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=50_000,
        sublinear_tf=True,
        lowercase=False,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=50_000,
        sublinear_tf=True,
        lowercase=False,
    )
    return FeatureUnion([("word", word), ("char", char)], n_jobs=1)


def make_pipeline(model_name: str, params: dict | None = None) -> Pipeline:
    params = params or {}
    if model_name == "nb":
        classifier = MultinomialNB(alpha=params.get("alpha", 1.0))
    elif model_name == "lr":
        classifier = LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=2_000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    elif model_name == "svm":
        classifier = LinearSVC(
            C=params.get("C", 1.0),
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=5_000,
        )
    else:
        raise ValueError(f"Model tidak dikenal: {model_name}")
    return Pipeline([("features", make_vectorizer()), ("clf", classifier)])


def tune_on_validation(
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
) -> tuple[dict, dict]:
    """Latih beberapa konfigurasi pada train, pilih yang terbaik di validation."""
    grids = {
        "nb": [{"alpha": alpha} for alpha in (0.1, 0.5, 1.0)],
        "lr": [{"C": c} for c in (0.5, 1.0, 5.0)],
        "svm": [{"C": c} for c in (0.1, 0.5, 1.0, 2.0)],
    }
    best: dict[str, dict] = {}
    trials: list[dict] = []

    # Vectorizer di-fit sekali pada train agar tuning classifier cepat.
    vectorizer = make_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    for model_name, configs in grids.items():
        for params in configs:
            classifier = make_pipeline(model_name, params).named_steps["clf"]
            classifier.fit(X_train_vec, y_train)
            pred = classifier.predict(X_val_vec)
            score = f1_score(y_val, pred, average="macro")
            trials.append(
                {
                    "model": model_name,
                    "params": params,
                    "val_macro_f1": round(float(score), 6),
                }
            )
            if model_name not in best or score > best[model_name]["score"]:
                best[model_name] = {"params": params, "score": float(score)}

    best_params = {name: info["params"] for name, info in best.items()}
    best_scores = {name: info["score"] for name, info in best.items()}
    return best_params, {"best": best_scores, "trials": trials}


def evaluate_predictions(y_true, y_pred, labels: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 6),
        "per_class": {
            label: {
                "precision": round(float(precision[i]), 6),
                "recall": round(float(recall[i]), 6),
                "f1": round(float(f1[i]), 6),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "classification_report": report,
    }


def get_scores(pipeline: Pipeline, X) -> np.ndarray:
    """Kembalikan skor kelas; probabilitas bila tersedia, softmax bila SVM."""
    classifier = pipeline.named_steps["clf"]
    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(X)
    decision = classifier.decision_function(X)
    decision = np.atleast_2d(decision)
    decision = decision - decision.max(axis=1, keepdims=True)
    exp = np.exp(decision)
    return exp / exp.sum(axis=1, keepdims=True)


def plot_class_distribution(frame: pd.DataFrame) -> None:
    counts = frame["label"].value_counts().reindex(LABELS)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(LABELS, counts.values, color=PALETTE, edgecolor="white")
    ax.set_title("Distribusi kelas setelah praproses")
    ax.set_ylabel("Jumlah tweet")
    ax.set_xlabel("Kelas")
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 80,
            f"{value:,}".replace(",", "."),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, counts.max() * 1.14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "eda_class_distribution.png", dpi=200)
    plt.close(fig)


def plot_text_length(frame: pd.DataFrame) -> None:
    plot_frame = frame.copy()
    plot_frame["word_count"] = plot_frame["text_final"].str.split().map(len)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sns.histplot(
        data=plot_frame,
        x="word_count",
        hue="label",
        hue_order=LABELS,
        palette=PALETTE,
        bins=40,
        element="step",
        common_norm=False,
        ax=ax,
    )
    ax.set_title("Distribusi panjang tweet per kelas")
    ax.set_xlabel("Jumlah kata")
    ax.set_ylabel("Kepadatan")
    ax.set_xlim(0, 60)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "eda_text_length.png", dpi=200)
    plt.close(fig)


def plot_top_words(frame: pd.DataFrame) -> None:
    stopwords = set(
        """
        yang di dan itu ini aku kamu gue lu nya saya tidak ga nggak untuk dengan ada
        jadi dari ke the a an of to in is it on for and or at as by be this that
        juga sudah udah akan bisa kalo kalau tapi tapi apa siapa bagaimana kenapa
        kita kami mereka dia ia anda engkau kau dia orang banyak sangat lebih
        """.split()
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    for ax, label in zip(axes, LABELS):
        tokens: Counter[str] = Counter()
        for text in frame.loc[frame["label"] == label, "text_final"]:
            tokens.update(
                word for word in text.split() if word not in stopwords and len(word) > 2
            )
        top = tokens.most_common(12)
        words = [item[0] for item in top][::-1]
        values = [item[1] for item in top][::-1]
        ax.barh(words, values, color=PALETTE[LABELS.index(label)])
        ax.set_title(label)
        ax.set_xlabel("Frekuensi")
    fig.suptitle("Kata paling sering per kelas (stopword umum dibuang)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "eda_top_words.png", dpi=200)
    plt.close(fig)


def plot_confusion(y_true, y_pred, model_label: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        cbar=False,
        ax=ax,
    )
    ax.set_title(f"Confusion matrix — {model_label} (test)")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def save_top_features(pipeline: Pipeline, model_name: str) -> None:
    features = pipeline.named_steps["features"]
    word_vectorizer: TfidfVectorizer = features.transformer_list[0][1]
    word_names = word_vectorizer.get_feature_names_out()
    n_words = len(word_names)
    classifier = pipeline.named_steps["clf"]
    rows = []

    # Kelas pada classifier tersortir alfabetis, bukan sesuai LABELS.
    classes = [str(label) for label in classifier.classes_]

    if hasattr(classifier, "coef_"):
        coefficients = classifier.coef_
        for label in LABELS:
            class_index = classes.index(label)
            if coefficients.shape[0] == 1:
                weights = coefficients[0]
            else:
                weights = coefficients[class_index]
            # Fitur gabungan = word + char; penjelasan hanya memakai word.
            weights = weights[:n_words]
            top_indices = np.argsort(weights)[::-1][:20]
            for rank, index in enumerate(top_indices, start=1):
                rows.append(
                    {
                        "model": model_name,
                        "class": label,
                        "rank": rank,
                        "feature": word_names[index],
                        "weight": round(float(weights[index]), 6),
                    }
                )
    elif hasattr(classifier, "feature_log_prob_"):
        log_prob = classifier.feature_log_prob_
        baseline = log_prob.mean(axis=0)
        for label in LABELS:
            class_index = classes.index(label)
            weights = (log_prob[class_index] - baseline)[:n_words]
            top_indices = np.argsort(weights)[::-1][:20]
            for rank, index in enumerate(top_indices, start=1):
                rows.append(
                    {
                        "model": model_name,
                        "class": label,
                        "rank": rank,
                        "feature": word_names[index],
                        "weight": round(float(weights[index]), 6),
                    }
                )

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "top_features.csv", index=False)


def build_error_analysis(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    frame_test: pd.DataFrame,
) -> pd.DataFrame:
    pred = pipeline.predict(X_test)
    scores = get_scores(pipeline, pipeline.named_steps["features"].transform(X_test))
    confidence = scores.max(axis=1)
    output = frame_test.copy()
    output["prediksi"] = pred
    output["confidence"] = np.round(confidence, 6)
    output["benar"] = output["prediksi"] == output["label"]
    output.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    errors = output[~output["benar"]].copy()
    errors = errors.sort_values("confidence", ascending=False)
    errors.to_csv(RESULTS_DIR / "error_analysis.csv", index=False)
    return output


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Membaca dataset bersih...")
    frame = load_dataset()
    print(f"Total data: {len(frame):,}")

    X = frame["text_final"]
    y = frame["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )
    train_index = X_train.index
    val_index = X_val.index
    test_index = X_test.index
    frame_test = frame.loc[test_index].copy()

    print(f"Split: train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}")

    print("EDA plots...")
    plot_class_distribution(frame)
    plot_text_length(frame)
    plot_top_words(frame)

    print("Tuning hyperparameter di validation set...")
    best_params, tuning = tune_on_validation(X_train, y_train, X_val, y_val)
    print(json.dumps(tuning["best"], indent=2))

    # Cross-validation 5-fold pada train untuk model terbaik.
    best_family = max(tuning["best"], key=tuning["best"].get)
    print(f"Model terbaik di validation: {best_family} {best_params[best_family]}")
    cv_pipeline = make_pipeline(best_family, best_params[best_family])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        cv_pipeline, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1
    )
    print(f"CV macro-F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Latih ulang tiap keluarga model pada train+val, evaluasi di test.
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    comparison = []
    pipelines: dict[str, Pipeline] = {}
    test_metrics: dict[str, dict] = {}

    for family in ("nb", "lr", "svm"):
        print(f"Melatih ulang {family} pada train+val...")
        pipeline = make_pipeline(family, best_params[family])
        pipeline.fit(X_train_full, y_train_full)
        pred = pipeline.predict(X_test)
        metrics = evaluate_predictions(y_test, pred, LABELS)
        test_metrics[family] = metrics
        pipelines[family] = pipeline
        joblib.dump(pipeline, MODELS_DIR / f"pipeline_{family}.joblib")
        comparison.append(
            {
                "model": family,
                "params": json.dumps(best_params[family]),
                "val_macro_f1": round(float(tuning["best"][family]), 6),
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["macro_f1"],
                "test_weighted_f1": metrics["weighted_f1"],
            }
        )

    # Model final dipilih dari validation, bukan dari test, agar test tetap
    # menjadi evaluasi yang tidak bias.
    best_family = max(tuning["best"], key=tuning["best"].get)
    best_pipeline = pipelines[best_family]
    best_metrics = test_metrics[best_family]
    joblib.dump(best_pipeline, MODELS_DIR / "best_model.joblib")
    (MODELS_DIR / "preprocess_config.json").write_text(
        json.dumps({"use_stemming": USE_STEMMING}, indent=2), encoding="utf-8"
    )
    print(
        f"Model terpilih (validation): {best_family}; "
        f"macro-F1 test={best_metrics['macro_f1']:.4f}"
    )

    comparison_frame = pd.DataFrame(comparison)
    comparison_frame.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    # Evaluasi detail model terbaik.
    pred = best_pipeline.predict(X_test)
    plot_confusion(y_test, pred, best_family.upper())
    report_text = classification_report(
        y_test, pred, labels=LABELS, digits=4, zero_division=0
    )
    (RESULTS_DIR / "classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )
    save_top_features(best_pipeline, best_family)
    predictions = build_error_analysis(best_pipeline, X_test, y_test, frame_test)

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8")) if SUMMARY_PATH.exists() else {}
    metrics_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "dataset": summary,
        "split": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
            "train_full": int(len(X_train_full)),
        },
        "tuning": tuning,
        "cross_validation": {
            "model": best_family,
            "params": best_params[best_family],
            "folds": 5,
            "macro_f1_mean": round(float(cv_scores.mean()), 6),
            "macro_f1_std": round(float(cv_scores.std()), 6),
            "scores": [round(float(score), 6) for score in cv_scores],
        },
        "models": {
            family: {
                "params": best_params[family],
                "val_macro_f1": round(float(tuning["best"][family]), 6),
                "test": test_metrics[family],
            }
            for family in ("nb", "lr", "svm")
        },
        "best_model": best_family,
        "selected_on": "validation_macro_f1",
        "use_stemming": USE_STEMMING,
        "best_metrics": best_metrics,
        "label_order": LABELS,
        "test_size": int(len(y_test)),
        "misclassified": int((predictions["prediksi"] != predictions["label"]).sum()),
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nRingkasan model:")
    print(comparison_frame.to_string(index=False))
    print(f"\nHasil tersimpan di {RESULTS_DIR}")
    print(f"Model tersimpan di {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
