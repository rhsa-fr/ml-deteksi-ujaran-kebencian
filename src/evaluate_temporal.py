"""Evaluasi model pada sampel teks baru (uji validitas temporal 2026).

Isi `data/temporal/temporal_2026.csv` terlebih dahulu. Skrip ini tidak
memalsukan data: jika CSV masih kosong, ia berhenti dengan pesan jelas.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import prepare_text
from src.preprocess import load_slang_map
MODEL_PATH = ROOT / "models" / "best_model.joblib"
CONFIG_PATH = ROOT / "models" / "preprocess_config.json"
DATA_PATH = ROOT / "data" / "temporal" / "temporal_2026.csv"
RESULTS_DIR = ROOT / "results"
LABELS = ["Normal", "Abusive", "Hate Speech"]


def main() -> int:
    if not MODEL_PATH.exists():
        print("Model belum ada. Jalankan `python src/train.py` dahulu.")
        return 1
    if not DATA_PATH.exists():
        print(f"File tidak ditemukan: {DATA_PATH}")
        return 1

    frame = pd.read_csv(DATA_PATH)
    if frame.empty:
        print(
            "CSV uji temporal masih kosong. Isi minimal 100 baris sesuai "
            "panduan di data/temporal/README.md."
        )
        return 0

    required = {"text", "label"}
    missing = required.difference(frame.columns)
    if missing:
        print(f"Kolom wajib tidak ada: {sorted(missing)}")
        return 1

    model = joblib.load(MODEL_PATH)
    config = (
        json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if CONFIG_PATH.exists()
        else {"use_stemming": False}
    )
    slang_map = load_slang_map()
    use_stemming = bool(config.get("use_stemming", False))

    frame = frame.dropna(subset=["text", "label"]).copy()
    frame["prediksi"] = [
        model.predict(
            model.named_steps["features"].transform(
                [prepare_text(str(text), slang_map, use_stemming)]
            )
        )[0]
        for text in frame["text"]
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_DIR / "temporal_predictions.csv", index=False)

    payload = {
        "jumlah_sampel": int(len(frame)),
        "akurasi": round(float(accuracy_score(frame["label"], frame["prediksi"])), 6),
        "macro_f1": round(
            float(f1_score(frame["label"], frame["prediksi"], average="macro")), 6
        ),
        "classification_report": classification_report(
            frame["label"],
            frame["prediksi"],
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }

    if {"annotator_1", "annotator_2"}.issubset(frame.columns):
        pairs = frame.dropna(subset=["annotator_1", "annotator_2"])
        if len(pairs) > 1:
            from sklearn.metrics import cohen_kappa_score

            payload["cohen_kappa_annotator"] = round(
                float(cohen_kappa_score(pairs["annotator_1"], pairs["annotator_2"])),
                6,
            )

    (RESULTS_DIR / "temporal_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
