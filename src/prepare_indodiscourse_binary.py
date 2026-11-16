"""Siapkan IndoDiscourse sebagai dataset biner Toxic / Non-Toxic.

Label mengikuti label resmi dataset (`toxicity`), bukan turunan 3 kelas.
Kelas negatif di-*downsample* menjadi 3x kelas positif, meniru setelan
"balanced" yang dipakai paper IndoDiscourse:

    2.156 Toxic + 6.468 Non-Toxic = 8.624 teks

Downsampling dilakukan SEBELUM split, sehingga train/validation/test punya
distribusi yang sama. Tanpa ini, test akan 92% Non-Toxic dan accuracy jadi
metrik yang menyesatkan.

Jalankan:

    python src/prepare_indodiscourse_binary.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import basic_clean, load_slang_map, normalize_slang  # noqa: E402
from src.prepare_indodiscourse import download_if_needed, majority  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"
OUT_CSV = PROCESSED_DIR / "indodiscourse_binary_clean.csv"
OUT_SUMMARY = PROCESSED_DIR / "indodiscourse_binary_summary.json"

LABEL_TOXIC = "Toxic"
LABEL_NON_TOXIC = "Non-Toxic"
NEGATIVE_RATIO = 3
SEED = 42


def map_binary(row: dict) -> str | None:
    if majority(row.get("is_noise_or_spam_text")) == 1:
        return None
    toxic = majority(row.get("toxicity"))
    if toxic == 1:
        return LABEL_TOXIC
    if toxic == 0:
        return LABEL_NON_TOXIC
    return None


def main() -> None:
    path = download_if_needed()
    slang = load_slang_map()

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Baris mentah: {len(rows):,}")

    seen: set[str] = set()
    buckets: dict[str, list[dict]] = {LABEL_TOXIC: [], LABEL_NON_TOXIC: []}
    reasons: Counter = Counter()

    for row in rows:
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            reasons["teks_kosong"] += 1
            continue
        label = map_binary(row)
        if label is None:
            reasons["dibuang_spam_atau_tanpa_label"] += 1
            continue
        cleaned = basic_clean(text)
        final = normalize_slang(cleaned, slang)
        if not final.strip():
            reasons["kosong_setelah_cleaning"] += 1
            continue
        if final in seen:
            reasons["duplikat"] += 1
            continue
        seen.add(final)
        buckets[label].append(
            {
                "text_raw": text,
                "text_clean": cleaned,
                "text_final": final,
                "label": label,
                "text_id": row.get("text_id", ""),
            }
        )

    before = {label: len(items) for label, items in buckets.items()}
    print("Sebelum downsample:", before)

    negatives = buckets[LABEL_NON_TOXIC]
    target = len(buckets[LABEL_TOXIC]) * NEGATIVE_RATIO
    if len(negatives) > target:
        random.Random(SEED).shuffle(negatives)
        buckets[LABEL_NON_TOXIC] = negatives[:target]
        reasons["negatif_di_downsample"] += len(negatives) - target

    records = buckets[LABEL_TOXIC] + buckets[LABEL_NON_TOXIC]
    random.Random(SEED).shuffle(records)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["text_raw", "text_clean", "text_final", "label", "text_id"],
        )
        writer.writeheader()
        writer.writerows(records)

    counts = Counter(record["label"] for record in records)
    summary = {
        "sumber": "IndoDiscourse (Exqrch/IndoDiscourse, Apache-2.0)",
        "sitasi": "Susanto et al., arXiv:2503.00417",
        "skema_label": "biner resmi dataset: toxicity 1 = Toxic, 0 = Non-Toxic",
        "rasio_negatif_positif": f"{NEGATIVE_RATIO}:1",
        "sebelum_downsample": before,
        "setelah_downsample": dict(counts),
        "jumlah_baris_final": len(records),
        "alasan_dibuang": dict(reasons),
        "catatan": (
            "Downsampling dilakukan sebelum split agar train/val/test "
            "berdistribusi sama; meniru setelan 'balanced' pada paper."
        ),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    total = len(records) or 1
    print(f"\nTersimpan: {OUT_CSV} ({len(records):,} baris)")
    for label, count in counts.most_common():
        print(f"  {label:10s} {count:6d}  ({count / total * 100:5.1f}%)")
    print(f"Ringkasan: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
