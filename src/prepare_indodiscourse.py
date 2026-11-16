"""Siapkan IndoDiscourse (2024) sebagai dataset latih 3 kelas.

Dataset asli bersifat multilabel. Skema 3 kelas dibentuk dari anotasi
IndoDiscourse sendiri (bukan dari dataset lama), memakai majority vote
antar-anotator:

    Hate Speech  <- identity_attack = 1
    Abusive      <- insults / profanity_obscenity / threat_incitement_to_violence = 1
    Normal       <- toxicity = 0 dan tidak ada sublabel yang aktif

Aturan di atas dieksekusi berurutan: teks yang sekaligus identity_attack dan
insults dihitung sebagai Hate Speech, karena serangan identitas lebih berat.

Teks yang berlabel `toxicity = 1` tetapi tidak punya sublabel spesifik
dibuang (tidak bisa dipetakan tanpa menebak), begitu juga spam.

Pembersihan memakai fungsi yang sama dengan pipeline Ibrohim
(`basic_clean` + `normalize_slang`) supaya kedua dataset dilatih dengan
distribusi teks yang setara.

Jalankan:

    python src/prepare_indodiscourse.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import basic_clean, load_slang_map, normalize_slang  # noqa: E402

RAW_PATH = ROOT / "data" / "raw" / "indodiscourse_main.jsonl"
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_CSV = PROCESSED_DIR / "indodiscourse_clean.csv"
OUT_SUMMARY = PROCESSED_DIR / "indodiscourse_summary.json"

DEFAULT_URL = (
    "https://huggingface.co/datasets/Exqrch/IndoDiscourse/resolve/main/"
    "indotoxic2024_annotated_data_v2_final.jsonl"
)

LABEL_NORMAL = "Normal"
LABEL_ABUSIVE = "Abusive"
LABEL_HATE = "Hate Speech"

HATE_COLUMNS = ("identity_attack",)
ABUSIVE_COLUMNS = (
    "insults",
    "profanity_obscenity",
    "threat_incitement_to_violence",
)


def majority(value) -> int | None:
    """1 bila mayoritas anotator menandai, 0 bila tidak, None bila kosong."""
    if not value:
        return None
    votes = [str(item).strip() for item in value]
    ones = sum(1 for item in votes if item == "1")
    return 1 if ones * 2 > len(votes) else 0


def download_if_needed() -> Path:
    if RAW_PATH.exists() and RAW_PATH.stat().st_size > 0:
        return RAW_PATH
    import urllib.request

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Mengunduh IndoDiscourse -> {RAW_PATH}")
    urllib.request.urlretrieve(DEFAULT_URL, RAW_PATH)
    return RAW_PATH


def map_label(row: dict) -> tuple[str | None, str]:
    """Kembalikan (label, alasan) ; label None berarti dibuang."""
    if majority(row.get("is_noise_or_spam_text")) == 1:
        return None, "spam"
    if any(majority(row.get(column)) == 1 for column in HATE_COLUMNS):
        return LABEL_HATE, "identity_attack"
    if any(majority(row.get(column)) == 1 for column in ABUSIVE_COLUMNS):
        return LABEL_ABUSIVE, "abusive_sublabel"
    if majority(row.get("toxicity")) == 0:
        return LABEL_NORMAL, "non_toxic"
    return None, "toxic_tanpa_sublabel"


def main() -> None:
    path = download_if_needed()
    slang = load_slang_map()
    print(f"Kamus slang: {len(slang):,} entri")

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Baris mentah: {len(rows):,}")

    reasons: Counter = Counter()
    label_counts: Counter = Counter()
    seen: set[str] = set()
    records: list[dict[str, str]] = []

    for row in rows:
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            reasons["teks_kosong"] += 1
            continue
        label, reason = map_label(row)
        reasons[reason] += 1
        if label is None:
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

        label_counts[label] += 1
        records.append(
            {
                "text_raw": text,
                "text_clean": cleaned,
                "text_final": final,
                "label": label,
                "text_id": row.get("text_id", ""),
            }
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["text_raw", "text_clean", "text_final", "label", "text_id"],
        )
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "sumber": "IndoDiscourse (Exqrch/IndoDiscourse, Apache-2.0)",
        "sitasi": "Susanto et al., arXiv:2503.00417",
        "periode": "September 2023 - Januari 2024",
        "jumlah_baris_mentah": len(rows),
        "jumlah_baris_setelah_cleaning": len(records),
        "distribusi_label": dict(label_counts),
        "alasan_dibuang": {k: v for k, v in reasons.items() if k != "dipetakan"},
        "aturan_pemetaan": {
            "Hate Speech": "identity_attack mayoritas = 1",
            "Abusive": "insults / profanity_obscenity / threat_incitement_to_violence mayoritas = 1 (tanpa identity_attack)",
            "Normal": "toxicity mayoritas = 0 dan seluruh sublabel 0",
            "dibuang": "spam, teks kosong, toxic tanpa sublabel, duplikat",
        },
        "mayority_vote": "1 bila suara '1' > separuh jumlah anotator",
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    total = sum(label_counts.values()) or 1
    print(f"\nTersimpan: {OUT_CSV} ({len(records):,} baris)")
    for label, count in label_counts.most_common():
        print(f"  {label:12s} {count:6d}  ({count / total * 100:5.1f}%)")
    print("\nDibuang:")
    for reason, count in reasons.most_common():
        if reason != "dipetakan":
            print(f"  {reason:26s} {count:6d}")
    print(f"\nRingkasan: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
