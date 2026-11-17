"""Praproses dataset ujaran kebencian bahasa Indonesia.

Tahapan:
1. Baca CSV mentah (encoding latin-1; berkas asli memuat karakter non-UTF-8).
2. Petakan label multilabel menjadi tiga kelas: Normal, Abusive, Hate Speech.
3. Bersihkan teks (URL, mention, RT, karakter berulang, non-alfabet).
4. Normalisasi slang memakai kamus `new_kamusalay.csv`.
5. Stemming opsional memakai Sastrawi.
6. Simpan dataset bersih + ringkasan untuk laporan.
"""

from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pandas hanya dibutuhkan untuk training/EDA, bukan inference
    import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

LABEL_NORMAL = "Normal"
LABEL_ABUSIVE = "Abusive"
LABEL_HATE = "Hate Speech"
LABELS = [LABEL_NORMAL, LABEL_ABUSIVE, LABEL_HATE]

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
RT_RE = re.compile(r"\brt\b")
USER_URL_TOKEN_RE = re.compile(r"\b(user|url)\b")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
NON_LETTER_RE = re.compile(r"[^a-z\s]")
MULTISPACE_RE = re.compile(r"\s+")

# Nama tokoh politik yang dinetralkan dari teks. Pada dataset 2019, nama-nama
# ini menjadi proksi kubu politik (cebong/kampret) sehingga model menghafalnya
# sebagai sinyal kebencian: "Jokowi" saja bisa memicu Hate Speech 87%.
# Menghapusnya memaksa model bersandar pada kata yang benar-benar menyerang.
# Yang dinetralkan hanya nama orang, bukan nama kelompok atau ideologi.
POLITICAL_ENTITIES = (
    "joko widodo",
    "prabowo subianto",
    "basuki tjahaja purnama",
    "anies baswedan",
    "sandiaga uno",
    "megawati soekarnoputri",
    "abdurrahman wahid",
    "susilo bambang yudhoyono",
    "ridwan kamil",
    "puan maharani",
    "agus harimurti",
    "fadli zon",
    "rocky gerung",
    "ma ruf amin",
    "maruf amin",
    "mahfud md",
    "ganjar pranowo",
    "airlangga hartarto",
    "erick thohir",
    "luhut pandjaitan",
    "rizieq shihab",
    "ratna sarumpaet",
    "najwa shihab",
    "denny siregar",
    "setya novanto",
    "budi gunawan",
    "sri mulyani",
    "rini soemarno",
    "jokowi",
    "prabowo",
    "ahok",
    "anies",
    "sandiaga",
    "megawati",
    "gus dur",
    "sby",
    "ganjar",
    "mahfud",
    "airlangga",
    "luhut",
    "rizieq",
    "setnov",
)

# Alternatif terpanjang didahulukan agar "joko widodo" tidak tersisa "joko".
ENTITY_RE = re.compile(
    r"\b(?:" + "|".join(sorted(POLITICAL_ENTITIES, key=len, reverse=True)) + r")\b"
)


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    """Baca dataset mentah dan petakan label menjadi tiga kelas."""
    import pandas as pd

    path = path or RAW_DIR / "re_dataset.csv"
    frame = pd.read_csv(path, encoding="latin-1")
    required = {"Tweet", "HS", "Abusive"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing)}")

    def map_label(row: pd.Series) -> str:
        if int(row["HS"]) == 1:
            return LABEL_HATE
        if int(row["Abusive"]) == 1:
            return LABEL_ABUSIVE
        return LABEL_NORMAL

    frame = frame.copy()
    frame["label"] = frame.apply(map_label, axis=1)
    frame["text_raw"] = frame["Tweet"].astype(str)
    return frame[["text_raw", "label", "HS", "Abusive"]]


def load_abusive_lexicon(path: Path | None = None) -> set[str]:
    """Muat leksikon kata abusif agar tidak ikut dinormalisasi."""
    path = path or RAW_DIR / "abusive.csv"
    if not path.exists():
        return set()
    values: set[str] = set()
    with path.open("r", encoding="latin-1", newline="") as handle:
        for index, row in enumerate(csv.reader(handle)):
            if not row:
                continue
            value = row[0].strip().lower()
            if index == 0 and value == "abusive":
                continue
            if value:
                values.add(value)
    return values


def _iter_two_columns(path: Path, encoding: str):
    with path.open("r", encoding=encoding, newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                yield row[0], row[1]


def load_slang_map(path: Path | None = None) -> dict[str, str]:
    """Muat kamus slang/typo -> bentuk formal, termasuk pembaruan 2026."""
    path = path or RAW_DIR / "new_kamusalay.csv"
    sources: list[tuple[Path, str]] = [(path, "latin-1")]
    extra_path = RAW_DIR / "slang_2026.csv"
    if extra_path.exists():
        sources.append((extra_path, "utf-8"))

    abusive = load_abusive_lexicon()
    mapping: dict[str, str] = {}
    for source, encoding in sources:
        for slang, formal in _iter_two_columns(source, encoding):
            key = slang.strip().lower()
            value = formal.strip().lower()
            # Kata abusif sengaja dipertahankan: ia justru sinyal kelas.
            if key and value and key != value and key not in abusive:
                mapping[key] = value
    return mapping


def neutralize_entities(text: str) -> str:
    """Buang nama tokoh politik dari teks yang sudah dibersihkan."""
    return MULTISPACE_RE.sub(" ", ENTITY_RE.sub(" ", text)).strip()


def basic_clean(text: str) -> str:
    """Bersihkan teks tanpa stemming, lalu netralkan nama tokoh politik."""
    text = text.lower()
    text = USER_URL_TOKEN_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = RT_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = REPEATED_CHAR_RE.sub(r"\1", text)
    text = NON_LETTER_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return neutralize_entities(text)


def normalize_slang(text: str, slang_map: dict[str, str]) -> str:
    """Ganti token slang dengan bentuk formalnya."""
    tokens = [slang_map.get(token, token) for token in text.split()]
    # Bentuk formal bisa terdiri dari beberapa kata ("anak jakarta asyik asyik").
    return MULTISPACE_RE.sub(" ", " ".join(tokens)).strip()


_STEMMER = None
_STEMMER_READY = False


def _build_stemmer():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    return StemmerFactory().create_stemmer()


def _get_stemmer():
    """Bangun stemmer hanya saat dibutuhkan (inference tidak memakai stemming)."""
    global _STEMMER, _STEMMER_READY
    if not _STEMMER_READY:
        try:
            _STEMMER = _build_stemmer()
        except Exception:  # pragma: no cover - Sastrawi opsional
            _STEMMER = None
        _STEMMER_READY = True
    return _STEMMER


@lru_cache(maxsize=200_000)
def _stem_word(word: str) -> str:
    return _STEMMER.stem(word)


def stem_text(text: str) -> str:
    """Stemming per token dengan cache; aman jika Sastrawi tidak tersedia."""
    if _get_stemmer() is None:
        return text
    return " ".join(_stem_word(token) for token in text.split())


def preprocess_dataframe(
    frame: pd.DataFrame,
    slang_map: dict[str, str],
    use_stemming: bool = True,
) -> pd.DataFrame:
    """Terapkan seluruh tahap praproses dan buang teks duplikat/kosong."""
    result = frame.copy()
    result["text_clean"] = result["text_raw"].map(basic_clean)
    result["text_clean"] = result["text_clean"].map(
        lambda text: normalize_slang(text, slang_map)
    )
    if use_stemming:
        result["text_final"] = result["text_clean"].map(stem_text)
    else:
        result["text_final"] = result["text_clean"]

    result = result[result["text_final"].str.len() >= 3]
    result = result.drop_duplicates(subset="text_final", keep="first")
    result = result.reset_index(drop=True)
    return result


def build_summary(raw: pd.DataFrame, processed: pd.DataFrame) -> dict:
    """Ringkasan angka untuk bab hasil/EDA."""
    length_chars = processed["text_final"].str.len()
    length_words = processed["text_final"].str.split().map(len)
    summary = {
        "jumlah_baris_mentah": int(len(raw)),
        "jumlah_baris_setelah_dedup": int(len(processed)),
        "jumlah_duplikat_dibuang": int(len(raw) - len(processed)),
        "distribusi_kelas": {
            label: int((processed["label"] == label).sum()) for label in LABELS
        },
        "persentase_kelas": {
            label: round(float((processed["label"] == label).mean() * 100), 2)
            for label in LABELS
        },
        "panjang_karakter": {
            "min": int(length_chars.min()),
            "median": float(length_chars.median()),
            "mean": round(float(length_chars.mean()), 2),
            "max": int(length_chars.max()),
        },
        "panjang_kata": {
            "min": int(length_words.min()),
            "median": float(length_words.median()),
            "mean": round(float(length_words.mean()), 2),
            "max": int(length_words.max()),
        },
        "jumlah_kosakata_unik": int(
            len(set(" ".join(processed["text_final"]).split()))
        ),
        "contoh_per_kelas": {
            label: processed.loc[processed["label"] == label, "text_final"]
            .head(3)
            .tolist()
            for label in LABELS
        },
    }
    return summary


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("Membaca dataset mentah...")
    raw = load_raw_dataset()
    slang_map = load_slang_map()
    print(f"Kamus slang: {len(slang_map):,} entri")

    print("Membersihkan dan menormalisasi teks...")
    # Stemming Sastrawi sengaja tidak dipakai pada pipeline utama: biayanya
    # sangat tinggi untuk 13 ribu tweet, sementara char n-gram 3-5 sudah
    # menangani variasi morfologi dan typo. Opsi tetap tersedia untuk eksperimen.
    processed = preprocess_dataframe(raw, slang_map, use_stemming=False)

    output_path = PROCESSED_DIR / "dataset_clean.csv"
    processed[["text_raw", "text_clean", "text_final", "label"]].to_csv(
        output_path, index=False, encoding="utf-8"
    )
    summary = build_summary(raw, processed)
    summary_path = PROCESSED_DIR / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nDataset bersih: {output_path}")
    print(f"Ringkasan: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
