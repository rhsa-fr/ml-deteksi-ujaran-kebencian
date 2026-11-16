"""Contoh kalimat untuk demo: Normal, Abusive, dan Hate Speech."""

from __future__ import annotations

import json
import urllib.request

BASE = "http://127.0.0.1:8000/predict"

EXAMPLES = [
    ("Non-hate / Normal", "Saya suka sekali makanan di warung ini, harganya murah dan rasanya enak."),
    ("Non-hate / Normal", "Terima kasih sudah membantu, semoga harimu menyenangkan."),
    ("Non-hate / Normal", "Besok kita belajar kelompok di perpustakaan kampus, jangan lupa bawa buku."),
    ("Non-hate / Normal", "Saya tidak setuju dengan kebijakan itu, tetapi kita harus berdiskusi dengan baik."),
    ("Non-hate / Normal", "Pemerintah perlu memperbaiki ekonomi dan pendidikan di daerah."),
    ("Non-hate / Abusive", "Anjir, gua lupa bawa dompet."),
    ("Non-hate / Abusive", "Dasar bodoh, kerjaanmu selalu berantakan."),
    ("Non-hate / Abusive", "Goblok banget sih kamu, kerjaanmu tidak pernah benar."),
    ("Non-hate / Abusive", "Dasar bego, ngapain juga kamu ikut campur urusan orang."),
    ("Hate Speech", "Dasar cebong munafik, kalian semua harus pergi dari tempat ini."),
    ("Hate Speech", "Kelompok itu tidak pantas tinggal di sini, mereka semua harus diusir."),
    ("Hate Speech", "Dasar tolol, kelompok itu harus diusir dari sini."),
    ("Hate Speech", "Orang-orang seperti mereka tidak pantas hidup di negeri ini."),
    ("Hate Speech", "Cebong memang pantas diusir dari tempat ini."),
]


def predict(text: str) -> dict:
    request = urllib.request.Request(
        BASE,
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode())


def main() -> int:
    current = None
    for expected, text in EXAMPLES:
        if expected != current:
            current = expected
            print(f"\n== {expected} ==")
        result = predict(text)
        print(
            f"{result['label']:12s} {result['confidence']:.3f}  {text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
