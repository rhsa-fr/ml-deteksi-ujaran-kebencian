"""Cari contoh demo yang diklasifikasi benar dan layak ditampilkan."""

from __future__ import annotations

import json
import urllib.request

BASE = "http://127.0.0.1:8000/predict"
CANDIDATES = {
    "Normal": [
        "Terima kasih sudah membantu, semoga harimu menyenangkan.",
        "Saya suka sekali makanan di warung ini, harganya murah dan rasanya enak.",
        "Besok kita belajar kelompok di perpustakaan kampus, jangan lupa bawa buku.",
        "Selamat pagi, semoga semua urusan hari ini berjalan lancar.",
    ],
    "Abusive": [
        "Anjir, dasar bodoh, kerjaanmu selalu berantakan.",
        "Goblok banget sih kamu, kerjaanmu tidak pernah benar.",
        "Dasar bego, ngapain juga kamu ikut campur urusan orang.",
        "Anjir, kamu memang tidak berguna untuk tim ini.",
        "Dasar tolol, ngapain juga kamu bicara di sini.",
    ],
    "Hate Speech": [
        "Dasar cebong, kalian memang tidak pantas tinggal di sini.",
        "Cebong memang pantas diusir dari tempat ini.",
        "Kelompok cebong tidak pantas tinggal di sini, harus diusir.",
        "Dasar cebong munafik, kalian semua harus pergi.",
        "Dasar tolol, kelompok itu harus diusir dari sini.",
        "Orang-orang seperti mereka tidak pantas hidup di sini.",
    ],
}


def predict(text: str) -> dict:
    request = urllib.request.Request(
        BASE,
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode())


def main() -> int:
    for expected, texts in CANDIDATES.items():
        print(f"\n== {expected} ==")
        for text in texts:
            result = predict(text)
            print(
                f"{result['label']:12s} {result['confidence']:.3f}  {text}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
