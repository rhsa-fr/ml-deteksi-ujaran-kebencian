"""Unduh dataset Ibrohim & Budi (2019) dari repositori GitHub resmi.

Sumber:
https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection

Lisensi dataset: CC BY-NC-SA 4.0 (penggunaan akademik/non-komersial).
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE_URL = (
    "https://raw.githubusercontent.com/okkyibrohim/"
    "id-multi-label-hate-speech-and-abusive-language-detection/master/"
)

FILES = {
    "re_dataset.csv": "re_dataset.csv",
    "new_kamusalay.csv": "new_kamusalay.csv",
    "abusive.csv": "abusive.csv",
}

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    if len(payload) < 100:
        raise RuntimeError(f"Unduhan terlalu kecil ({len(payload)} byte): {url}")
    destination.write_bytes(payload)
    print(f"[ok] {destination.name}: {len(payload):,} byte")


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url_name in FILES.items():
        destination = RAW_DIR / filename
        if destination.exists() and destination.stat().st_size > 100:
            print(f"[skip] {filename} sudah ada")
            continue
        download(BASE_URL + url_name, destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
