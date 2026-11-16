"""Smoke test endpoint demo web (jalankan setelah uvicorn aktif)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
SAMPLES = {
    "normal": "Saya suka sekali makanan di warung ini, harganya murah dan rasanya enak.",
    "abusive": "Anjir, gua lupa bawa dompet.",
    "hate": "Dasar cebong munafik, kalian semua harus pergi.",
}


def request(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode())


def main() -> int:
    for attempt in range(20):
        try:
            health = request("/health")
            break
        except (urllib.error.URLError, ConnectionError):
            if attempt == 19:
                raise
            time.sleep(1)
    print("HEALTH", health)
    print("MODEL", json.dumps(request("/model-info"), ensure_ascii=False))
    for name, text in SAMPLES.items():
        result = request("/predict", {"text": text})
        print(
            name,
            "->",
            result["label"],
            round(result["confidence"], 4),
            result["probabilities"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
