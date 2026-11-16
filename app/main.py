"""Web demo FastAPI untuk klasifikasi ujaran kebencian bahasa Indonesia.

Struktur ini kompatibel dengan Vercel FastAPI preset:
- `app.main:app` adalah entrypoint ASGI.
- `public/` di-mount ke `/` dan dipromosikan ke CDN oleh Vercel.
- Endpoint API tersedia di `/api/*` (dan alias lama tanpa prefix untuk lokal).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import predict_text  # noqa: E402
from src.preprocess import load_slang_map  # noqa: E402

MODEL_PATH = ROOT / "models" / "best_model.joblib"
METRICS_PATH = ROOT / "results" / "metrics.json"
CONFIG_PATH = ROOT / "models" / "preprocess_config.json"

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model belum ada di {MODEL_PATH}. Jalankan `python src/train.py` dahulu."
    )

MODEL = joblib.load(MODEL_PATH)
SLANG_MAP = load_slang_map()
CONFIG = (
    json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if CONFIG_PATH.exists()
    else {"use_stemming": False}
)
METRICS = (
    json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if METRICS_PATH.exists()
    else {}
)

app = FastAPI(
    title="Deteksi Ujaran Kebencian Bahasa Indonesia",
    description="Demo klasifikasi teks: Normal, Abusive, Hate Speech.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class TokenContribution(BaseModel):
    token: str
    weight: float


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    cleaned_text: str
    supporting_tokens: list[TokenContribution]
    opposing_tokens: list[TokenContribution]


@app.get("/health")
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_PATH.name}


@app.get("/model-info")
@app.get("/api/model-info")
def model_info() -> dict:
    best = METRICS.get("best_metrics", {})
    return {
        "model": METRICS.get("best_model", "unknown"),
        "accuracy": best.get("accuracy"),
        "macro_f1": best.get("macro_f1"),
        "per_class": best.get("per_class", {}),
        "test_size": METRICS.get("test_size"),
        "dataset": METRICS.get("dataset", {}).get("jumlah_baris_setelah_dedup"),
        "use_stemming": CONFIG.get("use_stemming", False),
    }


@app.post("/predict", response_model=PredictResponse)
@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Teks tidak boleh kosong.")
    result = predict_text(
        MODEL,
        text,
        SLANG_MAP,
        use_stemming=bool(CONFIG.get("use_stemming", False)),
    )
    return PredictResponse(**result)


# Mount paling akhir: route API di atas tetap menang atas file statis.
# Vercel mempromosikan direktori `public` ini ke CDN.
app.mount("/", StaticFiles(directory="public", html=True), name="static")
