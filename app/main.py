"""Web demo FastAPI untuk klasifikasi ujaran kebencian bahasa Indonesia.

Struktur ini kompatibel dengan Vercel FastAPI preset:
- `app.main:app` adalah entrypoint ASGI.
- `public/` di-mount ke `/` dan dipromosikan ke CDN oleh Vercel.
- Endpoint API tersedia di `/api/*` (dan alias lama tanpa prefix untuk lokal).

Jika model/data tidak ikut ter-bundle, app tetap hidup dan menyediakan
`/api/diagnostics` agar penyebab 500 bisa dilihat tanpa membuka log Vercel.
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
STATIC_DIR = ROOT / "public"

MODEL = None
MODEL_ERROR: str | None = None
try:
    if MODEL_PATH.exists():
        MODEL = joblib.load(MODEL_PATH)
    else:
        MODEL_ERROR = f"File model tidak ditemukan: {MODEL_PATH}"
except Exception as exc:  # pragma: no cover - diagnostik runtime
    MODEL_ERROR = f"{type(exc).__name__}: {exc}"

SLANG_MAP: dict[str, str] = {}
SLANG_ERROR: str | None = None
try:
    SLANG_MAP = load_slang_map()
except Exception as exc:  # pragma: no cover - diagnostik runtime
    SLANG_ERROR = f"{type(exc).__name__}: {exc}"

CONFIG: dict = {"use_stemming": False}
CONFIG_ERROR: str | None = None
try:
    if CONFIG_PATH.exists():
        CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    else:
        CONFIG_ERROR = f"File config tidak ditemukan: {CONFIG_PATH}"
except Exception as exc:  # pragma: no cover
    CONFIG_ERROR = f"{type(exc).__name__}: {exc}"

METRICS: dict = {}
METRICS_ERROR: str | None = None
try:
    if METRICS_PATH.exists():
        METRICS = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    else:
        METRICS_ERROR = f"File metrik tidak ditemukan: {METRICS_PATH}"
except Exception as exc:  # pragma: no cover
    METRICS_ERROR = f"{type(exc).__name__}: {exc}"

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
    return {
        "status": "ok" if MODEL is not None else "degraded",
        "model": MODEL_PATH.name,
        "model_loaded": MODEL is not None,
        "slang_entries": len(SLANG_MAP),
    }


@app.get("/diagnostics")
@app.get("/api/diagnostics")
def diagnostics() -> dict:
    return {
        "python": sys.version.split()[0],
        "cwd": str(Path.cwd()),
        "root": str(ROOT),
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "model_loaded": MODEL is not None,
        "model_error": MODEL_ERROR,
        "metrics_path": str(METRICS_PATH),
        "metrics_exists": METRICS_PATH.exists(),
        "metrics_error": METRICS_ERROR,
        "config_path": str(CONFIG_PATH),
        "config_exists": CONFIG_PATH.exists(),
        "config_error": CONFIG_ERROR,
        "slang_entries": len(SLANG_MAP),
        "slang_error": SLANG_ERROR,
        "static_dir": str(STATIC_DIR),
        "static_exists": STATIC_DIR.exists(),
        "use_stemming": bool(CONFIG.get("use_stemming", False)),
    }


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
        "use_stemming": bool(CONFIG.get("use_stemming", False)),
        "model_loaded": MODEL is not None,
        "metrics_error": METRICS_ERROR,
    }


@app.post("/predict", response_model=PredictResponse)
@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model belum termuat. Buka /api/diagnostics untuk melihat "
                f"penyebabnya. {MODEL_ERROR or ''}"
            ).strip(),
        )
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
# try/except menjaga app tetap hidup bila direktori static tidak ikut bundle.
try:
    app.mount("/", StaticFiles(directory="public", html=True), name="static")
except Exception:  # pragma: no cover - Vercel menyajikan public dari CDN
    pass
