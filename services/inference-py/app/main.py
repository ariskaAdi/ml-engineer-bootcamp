"""FastAPI app.

Jalankan lokal (dari services/inference-py):
    uv run uvicorn app.main:app --port 8000
Dokumentasi interaktif: http://localhost:8000/docs
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .db import log_predictions, make_engine
from .model import load_model
from .schemas import ModelInfo, Prediction, PredictRequest, PredictResponse
from .settings import get_settings


class JsonFormatter(logging.Formatter):
    """Log terstruktur: satu baris JSON per event -> mudah dicari/diolah oleh tool logging."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": self.formatTime(record), "level": record.levelname, "msg": record.getMessage()}
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


logger = logging.getLogger("inference")
_handler = logging.StreamHandler()
_handler.setFormatter(JsonFormatter())
logger.handlers = [_handler]
logger.propagate = False


def log(level: int, msg: str, **fields) -> None:
    logger.log(level, msg, extra={"extra_fields": fields})


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.setLevel(settings.log_level)
    app.state.settings = settings
    app.state.model = None
    try:
        app.state.model = load_model(settings.model_dir)
        log(logging.INFO, "model loaded", model_version=app.state.model.version)
    except Exception as exc:  # service tetap hidup tapi /ready = 503, supaya penyebabnya terlihat di log
        log(logging.ERROR, "model failed to load", error=type(exc).__name__, model_dir=str(settings.model_dir))
    app.state.engine = make_engine(settings.database_url)
    yield


app = FastAPI(title="Bank Marketing Inference", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    # Hanya metadata request yang dicatat — BUKAN isi body (bisa berisi data pribadi).
    log(logging.INFO, "request", request_id=request_id, method=request.method, path=request.url.path,
        status=response.status_code, latency_ms=round((time.perf_counter() - start) * 1e3, 2))
    return response


@app.get("/health")
def health():
    """Liveness: proses hidup."""
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request):
    """Readiness: siap melayani prediksi (model sudah dimuat)."""
    if request.app.state.model is None:
        raise HTTPException(503, "model not loaded")
    return {"status": "ready", "model_version": request.app.state.model.version}


@app.get("/v1/model-info", response_model=ModelInfo)
def model_info(request: Request):
    m = request.app.state.model
    if m is None:
        raise HTTPException(503, "model not loaded")
    return ModelInfo(model_name=m.metadata["model_name"], model_version=m.version,
                     trained_at=m.metadata["trained_at"], threshold=m.threshold,
                     test_metrics=m.metadata.get("test", {}))


@app.post("/v1/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    state = request.app.state
    if state.model is None:
        raise HTTPException(503, "model not loaded")
    if len(body.instances) > state.settings.max_batch:
        raise HTTPException(413, f"maksimal {state.settings.max_batch} instance per request")
    start = time.perf_counter()
    try:
        probs = state.model.predict_proba(body.instances)
    except Exception as exc:
        log(logging.ERROR, "inference failed", request_id=request.state.request_id, error=type(exc).__name__)
        raise HTTPException(500, "inference failed") from None   # jangan bocorkan detail internal ke klien
    latency_ms = (time.perf_counter() - start) * 1e3
    preds = [Prediction(probability=round(p, 4), should_call=p >= state.model.threshold) for p in probs]
    try:
        log_predictions(state.engine, request.state.request_id, state.model.version, latency_ms,
                        [(i.model_dump(), p.probability, p.should_call) for i, p in zip(body.instances, preds)])
    except Exception as exc:  # gagal mencatat log TIDAK boleh menggagalkan prediksi
        log(logging.WARNING, "prediction logging failed", error=type(exc).__name__)
    return PredictResponse(model_version=state.model.version, threshold=state.model.threshold, predictions=preds)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
