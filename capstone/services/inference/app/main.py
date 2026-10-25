"""Capstone inference service.

    cd capstone/services/inference && uv run uvicorn app.main:app --port 8001
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from .db import log_predictions, make_engine
from .model import load_model
from .schemas import Classification, ClassifyRequest, ClassifyResponse
from .settings import get_settings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": self.formatTime(record), "level": record.levelname, "msg": record.getMessage()}
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


logger = logging.getLogger("spam-inference")
_h = logging.StreamHandler()
_h.setFormatter(JsonFormatter())
logger.handlers = [_h]
logger.propagate = False


def log(level: int, msg: str, **fields) -> None:
    logger.log(level, msg, extra={"extra_fields": fields})


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    logger.setLevel(s.log_level)
    app.state.settings, app.state.model = s, None
    try:
        app.state.model = load_model(s.model_dir)
        log(logging.INFO, "model loaded", model_version=app.state.model.version)
    except Exception as exc:
        log(logging.ERROR, "model failed to load", error=type(exc).__name__, model_dir=str(s.model_dir))
    app.state.engine = make_engine(s.database_url)
    yield


app = FastAPI(title="SMS Spam Inference", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = rid
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    log(logging.INFO, "request", request_id=rid, method=request.method, path=request.url.path,
        status=response.status_code, latency_ms=round((time.perf_counter() - start) * 1e3, 2))
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request):
    if request.app.state.model is None:
        raise HTTPException(503, "model not loaded")
    return {"status": "ready", "model_version": request.app.state.model.version}


@app.get("/v1/model-info")
def model_info(request: Request):
    m = request.app.state.model
    if m is None:
        raise HTTPException(503, "model not loaded")
    return {"model_version": m.version, "threshold": m.threshold, "test": m.metadata.get("test", {}),
            "target_precision": m.metadata.get("target_precision")}


@app.post("/v1/classify", response_model=ClassifyResponse)
def classify(body: ClassifyRequest, request: Request):
    st = request.app.state
    if st.model is None:
        raise HTTPException(503, "model not loaded")
    if len(body.instances) > st.settings.max_batch:
        raise HTTPException(413, f"maksimal {st.settings.max_batch} pesan per request")
    texts = [m.text for m in body.instances]
    start = time.perf_counter()
    try:
        probs = st.model.predict_proba(texts)
    except Exception as exc:
        log(logging.ERROR, "inference failed", request_id=request.state.request_id, error=type(exc).__name__)
        raise HTTPException(500, "inference failed") from None
    latency_ms = (time.perf_counter() - start) * 1e3
    preds = [Classification(spam_probability=round(p, 4), is_spam=p >= st.model.threshold) for p in probs]
    try:
        log_predictions(st.engine, request.state.request_id, st.model.version, latency_ms,
                        [(t, p.spam_probability, p.is_spam) for t, p in zip(texts, preds)])
    except Exception as exc:
        log(logging.WARNING, "prediction logging failed", error=type(exc).__name__)
    return ClassifyResponse(model_version=st.model.version, threshold=st.model.threshold, predictions=preds)
