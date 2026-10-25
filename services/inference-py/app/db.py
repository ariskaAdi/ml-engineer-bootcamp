"""Log prediksi ke database (SQLite lokal, PostgreSQL di Docker) untuk monitoring Week 36.

Yang disimpan: fitur yang dibutuhkan untuk memantau drift + hasil + latensi.
Yang TIDAK disimpan: identitas nasabah (nama, nomor telepon, ID). Fitur saja tidak menunjuk orang tertentu.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, String, Table, Text, create_engine,
                        insert)
from sqlalchemy.engine import Engine

metadata = MetaData()
predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("request_id", String(64), nullable=False),
    Column("model_version", String(128), nullable=False, index=True),
    Column("latency_ms", Float, nullable=False),
    Column("probability", Float, nullable=False),
    Column("should_call", Integer, nullable=False),
    Column("features", Text, nullable=False),     # JSON
)


def make_engine(url: str) -> Engine:
    engine = create_engine(url, pool_pre_ping=True)
    metadata.create_all(engine)                    # buat tabel kalau belum ada
    return engine


def log_predictions(engine: Engine, request_id: str, model_version: str, latency_ms: float,
                    rows: list[tuple[dict, float, bool]]) -> None:
    now = datetime.now(timezone.utc)
    with engine.begin() as con:
        con.execute(insert(predictions), [
            {"created_at": now, "request_id": request_id, "model_version": model_version, "latency_ms": latency_ms,
             "probability": p, "should_call": int(c), "features": json.dumps(f)} for f, p, c in rows])
