"""Log prediksi untuk monitoring.

PRIVASI: isi SMS adalah data pribadi -> TIDAK disimpan. Yang disimpan hanya:
panjang pesan, hash SHA-256 (untuk mendeteksi pesan yang sama tanpa membaca isinya), skor, keputusan.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, create_engine, insert
from sqlalchemy.engine import Engine

metadata = MetaData()
predictions = Table(
    "sms_predictions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("request_id", String(64), nullable=False),
    Column("model_version", String(128), nullable=False, index=True),
    Column("latency_ms", Float, nullable=False),
    Column("text_length", Integer, nullable=False),
    Column("text_sha256", String(64), nullable=False),
    Column("spam_probability", Float, nullable=False),
    Column("is_spam", Integer, nullable=False),
)


def make_engine(url: str) -> Engine:
    engine = create_engine(url, pool_pre_ping=True)
    metadata.create_all(engine)
    return engine


def log_predictions(engine: Engine, request_id: str, model_version: str, latency_ms: float,
                    rows: list[tuple[str, float, bool]]) -> None:
    now = datetime.now(timezone.utc)
    with engine.begin() as con:
        con.execute(insert(predictions), [
            {"created_at": now, "request_id": request_id, "model_version": model_version, "latency_ms": latency_ms,
             "text_length": len(text), "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
             "spam_probability": p, "is_spam": int(s)} for text, p, s in rows])
