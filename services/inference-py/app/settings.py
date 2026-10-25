"""Konfigurasi dari environment variable (12-factor app). Tidak ada rahasia di kode."""

import os
from dataclasses import dataclass
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    database_url: str
    log_level: str
    max_batch: int


def get_settings() -> Settings:
    return Settings(
        model_dir=Path(os.getenv("MODEL_DIR", SERVICE_DIR / "model")),
        database_url=os.getenv("DATABASE_URL", f"sqlite:///{SERVICE_DIR / 'predictions.db'}"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        max_batch=int(os.getenv("MAX_BATCH", "100")),
    )
