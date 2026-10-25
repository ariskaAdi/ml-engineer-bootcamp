"""Memuat model package dan memberi skor. Normalisasi teks HARUS sama dengan training (spam.data.normalize_text)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib


def normalize_text(text: str) -> str:
    # Salinan persis dari capstone/ml/src/spam/data.py (dites di tests/test_consistency.py)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class LoadedModel:
    pipeline: object
    metadata: dict

    @property
    def version(self) -> str:
        m = self.metadata
        return f"{m['model_name']}-{m['trained_at'][:19].replace(':', '').replace('-', '')}-{m['data_sha256'][:8]}"

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    def predict_proba(self, texts: list[str]) -> list[float]:
        return self.pipeline.predict_proba([normalize_text(t) for t in texts])[:, 1].tolist()


def load_model(model_dir: Path) -> LoadedModel:
    return LoadedModel(joblib.load(model_dir / "model.joblib"), json.loads((model_dir / "metadata.json").read_text()))
