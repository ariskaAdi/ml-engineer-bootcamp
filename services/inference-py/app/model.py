"""Memuat artefak model dan menjalankan prediksi. Tidak tahu apa-apa tentang HTTP."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from .schemas import ClientFeatures

# Nama field API (snake_case) -> nama kolom yang dipakai saat training
API_TO_COLUMN = {"emp_var_rate": "emp.var.rate", "cons_price_idx": "cons.price.idx",
                 "cons_conf_idx": "cons.conf.idx", "nr_employed": "nr.employed"}


@dataclass
class LoadedModel:
    pipeline: object
    metadata: dict

    @property
    def version(self) -> str:
        # versi = nama + waktu training + 8 karakter hash data -> unik dan bisa dilacak
        m = self.metadata
        return f"{m['model_name']}-{m['trained_at'][:19].replace(':', '').replace('-', '')}-{m.get('data_sha256', '')[:8]}"

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    def predict_proba(self, instances: list[ClientFeatures]) -> list[float]:
        df = pd.DataFrame([i.model_dump() for i in instances]).rename(columns=API_TO_COLUMN)
        return self.pipeline.predict_proba(df)[:, 1].tolist()


def load_model(model_dir: Path) -> LoadedModel:
    """model_dir berisi model.joblib, metadata.json, dan (opsional) code/ untuk kode yang dirujuk pickle."""
    code_dir = model_dir / "code"
    if code_dir.exists() and str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))   # pickle P1 merujuk fungsi bank_marketing.features.add_features
    pipeline = joblib.load(model_dir / "model.joblib")
    metadata = json.loads((model_dir / "metadata.json").read_text())
    return LoadedModel(pipeline, metadata)
