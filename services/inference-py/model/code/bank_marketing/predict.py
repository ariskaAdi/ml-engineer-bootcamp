"""Inference: muat model tersimpan, prediksi untuk satu atau banyak nasabah.

Contoh CLI (dari root repo):
    uv run python projects/p1-bank-marketing/src/bank_marketing/predict.py
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bank_marketing.config import ARTIFACTS_DIR, CATEGORICAL, NUMERIC  # noqa: E402

REQUIRED_FIELDS = CATEGORICAL + NUMERIC

EXAMPLE_CLIENT = {
    "age": 35, "job": "admin.", "marital": "married", "education": "university.degree",
    "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
    "month": "may", "day_of_week": "thu", "campaign": 1, "pdays": 999, "previous": 0,
    "poutcome": "nonexistent", "emp.var.rate": -1.8, "cons.price.idx": 92.893,
    "cons.conf.idx": -46.2, "euribor3m": 1.313, "nr.employed": 5099.1,
}


@lru_cache(maxsize=1)
def load_model(artifacts_dir: Path = ARTIFACTS_DIR):
    model = joblib.load(artifacts_dir / "model.joblib")
    meta = json.loads((artifacts_dir / "metadata.json").read_text())
    return model, meta


def predict(records: list[dict], artifacts_dir: Path = ARTIFACTS_DIR) -> list[dict]:
    missing = [f for f in REQUIRED_FIELDS if any(f not in r for r in records)]
    if missing:
        raise ValueError(f"field wajib tidak ada: {sorted(set(missing))}")
    model, meta = load_model(artifacts_dir)
    proba = model.predict_proba(pd.DataFrame(records)[REQUIRED_FIELDS])[:, 1]
    return [{"probability": round(float(p), 4), "should_call": bool(p >= meta["threshold"]),
             "model": meta["model_name"]} for p in proba]


if __name__ == "__main__":
    print(predict([EXAMPLE_CLIENT, {**EXAMPLE_CLIENT, "poutcome": "success", "pdays": 6, "previous": 1}]))
