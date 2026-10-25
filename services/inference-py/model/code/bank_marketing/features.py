"""Feature engineering + preprocessing. Semua langkah yang 'belajar' ada di dalam Pipeline."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import CATEGORICAL, ENGINEERED_NUMERIC, LEAKY_COLUMNS, NUMERIC

PDAYS_NEVER = 999  # kode di dataset: nasabah belum pernah dihubungi di kampanye sebelumnya


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tidak mengubah input. Aman dipanggil saat training maupun inference."""
    out = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns]).copy()
    out["was_contacted_before"] = (out["pdays"] != PDAYS_NEVER).astype(int)
    # 999 bukan "999 hari"; ganti -1 supaya model linear tidak salah membaca skala
    out["pdays"] = out["pdays"].where(out["pdays"] != PDAYS_NEVER, -1)
    return out


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    num_cols = NUMERIC + ENGINEERED_NUMERIC
    num = StandardScaler() if scale else "passthrough"
    # "unknown" diperlakukan sebagai kategori sendiri (bisa jadi informatif).
    # Kategori langka (< 20 baris) digabung menjadi satu kolom "infrequent".
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, CATEGORICAL)])


def build_pipeline(model, scale: bool = True) -> Pipeline:
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", build_preprocessor(scale=scale)),
        ("model", model),
    ])
