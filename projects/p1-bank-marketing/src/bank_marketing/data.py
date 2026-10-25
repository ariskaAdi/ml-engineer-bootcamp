"""Download, baca, dan validasi data Bank Marketing (UCI, lisensi CC BY 4.0)."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .config import CATEGORICAL, DATA_URL, NUMERIC, RAW_CSV, TARGET


def download(dest: Path = RAW_CSV) -> Path:
    """Unduh zip UCI (berisi zip lagi di dalamnya) dan ambil bank-additional-full.csv."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATA_URL, timeout=60) as resp:
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    inner = zipfile.ZipFile(io.BytesIO(outer.read("bank-additional.zip")))
    dest.write_bytes(inner.read("bank-additional/bank-additional-full.csv"))
    return dest


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    if not path.exists():
        download(path)
    return pd.read_csv(path, sep=";")


def validate(df: pd.DataFrame) -> None:
    """Gagal cepat kalau data tidak sesuai harapan (kolom hilang, target aneh, umur mustahil)."""
    missing = set(CATEGORICAL + NUMERIC + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"kolom hilang: {sorted(missing)}")
    if not set(df[TARGET].unique()) <= {"yes", "no"}:
        raise ValueError(f"nilai target tidak dikenal: {df[TARGET].unique()}")
    if not df["age"].between(15, 110).all():
        raise ValueError("ada umur di luar 15-110")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Langkah pembersihan deterministik (tidak belajar dari data -> aman dari leakage)."""
    df = df.drop_duplicates().copy()
    df[TARGET] = (df[TARGET] == "yes").astype(int)
    return df.reset_index(drop=True)


def load_dataset() -> pd.DataFrame:
    df = load_raw()
    validate(df)
    return clean(df)
