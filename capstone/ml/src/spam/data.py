"""Data SMS Spam Collection (UCI, CC BY 4.0): unduh, validasi, bersihkan, split."""

from __future__ import annotations

import hashlib
import io
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ML_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ML_DIR / "data"
ARTIFACTS = ML_DIR / "artifacts"
RAW = DATA_DIR / "SMSSpamCollection"
URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SEED = 42
MAX_CHARS = 1000


def download(dest: Path = RAW) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as r:
            dest.write_bytes(zipfile.ZipFile(io.BytesIO(r.read())).read("SMSSpamCollection"))
    return dest


def data_hash(path: Path = RAW) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_text(text: str) -> str:
    """Normalisasi ringan yang SAMA saat training dan inference: rapikan spasi saja.
    Huruf besar, angka, dan simbol (£, !!!, 0800...) adalah sinyal spam, jadi tidak dibuang."""
    return re.sub(r"\s+", " ", text).strip()


def load(path: Path | None = None) -> pd.DataFrame:
    path = path or download()
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], quoting=3, encoding="utf-8")
    if not set(df["label"]) <= {"ham", "spam"}:
        raise ValueError(f"label tidak dikenal: {set(df['label'])}")
    df["text"] = df["text"].astype(str).map(normalize_text)
    df["y"] = (df["label"] == "spam").astype(int)
    return df


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Hapus pesan duplikat (teks sama). Kalau tidak, salinan pesan yang sama bisa masuk train DAN test
    sehingga skor test terlalu optimis (leakage)."""
    return df.drop_duplicates(subset="text").reset_index(drop=True)


def split(df: pd.DataFrame, test_size: float = 0.2):
    dev, test = train_test_split(df, test_size=test_size, random_state=SEED, stratify=df["y"])
    return dev.reset_index(drop=True), test.reset_index(drop=True)
