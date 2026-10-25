"""Memuat AG News dan membuat split yang tetap (seed) untuk semua eksperimen."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_DIR / "artifacts"
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
SEED = 42


def clean_text(text: str) -> str:
    """Bersihkan artefak HTML/escape yang ada di AG News (mis. '#39;' = apostrof, '\\' sisa escape)."""
    text = re.sub(r"&lt;.*?&gt;|<[^>]+>", " ", text)
    text = text.replace("#39;", "'").replace("quot;", '"').replace("\\", " ")
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_splits(n_train: int | None = None, n_val: int = 5000, n_test: int | None = None) -> Splits:
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news")
    full = ds["train"].to_pandas()
    test = ds["test"].to_pandas()
    for df in (full, test):
        df["text"] = df["text"].map(clean_text)
    val = full.sample(n_val, random_state=SEED)
    train = full.drop(val.index)
    if n_train:
        train = train.groupby("label", group_keys=False).sample(n_train // 4, random_state=SEED)  # seimbang per kelas
    if n_test:
        test = test.groupby("label", group_keys=False).sample(n_test // 4, random_state=SEED)
    return Splits(train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))
