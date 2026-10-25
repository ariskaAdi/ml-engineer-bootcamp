"""Konstanta proyek: path, kolom, seed. Satu tempat supaya mudah diubah."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
RAW_CSV = DATA_DIR / "bank-additional-full.csv"

SEED = 42
TARGET = "y"

# `duration` (lama telepon) DIBUANG: nilainya baru diketahui SETELAH telepon selesai,
# padahal kita ingin memprediksi SEBELUM menelepon -> leakage (disebutkan di dokumentasi UCI).
LEAKY_COLUMNS = ["duration"]

CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "day_of_week", "poutcome"]
NUMERIC = ["age", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx",
           "cons.conf.idx", "euribor3m", "nr.employed"]
ENGINEERED_NUMERIC = ["was_contacted_before"]
