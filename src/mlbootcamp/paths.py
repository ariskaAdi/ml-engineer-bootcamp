"""Lokasi folder penting, supaya script bisa dijalankan dari folder mana pun."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # folder yang berisi pyproject.toml
DATA_DIR = ROOT / "data"          # dataset (tidak di-commit)
MODELS_DIR = ROOT / "models"      # model hasil training (tidak di-commit)
OUTPUTS_DIR = ROOT / "outputs"    # gambar/plot hasil script (tidak di-commit)

for _d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(exist_ok=True)


def out(name: str) -> Path:
    """Path untuk menyimpan plot, misalnya out('week03_hist.png')."""
    return OUTPUTS_DIR / name
