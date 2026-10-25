"""Salin artefak capstone/ml/artifacts -> capstone/services/inference/model (model package).

    uv run python capstone/ml/src/spam/train.py
    uv run python capstone/services/inference/scripts/export_model.py
"""

import shutil
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "ml" / "artifacts"
DEST = Path(__file__).resolve().parents[1] / "model"
if not (SRC / "model.joblib").exists():
    raise SystemExit("Latih model dulu: uv run python capstone/ml/src/spam/train.py")
DEST.mkdir(exist_ok=True)
for name in ["model.joblib", "metadata.json"]:
    shutil.copy2(SRC / name, DEST / name)
print(f"model package siap: {DEST} ({(DEST / 'model.joblib').stat().st_size / 1e6:.1f} MB)")
