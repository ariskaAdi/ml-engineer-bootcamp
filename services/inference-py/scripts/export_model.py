"""Salin artefak P1 ke services/inference-py/model/ (= "model package" yang ikut ke Docker image).

    uv run python projects/p1-bank-marketing/src/bank_marketing/train.py   # kalau belum
    uv run python services/inference-py/scripts/export_model.py

Kenapa kode ikut disalin? Pipeline P1 disimpan dengan pickle, dan pickle hanya menyimpan NAMA fungsi
`bank_marketing.features.add_features`, bukan isinya. Tanpa kode itu, joblib.load() gagal.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P1 = ROOT / "projects" / "p1-bank-marketing"
DEST = Path(__file__).resolve().parents[1] / "model"

if not (P1 / "artifacts" / "model.joblib").exists():
    raise SystemExit("Model P1 belum ada. Jalankan dulu projects/p1-bank-marketing/src/bank_marketing/train.py")

if DEST.exists():
    shutil.rmtree(DEST)
(DEST / "code" / "bank_marketing").mkdir(parents=True)
for name in ["model.joblib", "metadata.json"]:
    shutil.copy2(P1 / "artifacts" / name, DEST / name)
for name in ["__init__.py", "config.py", "features.py"]:
    shutil.copy2(P1 / "src" / "bank_marketing" / name, DEST / "code" / "bank_marketing" / name)
print(f"model package siap di {DEST}")
for p in sorted(DEST.rglob("*")):
    if p.is_file():
        print("  ", p.relative_to(DEST), f"{p.stat().st_size / 1024:.0f} KB")
