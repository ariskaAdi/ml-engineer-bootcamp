"""Catatan eksperimen sederhana: satu baris JSON per eksperimen.

Contoh:
    log_experiment("week16", hypothesis="...", changed="...", params={...}, metrics={...}, conclusion="...")
Hasil: outputs/experiments_week16.jsonl  (bisa dibaca dengan pd.read_json(path, lines=True))
"""

from __future__ import annotations

import json
from datetime import datetime

from .paths import OUTPUTS_DIR


def log_experiment(name: str, **fields) -> dict:
    record = {"time": datetime.now().isoformat(timespec="seconds"), **fields}
    path = OUTPUTS_DIR / f"experiments_{name}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record
