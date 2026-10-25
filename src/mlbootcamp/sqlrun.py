"""Membaca file .sql yang berisi banyak query bernama, lalu menjalankannya.

Format file:
    -- name: nama_query
    SELECT ...;
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def load_queries(sql_file: str | Path) -> dict[str, str]:
    text = Path(sql_file).read_text(encoding="utf-8")
    parts = re.split(r"^--\s*name:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    # parts = [header, name1, body1, name2, body2, ...]
    return {name: body.strip().rstrip(";") for name, body in zip(parts[1::2], parts[2::2])}


def run_file(con, sql_file: str | Path, show: bool = True) -> dict[str, pd.DataFrame]:
    results = {}
    for name, query in load_queries(sql_file).items():
        df = pd.read_sql_query(query, con)
        results[name] = df
        if show:
            print(f"\n=== {name} ===")
            print(df.to_string(index=False))
    return results
