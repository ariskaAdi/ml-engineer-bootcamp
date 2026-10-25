# %% [markdown]
# # Week 2 — Latihan SQL dari Python
#
# Persiapan (sekali saja):
# 1. `uv run python weeks/week-01-numpy-pandas/05-reference-solution.py`  -> data/sales_clean.csv
# 2. `uv run python scripts/load_sales_to_sqlite.py`                      -> data/sales.db
#
# Cara belajar terbaik: buka file di `sql/`, tebak hasilnya, lalu jalankan cell di bawah.
# Kamu juga bisa membuka `data/sales.db` dengan ekstensi VS Code "SQLite Viewer"
# atau aplikasi DB Browser for SQLite.

# %%
import sqlite3
from pathlib import Path

from mlbootcamp.paths import DATA_DIR
from mlbootcamp.sqlrun import run_file

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
con = sqlite3.connect(DATA_DIR / "sales.db")

# %% Skema database: tabel apa saja, kolom apa saja
for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
    cols = [row[1] for row in con.execute(f"PRAGMA table_info({table})")]
    print(table, cols)

# %% Bagian 1 — dasar
run_file(con, HERE / "sql/01_basics.sql");

# %% Bagian 2 — JOIN dan subquery
run_file(con, HERE / "sql/02_joins.sql");

# %% Bagian 3 — window functions
run_file(con, HERE / "sql/03_window_functions.sql");

# %% Query dengan parameter: JANGAN pakai f-string untuk nilai dari user (SQL injection!)
import pandas as pd

city = "Bandung"
safe = pd.read_sql_query(
    """SELECT COUNT(*) AS n FROM orders o JOIN cities c USING (city_id)
       WHERE c.city_name = ?""",
    con, params=(city,),
)
print(safe)
# Salah:  f"... WHERE city_name = '{city}'"  -> kalau city = "x' OR '1'='1", semua baris bocor.

# %%
con.close()
