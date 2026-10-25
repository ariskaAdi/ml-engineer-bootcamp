# %% [markdown]
# # Week 2 (opsional) — Data yang sama di PostgreSQL
#
# SQLite = database dalam satu file, tanpa server. Cocok untuk belajar dan aplikasi kecil.
# PostgreSQL = database server sungguhan, yang paling sering dipakai di perusahaan.
#
# Langkah:
# 1. Nyalakan Docker Desktop.
# 2. `docker compose -f weeks/week-02-sql/docker-compose.yml up -d`
# 3. Jalankan file ini.
#
# Kita memakai SQLAlchemy: satu cara menulis koneksi untuk banyak jenis database.
# URL koneksi:  dialect+driver://user:password@host:port/database

# %%
import os
import sqlite3

import pandas as pd
from sqlalchemy import create_engine, text

from mlbootcamp.paths import DATA_DIR

PG_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://bootcamp:bootcamp@localhost:5432/sales")
engine = create_engine(PG_URL)

# %% Salin 3 tabel dari SQLite ke PostgreSQL
with sqlite3.connect(DATA_DIR / "sales.db") as lite:
    tables = {t: pd.read_sql_query(f"SELECT * FROM {t}", lite) for t in ["products", "cities", "orders"]}
tables["orders"]["order_date"] = pd.to_datetime(tables["orders"]["order_date"]).dt.date  # tipe DATE asli

with engine.begin() as con:  # begin() = transaksi; otomatis COMMIT kalau sukses, ROLLBACK kalau error
    con.execute(text("DROP TABLE IF EXISTS orders, products, cities"))
    con.execute(text("""
        CREATE TABLE products (product_id INT PRIMARY KEY, product_name TEXT UNIQUE NOT NULL,
                               category TEXT NOT NULL, list_price NUMERIC(12,2) NOT NULL);
        CREATE TABLE cities   (city_id INT PRIMARY KEY, city_name TEXT UNIQUE NOT NULL, province TEXT);
        CREATE TABLE orders   (order_id INT PRIMARY KEY, order_date DATE NOT NULL,
                               product_id INT NOT NULL REFERENCES products, city_id INT NOT NULL REFERENCES cities,
                               quantity INT NOT NULL CHECK (quantity > 0), unit_price NUMERIC(12,2) NOT NULL);
    """))
    for name in ["products", "cities", "orders"]:  # urutan penting karena FOREIGN KEY
        tables[name].to_sql(name, con, if_exists="append", index=False)
print("Tabel tersalin ke PostgreSQL")

# %% Perbedaan dialek: per bulan di PostgreSQL pakai date_trunc, di SQLite pakai strftime
q = """
WITH monthly AS (
    SELECT date_trunc('month', order_date)::date AS month,
           SUM(quantity * unit_price)            AS revenue
    FROM orders
    GROUP BY 1
)
SELECT month, revenue,
       SUM(revenue) OVER (ORDER BY month) AS running_total
FROM monthly
ORDER BY month
"""
print(pd.read_sql_query(text(q), engine))

# %% Parameter yang aman (named parameter :city)
q = text("SELECT COUNT(*) AS n FROM orders JOIN cities USING (city_id) WHERE city_name = :city")
print(pd.read_sql_query(q, engine, params={"city": "Bandung"}))

# %% EXPLAIN: lihat rencana eksekusi query (untuk memahami index)
with engine.connect() as con:
    for row in con.execute(text("EXPLAIN SELECT * FROM orders WHERE order_id = 10")):
        print(row[0])
