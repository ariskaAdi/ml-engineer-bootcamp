# %% [markdown]
# # Week 2 — Mini-task: 5 pertanyaan bisnis dengan SQL, diverifikasi dengan Pandas
#
# Prinsip: dua cara berbeda yang menghasilkan angka sama = kita lebih yakin angkanya benar.

# %%
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from mlbootcamp.paths import DATA_DIR
from mlbootcamp.sqlrun import run_file

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
con = sqlite3.connect(DATA_DIR / "sales.db")
sql = run_file(con, HERE / "sql/04_business_questions.sql")

# %% Versi Pandas: muat tabel lalu gabungkan (setara JOIN)
orders = pd.read_sql_query("SELECT * FROM orders", con, parse_dates=["order_date"])
products = pd.read_sql_query("SELECT * FROM products", con)
cities = pd.read_sql_query("SELECT * FROM cities", con)
df = orders.merge(products, on="product_id").merge(cities, on="city_id")
df["revenue"] = df["quantity"] * df["unit_price"]

# %% Q1 revenue per bulan
pd_q1 = df.groupby(df["order_date"].dt.strftime("%Y-%m"))["revenue"].sum()
assert np.allclose(pd_q1.values, sql["q1_revenue_per_bulan"]["revenue"].values)

# %% Q2 produk teratas
pd_q2 = df.groupby("product_name")["revenue"].sum().sort_values(ascending=False)
assert list(pd_q2.index) == list(sql["q2_produk_teratas_dan_pangsa"]["product_name"])

# %% Q3 per kota — LEFT JOIN di pandas: merge(how="left")
pd_q3 = (cities.merge(df[["city_id", "order_id", "revenue"]], on="city_id", how="left")
               .groupby("city_name").agg(n_orders=("order_id", "count"), revenue=("revenue", "sum")))
s3 = sql["q3_revenue_per_kota_termasuk_yang_kosong"].set_index("city_name")
assert (pd_q3.loc[s3.index, "n_orders"].values == s3["n_orders"].values).all()
assert pd_q3.loc["Medan", "n_orders"] == 0

# %% Q4 produk terlaris per kota — ROW_NUMBER di pandas: sort + groupby().head(1)
pd_q4 = (df.groupby(["city_name", "product_name"], as_index=False)["revenue"].sum()
           .sort_values(["city_name", "revenue"], ascending=[True, False])
           .groupby("city_name").head(1))
assert list(pd_q4["product_name"]) == list(sql["q4_produk_terlaris_per_kota"]["product_name"])

# %% Q5 pangsa kategori per bulan — window SUM() OVER (PARTITION BY) di pandas: transform("sum")
m = df.assign(month=df["order_date"].dt.strftime("%Y-%m")).groupby(["month", "category"], as_index=False)["revenue"].sum()
m["share"] = (100 * m["revenue"] / m.groupby("month")["revenue"].transform("sum")).round(1)
s5 = sql["q5_kategori_per_bulan_dengan_pangsa"].sort_values(["month", "category"])
assert np.allclose(m.sort_values(["month", "category"])["share"].values, s5["share_in_month_pct"].values)

print("\nSemua jawaban SQL == Pandas: OK")
con.close()
