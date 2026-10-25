"""Memasukkan data penjualan bersih (Week 1) ke database SQLite untuk Week 2.

Data dipecah menjadi 3 tabel (normalisasi), supaya ada yang bisa di-JOIN:
    products(product_id, product_name, category, list_price)
    cities(city_id, city_name, province)
    orders(order_id, order_date, product_id, city_id, quantity, unit_price)

Jalankan:  uv run python scripts/load_sales_to_sqlite.py
Hasil:     data/sales.db
"""

import sqlite3

import pandas as pd

from mlbootcamp.paths import DATA_DIR

src = DATA_DIR / "sales_clean.csv"
if not src.exists():
    raise SystemExit("data/sales_clean.csv belum ada. Jalankan dulu weeks/week-01-numpy-pandas/05-reference-solution.py")

df = pd.read_csv(src, parse_dates=["order_date"])

products = (
    df.groupby(["product", "category"], as_index=False)["unit_price"].median()
    .rename(columns={"product": "product_name", "unit_price": "list_price"})
)
products.insert(0, "product_id", range(1, len(products) + 1))

provinces = {"Jakarta": "DKI Jakarta", "Bandung": "Jawa Barat", "Surabaya": "Jawa Timur",
             "Yogyakarta": "DI Yogyakarta", "Medan": "Sumatera Utara"}
# Medan sengaja ada di tabel cities tapi tidak punya order -> latihan LEFT JOIN
cities = pd.DataFrame({"city_name": sorted(set(df["city"]) | {"Medan"})})
cities["province"] = cities["city_name"].map(provinces)
cities.insert(0, "city_id", range(1, len(cities) + 1))

orders = (
    df.merge(products[["product_id", "product_name"]], left_on="product", right_on="product_name")
      .merge(cities[["city_id", "city_name"]], left_on="city", right_on="city_name")
)
orders = orders[["order_id", "order_date", "product_id", "city_id", "quantity", "unit_price"]]
orders["order_date"] = orders["order_date"].dt.strftime("%Y-%m-%d")  # SQLite menyimpan tanggal sebagai teks ISO

db_path = DATA_DIR / "sales.db"
db_path.unlink(missing_ok=True)
with sqlite3.connect(db_path) as con:
    con.executescript("""
        CREATE TABLE products (
            product_id   INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL UNIQUE,
            category     TEXT NOT NULL,
            list_price   REAL NOT NULL
        );
        CREATE TABLE cities (
            city_id   INTEGER PRIMARY KEY,
            city_name TEXT NOT NULL UNIQUE,
            province  TEXT
        );
        CREATE TABLE orders (
            order_id   INTEGER PRIMARY KEY,
            order_date TEXT NOT NULL,
            product_id INTEGER NOT NULL REFERENCES products(product_id),
            city_id    INTEGER NOT NULL REFERENCES cities(city_id),
            quantity   INTEGER NOT NULL CHECK (quantity > 0),
            unit_price REAL NOT NULL
        );
        CREATE INDEX idx_orders_date ON orders(order_date);
    """)
    products.to_sql("products", con, if_exists="append", index=False)
    cities.to_sql("cities", con, if_exists="append", index=False)
    orders.to_sql("orders", con, if_exists="append", index=False)

print(f"Tersimpan: {db_path}")
print(f"  products={len(products)}, cities={len(cities)}, orders={len(orders)}")
