"""Membuat dataset penjualan latihan yang sengaja dibuat 'kotor' untuk Week 1."""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # seed: hasil acak selalu sama setiap dijalankan
n = 1000

products = {
    "Kopi Arabika": ("Minuman", 45000),
    "Teh Hijau": ("Minuman", 25000),
    "Roti Gandum": ("Makanan", 18000),
    "Keripik Singkong": ("Makanan", 12000),
    "Madu Hutan": ("Lainnya", 85000),
}

product = rng.choice(list(products), size=n)
dates = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 181, size=n), unit="D")

df = pd.DataFrame({
    "order_id": np.arange(1, n + 1),
    "order_date": dates.strftime("%Y-%m-%d"),
    "product": product,
    "category": [products[p][0] for p in product],
    "quantity": rng.integers(1, 6, size=n),
    "unit_price": [float(products[p][1]) for p in product],
    "city": rng.choice(["Jakarta", "Bandung", "Surabaya", "Yogyakarta"], size=n),
})

# --- Sengaja dikotori, seperti data asli di dunia kerja ---
def pick(k):
    return rng.choice(n, size=k, replace=False)

df.loc[pick(40), "unit_price"] = np.nan                       # harga hilang
idx = pick(30); df.loc[idx, "city"] = df.loc[idx, "city"].str.upper()        # huruf besar semua
idx = pick(30); df.loc[idx, "city"] = "  " + df.loc[idx, "city"] + " "      # spasi berlebih
idx = pick(15); df.loc[idx, "quantity"] = -df.loc[idx, "quantity"]          # angka negatif
df["quantity"] = df["quantity"].astype(str)
idx = pick(10); df.loc[idx, "quantity"] = df.loc[idx, "quantity"] + " pcs"  # angka bercampur teks
df.loc[pick(10), "order_date"] = None                          # tanggal hilang
df = pd.concat([df, df.sample(20, random_state=1)], ignore_index=True)  # baris duplikat

data_dir = Path(__file__).resolve().parents[1] / "data"
data_dir.mkdir(exist_ok=True)
df.to_csv(data_dir / "sales_raw.csv", index=False)
print(f"Tersimpan: {data_dir / 'sales_raw.csv'} ({len(df)} baris)")
