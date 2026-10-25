# %% [markdown]
# # Week 1 — Kunci jawaban (referensi)
#
# File ini BUKAN pengganti notebook-mu. Kerjakan dulu `01`–`04`, baru bandingkan.
# Cara jalan: buka di VS Code lalu klik "Run Cell" di atas setiap `# %%`,
# atau jalankan sekaligus: `uv run python weeks/week-01-numpy-pandas/05-reference-solution.py`
#
# Data: `uv run python scripts/make_sales_data.py` -> `data/sales_raw.csv`

# %% Bagian A — jawaban task-numpy.md
import numpy as np
import pandas as pd

A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
B = np.array([[2, 1, 0, 1], [1, 2, 1, 0], [0, 1, 2, 1]])

print("shape:", A.shape)              # (3, 4) -> 3 baris, 4 kolom
print("A[1,1]:", A[1, 1])             # 6 (baris ke-2, kolom ke-2 versi manusia)
print("baris pertama:", A[0, :])      # [1 2 3 4]
print("kolom ketiga:", A[:, 2])       # [ 3  7 11]
print("baris terakhir:", A[-1, :])    # [ 9 10 11 12]
print("dua kolom pertama:\n", A[:, :2])

# %% Skalar, penjumlahan, elemen-demi-elemen, transpose
print("A + 2:\n", A + 2)
print("A * 2:\n", A * 2)
print("A / 2:\n", A / 2)              # hasilnya float!
print("A + B:\n", A + B)
print("A * B (elemen demi elemen, BUKAN perkalian matriks):\n", A * B)
print("A.T shape:", A.T.shape)        # (4, 3)
# Perkalian matriks butuh kolom kiri == baris kanan: (3,4) @ (4,3) -> (3,3)
print("A @ B.T:\n", A @ B.T)

# %% Sum per axis
print("per baris (axis=1):", A.sum(axis=1))   # [10 26 42]
print("per kolom (axis=0):", A.sum(axis=0))   # [15 18 21 24]
print("total:", A.sum())                      # 78

C = 2 * A + B
D = A - 2 * B
print("C.sum():", C.sum())   # 2*78 + 12 = 168
print("D[2,3]:", D[2, 3])    # 12 - 2*1 = 10

# %% Broadcasting: kenapa (3,1)+(4,) berhasil tapi (3,)+(4,) gagal
col = np.array([[10], [20], [30]])    # (3,1)
row = np.array([1, 2, 3, 4])          # (4,)
print((col + row).shape)              # (3,4): dimensi berukuran 1 "direntangkan"
try:
    np.array([1, 2, 3]) + row
except ValueError as e:
    print("Gagal:", e)                # 3 vs 4, tidak ada yang bernilai 1

# %% [markdown]
# ## Bagian B — Mini-task sales (ringkas)

# %%
from mlbootcamp.paths import DATA_DIR

df = pd.read_csv(DATA_DIR / "sales_raw.csv")
print("Awal:", df.shape)
print(df.isna().sum())

df = df.drop_duplicates()                                   # 1) duplikat
df["city"] = df["city"].str.strip().str.title()             # 2) nama kota
df["quantity"] = pd.to_numeric(df["quantity"].str.replace(" pcs", "", regex=False))  # 3) teks->angka
df = df[df["quantity"] > 0]                                 # 4) quantity negatif dibuang (asumsi: salah input)
df["order_date"] = pd.to_datetime(df["order_date"])         # 5) tanggal
df = df.dropna(subset=["order_date"])
df_before_fill = df.copy()
df["unit_price"] = df["unit_price"].fillna(                 # 6) harga kosong -> median per produk
    df.groupby("product")["unit_price"].transform("median"))
df["revenue"] = df["quantity"] * df["unit_price"]
print("Akhir:", df.shape)   # 975 baris

# %% Pertanyaan bisnis
monthly = df.groupby(df["order_date"].dt.to_period("M"))["revenue"].sum()
print(monthly)
print(df.groupby("product")["revenue"].sum().sort_values(ascending=False))
print(df.groupby("city")["revenue"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False))
print((df.groupby("category")["revenue"].sum() / df["revenue"].sum() * 100).round(1))

# %% Statistik deskriptif "manual" vs pandas (exit criteria)
x = df["revenue"].to_numpy()
mean = x.sum() / len(x)
var = ((x - mean) ** 2).sum() / (len(x) - 1)   # ddof=1, sama seperti pandas
median = np.sort(x)[len(x) // 2] if len(x) % 2 else np.sort(x)[len(x) // 2 - 1: len(x) // 2 + 1].mean()
print(f"mean   manual={mean:,.0f}  pandas={df['revenue'].mean():,.0f}")
print(f"std    manual={var ** 0.5:,.0f}  pandas={df['revenue'].std():,.0f}")
print(f"median manual={median:,.0f}  pandas={df['revenue'].median():,.0f}")

# %% Simpan untuk Week 2 (SQL) dan Week 3 (EDA)
df.to_csv(DATA_DIR / "sales_clean.csv", index=False)
df_before_fill.to_csv(DATA_DIR / "sales_before_fill.csv", index=False)
print("Tersimpan: sales_clean.csv, sales_before_fill.csv")
