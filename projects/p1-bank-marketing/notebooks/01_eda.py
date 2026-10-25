# %% [markdown]
# # P1 — EDA: Bank Marketing
# Pertanyaan:
# 1. Seberapa tidak seimbang targetnya?
# 2. Kolom mana yang bocor (tidak tersedia sebelum menelepon)?
# 3. Apa arti nilai khusus (pdays=999, "unknown")?
# 4. Apakah data berurutan waktu, dan apakah kondisinya berubah?
# 5. Segmen mana yang tingkat "yes"-nya tinggi?

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(PROJECT / "src"))
from bank_marketing.data import load_raw  # noqa: E402

raw = load_raw()
print(raw.shape)
print("duplikat:", raw.duplicated().sum())
df = raw.drop_duplicates().assign(y=lambda d: (d["y"] == "yes").astype(int))

# %% Q1 — target
print(df["y"].value_counts(normalize=True).round(3))

# %% Q2 — leakage: `duration`. Telepon 0 detik -> pasti "no"; lama telepon diketahui SETELAH menelepon.
print(df.groupby(pd.qcut(df["duration"], 5))["y"].mean().round(3))
print("duration == 0 -> rate yes =", df.loc[df["duration"] == 0, "y"].mean())

# %% Q3 — nilai khusus
print("pdays == 999 (belum pernah dihubungi):", (df["pdays"] == 999).mean().round(3))
print(df.groupby(df["pdays"] == 999)["y"].mean().rename("yes rate").round(3))
for col in ["job", "marital", "education", "default", "housing", "loan"]:
    print(f"'unknown' di {col}: {(df[col] == 'unknown').mean():.1%}")
print(df.groupby("default")["y"].agg(["mean", "size"]).round(3))   # 'yes' hanya 3 baris!

# %% Q4 — urutan waktu: indikator ekonomi berubah, begitu juga tingkat 'yes'
chunks = pd.Series(range(len(df)), index=df.index) // (len(df) // 10)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
df.groupby(chunks.values)["y"].mean().plot(ax=ax[0], marker="o", title="Tingkat 'yes' per 10% data (urut waktu)")
df.groupby(chunks.values)["euribor3m"].mean().plot(ax=ax[1], marker="o", title="Euribor 3 bulan (urut waktu)")
plt.tight_layout(); plt.show()

# %% Q5 — segmen
for col in ["poutcome", "contact", "month"]:
    print(df.groupby(col)["y"].agg(["mean", "size"]).sort_values("mean", ascending=False).round(3), "\n")
