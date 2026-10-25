# %% [markdown]
# # Week 3 — EDA: Harga Rumah
#
# ## Pertanyaan (ditulis SEBELUM melihat data)
# 1. Bagaimana distribusi harga? Simetris atau miring? Ada outlier?
# 2. Fitur apa yang paling berhubungan dengan harga?
# 3. Apakah harga berbeda antar kota?
# 4. Apakah harga berubah seiring waktu?
# 5. Apakah jumlah kamar tidur *menyebabkan* harga naik, atau hanya ikut-ikutan luas bangunan?
# 6. Kolom mana yang punya nilai kosong, dan apakah kosongnya acak?
#
# Semua plot disimpan ke `outputs/week03_*.png`. Di VS Code (Run Cell) plot tampil langsung.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mlbootcamp.datasets import make_house_prices
from mlbootcamp.paths import out

sns.set_theme(style="whitegrid")
df = make_house_prices()
print(df.shape)
df.head()

# %% Ringkasan cepat
print(df.dtypes)
print(df.describe().T.round(1))

# %% Q6 — nilai kosong
missing = df.isna().mean().sort_values(ascending=False) * 100
print(missing[missing > 0].round(1))
# Apakah kosongnya acak? Bandingkan harga baris yang kosong vs tidak.
for col in ["building_area", "age_years"]:
    m = df[col].isna()
    print(f"{col}: median harga kosong={df.loc[m, 'price'].median():.0f}, tidak kosong={df.loc[~m, 'price'].median():.0f}")

# %% Q1 — distribusi harga (Plot 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["price"], bins=60, ax=axes[0])
axes[0].axvline(df["price"].mean(), color="red", label=f"mean {df['price'].mean():.0f}")
axes[0].axvline(df["price"].median(), color="green", label=f"median {df['price'].median():.0f}")
axes[0].set_title("Harga (juta Rp)")
axes[0].legend()
sns.histplot(np.log10(df["price"]), bins=60, ax=axes[1])
axes[1].set_title("log10(harga) — lebih simetris")
plt.tight_layout(); plt.savefig(out("week03_1_price_dist.png")); plt.show()
print("skewness harga:", round(df["price"].skew(), 2), "| skewness log:", round(np.log(df["price"]).skew(), 2))

# %% Q1 — outlier dengan aturan IQR (Plot 2)
q1, q3 = df["price"].quantile([0.25, 0.75])
iqr = q3 - q1
upper = q3 + 1.5 * iqr
outliers = df[df["price"] > upper]
print(f"Batas atas IQR = {upper:.0f}; outlier = {len(outliers)} rumah ({len(outliers) / len(df):.1%})")
print(outliers[["city", "land_area", "building_area", "price"]].sort_values("price").tail())
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df, x="city", y="price", ax=ax)
ax.set_title("Harga per kota (titik = outlier menurut IQR)")
plt.tight_layout(); plt.savefig(out("week03_2_box_city.png")); plt.show()

# %% Q2 — korelasi (Plot 3)
num_cols = ["land_area", "building_area", "bedrooms", "bathrooms", "age_years", "distance_km", "has_garage", "price"]
corr = df[num_cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
ax.set_title("Korelasi Spearman")
plt.tight_layout(); plt.savefig(out("week03_3_corr.png")); plt.show()
print(corr["price"].drop("price").sort_values(ascending=False).round(2))

# %% Q2/Q3 — scatter luas tanah vs harga per kota (Plot 4)
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="land_area", y="price", hue="city", alpha=0.5, s=15, ax=ax)
ax.set(xscale="log", yscale="log", title="Luas tanah vs harga (skala log)")
plt.tight_layout(); plt.savefig(out("week03_4_scatter_land.png")); plt.show()

# %% Q3 — harga per m2 per kota: bandingkan "apel dengan apel"
df["price_per_m2_land"] = df["price"] / df["land_area"]
print(df.groupby("city")["price_per_m2_land"].median().sort_values(ascending=False).round(2))

# %% Q4 — tren waktu (Plot 5)
monthly = df.set_index("sold_date").resample("QE")["price_per_m2_land"].median()
fig, ax = plt.subplots(figsize=(9, 4))
monthly.plot(marker="o", ax=ax)
ax.set_title("Median harga per m2 tanah per kuartal")
plt.tight_layout(); plt.savefig(out("week03_5_trend.png")); plt.show()
growth = (monthly.iloc[-4:].mean() / monthly.iloc[:4].mean()) ** (1 / 3) - 1
print(f"Pertumbuhan rata-rata per tahun ~ {growth:.1%}")

# %% Q5 — kamar tidur: korelasi atau sebab? (Plot 6)
# Kalau kamar tidur benar-benar menaikkan harga, efeknya harus tetap ada
# saat luas bangunan dibuat (kurang lebih) sama.
d = df.dropna(subset=["building_area"]).copy()
d["area_bin"] = pd.qcut(d["building_area"], 5)
print("Korelasi bedrooms-harga (semua data):", round(d["bedrooms"].corr(d["price"], method="spearman"), 2))
print("Korelasi bedrooms-harga DI DALAM setiap kelompok luas bangunan:")
print(d.groupby("area_bin", observed=True).apply(
    lambda g: round(g["bedrooms"].corr(g["price"], method="spearman"), 2), include_groups=False))
fig, ax = plt.subplots(figsize=(9, 4))
sns.boxplot(data=d, x="area_bin", y="price", hue="bedrooms", showfliers=False, ax=ax)
ax.set(yscale="log", title="Dalam kelompok luas yang sama, kamar tidur hampir tidak berpengaruh")
ax.tick_params(axis="x", labelrotation=15)
plt.tight_layout(); plt.savefig(out("week03_6_bedrooms_confounding.png")); plt.show()
