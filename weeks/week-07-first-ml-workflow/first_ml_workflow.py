# %% [markdown]
# # Week 7 — Workflow ML pertama: prediksi harga rumah
#
# Alur standar (hafalkan urutannya):
# 1. Tentukan X (fitur) dan y (target)
# 2. Split train / validation / test  ← SEBELUM melihat apa pun yang lain
# 3. Baseline bodoh (DummyRegressor)
# 4. Model sungguhan, latih di TRAIN
# 5. Bandingkan di VALIDATION
# 6. Test set hanya SEKALI di akhir

# %%
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from mlbootcamp.datasets import make_house_prices

# %% 1. X dan y
df = make_house_prices(with_missing=False)   # Week 9 akan menangani nilai kosong
numeric_features = ["land_area", "building_area", "bedrooms", "bathrooms",
                    "age_years", "distance_km", "has_garage"]
X = df[numeric_features]
y = df["price"]
print(X.shape, y.shape)

# %% 2. Split 60/20/20. random_state supaya hasil bisa diulang.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)}")

# %% Metrik — hitung manual sekali supaya paham, lalu pakai sklearn
def report(name, y_true, y_pred):
    err = y_true - y_pred
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    r2 = 1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    assert np.isclose(mae, mean_absolute_error(y_true, y_pred))
    assert np.isclose(rmse, np.sqrt(mean_squared_error(y_true, y_pred)))
    assert np.isclose(r2, r2_score(y_true, y_pred))
    print(f"{name:<28} MAE={mae:8.1f}  RMSE={rmse:8.1f}  R2={r2:6.3f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

# %% 3. Baseline: selalu menebak rata-rata / median harga TRAIN
results = []
for strategy in ["mean", "median"]:
    dummy = DummyRegressor(strategy=strategy).fit(X_train, y_train)
    results.append(report(f"Dummy ({strategy})", y_val, dummy.predict(X_val)))

# %% 4–5. Regresi linear
lin = LinearRegression().fit(X_train, y_train)
results.append(report("LinearRegression", y_val, lin.predict(X_val)))
print("koefisien:", {f: round(float(c), 2) for f, c in zip(numeric_features, lin.coef_)})

# %% Tambahan: kota sebagai fitur (one-hot manual dengan pandas; Week 9 memakai OneHotEncoder)
X_city = pd.get_dummies(df[numeric_features + ["city"]], columns=["city"], drop_first=True, dtype=float)
Xc_train, Xc_val = X_city.loc[X_train.index], X_city.loc[X_val.index]
lin_city = LinearRegression().fit(Xc_train, y_train)
results.append(report("Linear + kota", y_val, lin_city.predict(Xc_val)))

# %% Prediksi log(harga): hubungan perkalian jadi penjumlahan
lin_log = LinearRegression().fit(Xc_train, np.log(y_train))
pred_log = np.exp(lin_log.predict(Xc_val))     # kembalikan ke juta Rupiah sebelum dievaluasi!
results.append(report("Linear + kota, log(y)", y_val, pred_log))

# %% Ringkasan validasi
summary = pd.DataFrame(results).set_index("model").round(3)
print(summary)

# %% 6. Test set — SEKALI, untuk model yang dipilih di validasi (berdasarkan RMSE)
# Catatan: kalau metrik utamanya MAE, model log(y) yang menang. Pilih metrik DULU, baru model.
best = summary["RMSE"].idxmin()
print("Model terpilih:", best)
use_log = "log" in best
Xc_test = X_city.loc[X_test.index]
Xc_trval = pd.concat([Xc_train, Xc_val]); y_trval = pd.concat([y_train, y_val])
final = LinearRegression().fit(Xc_trval, np.log(y_trval) if use_log else y_trval)  # latih ulang di train+val
test_pred = final.predict(Xc_test)
test_pred = np.exp(test_pred) if use_log else test_pred
report("FINAL di TEST", y_test, test_pred)

# %% Analisis error sederhana: di mana model paling salah?
res = df.loc[X_test.index, ["city", "land_area", "price"]].copy()
res["pred"] = test_pred
res["abs_err"] = (res["pred"] - res["price"]).abs()
res["abs_pct_err"] = res["abs_err"] / res["price"] * 100
print(res.groupby("city")["abs_pct_err"].median().round(1).rename("median % error per kota"))
print("5 error terbesar:")
print(res.sort_values("abs_err", ascending=False).head().round(1))
# Error terbesar ada di rumah termahal. Model linear menebak "garis lurus", padahal harga
# naik secara perkalian (kota x luas x tren). Week 11 (tree/boosting) bisa menangkap ini.

# %% [markdown]
# ## Bagian B — dataset asli: California Housing (download otomatis ±400 KB)

# %%
cal = fetch_california_housing(as_frame=True)
Xr, yr = cal.data, cal.target            # target: median harga rumah (satuan 100.000 USD)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=42)
Xr_tr, Xr_va, yr_tr, yr_va = train_test_split(Xr_tr, yr_tr, test_size=0.25, random_state=42)
report("CA Dummy(mean)", yr_va, DummyRegressor().fit(Xr_tr, yr_tr).predict(Xr_va))
report("CA LinearRegression", yr_va, LinearRegression().fit(Xr_tr, yr_tr).predict(Xr_va))
print("Catatan: target dipotong di 5.0 (", (yr >= 5).mean().round(3), "data) -> model linear kesulitan di ujung atas")
