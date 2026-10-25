# %% [markdown]
# # Week 10 — Generalisasi, Overfitting, dan Cross-Validation

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (KFold, TimeSeriesSplit, cross_val_score, cross_validate,
                                     learning_curve, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from mlbootcamp.datasets import make_house_prices
from mlbootcamp.features import house_pipeline
from mlbootcamp.paths import out

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Underfitting vs overfitting (data 1 dimensi supaya bisa digambar)

# %%
x = np.sort(rng.uniform(0, 1, 20))   # sedikit data -> model kompleks mudah overfit
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.25, len(x))
x_val = np.sort(rng.uniform(0, 1, 200))
y_val = np.sin(2 * np.pi * x_val) + rng.normal(0, 0.25, len(x_val))

degrees = range(1, 16)
train_err, val_err = [], []
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
grid = np.linspace(0, 1, 300)[:, None]
for d in degrees:
    m = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression()).fit(x[:, None], y)
    train_err.append(mean_squared_error(y, m.predict(x[:, None])))
    val_err.append(mean_squared_error(y_val, m.predict(x_val[:, None])))
    if d in (1, 4, 15):
        ax = axes[[1, 4, 15].index(d)]
        ax.scatter(x, y, s=12); ax.plot(grid, np.clip(m.predict(grid), -2, 2), "r")
        ax.set_title({1: "degree 1: UNDERFIT", 4: "degree 4: pas", 15: "degree 15: OVERFIT"}[d])
axes[3].plot(degrees, train_err, "o-", label="train"); axes[3].plot(degrees, val_err, "o-", label="validation")
axes[3].set(yscale="log", xlabel="kompleksitas (degree)", ylabel="MSE", title="Kurva kompleksitas"); axes[3].legend()
plt.tight_layout(); plt.savefig(out("week10_1_under_overfit.png")); plt.show()
for d in (1, 4, 15):
    print(f"degree {d:>2}: train MSE={train_err[d - 1]:.3f}  val MSE={val_err[d - 1]:.3f}")

# %% [markdown]
# ## 2. Regularisasi: model kompleks tapi bobotnya "dihukum" kalau terlalu besar
# Ridge: loss + alpha * sum(w^2) ; Lasso: loss + alpha * sum(|w|) -> bisa membuat bobot = 0

# %%
alphas = np.logspace(-6, 2, 30)
ridge_val = []
for a in alphas:
    m = make_pipeline(PolynomialFeatures(15), StandardScaler(), Ridge(alpha=a)).fit(x[:, None], y)
    ridge_val.append(mean_squared_error(y_val, m.predict(x_val[:, None])))
best_alpha = alphas[int(np.argmin(ridge_val))]
print(f"degree 15 tanpa regularisasi val MSE={val_err[14]:.3f} | Ridge alpha={best_alpha:.4f} val MSE={min(ridge_val):.3f}")

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(alphas, ridge_val, "o-"); ax.set(xscale="log", yscale="log", xlabel="alpha (kekuatan regularisasi)",
                                         ylabel="val MSE", title="Ridge: alpha terlalu kecil = overfit, terlalu besar = underfit")
plt.tight_layout(); plt.savefig(out("week10_2_ridge_alpha.png")); plt.show()

poly_x = StandardScaler().fit_transform(PolynomialFeatures(15, include_bias=False).fit_transform(x[:, None]))
for name, model in [("Linear", LinearRegression()), ("Ridge(0.01)", Ridge(0.01)), ("Lasso(0.01)", Lasso(0.01, max_iter=50_000))]:
    w = model.fit(poly_x, y).coef_
    print(f"{name:<12} max|w|={np.abs(w).max():12.1f}  bobot = 0: {(np.abs(w) < 1e-8).sum()}/15")

# %% [markdown]
# ## 3. K-fold cross-validation pada data rumah
# Satu split validasi = satu angka yang bisa beruntung/sial. K-fold: 5 split, 5 angka -> mean ± std.

# %%
house = make_house_prices()
X, y_h = house.drop(columns=["price"]), house["price"]
X_dev, X_test, y_dev, y_test = train_test_split(X, y_h, test_size=0.2, random_state=42)

model = house_pipeline(TransformedTargetRegressor(Ridge(1.0), func=np.log, inverse_func=np.exp))
# Variasi antar split tunggal:
for seed in range(3):
    a, b, ya, yb = train_test_split(X_dev, y_dev, test_size=0.25, random_state=seed)
    print(f"split acak seed={seed}: val MAE={np.mean(np.abs(yb - model.fit(a, ya).predict(b))):.1f}")

cv = cross_validate(model, X_dev, y_dev, cv=KFold(5, shuffle=True, random_state=42),
                    scoring="neg_mean_absolute_error", return_train_score=True)
print("5-fold val MAE   :", (-cv["test_score"]).round(1), f"-> {-cv['test_score'].mean():.1f} +- {cv['test_score'].std():.1f}")
print("5-fold train MAE :", f"{-cv['train_score'].mean():.1f}  (selisih kecil -> tidak overfit)")

# %% Learning curve: apakah menambah data masih membantu?
sizes, tr_scores, va_scores = learning_curve(model, X_dev, y_dev, cv=5, scoring="neg_mean_absolute_error",
                                             train_sizes=np.linspace(0.05, 1.0, 8), shuffle=True, random_state=0)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(sizes, -tr_scores.mean(1), "o-", label="train"); ax.plot(sizes, -va_scores.mean(1), "o-", label="validation (CV)")
ax.set(xlabel="jumlah data train", ylabel="MAE", title="Learning curve"); ax.legend()
plt.tight_layout(); plt.savefig(out("week10_3_learning_curve.png")); plt.show()

# %% [markdown]
# ## 4. Data berurutan waktu: random split BOCOR
# Penjualan harian toko dengan tren + musiman + noise yang "lengket" (hari ini mirip kemarin).
# Tugas: memprediksi MASA DEPAN. Model diberi fitur "nomor hari" dan hari-dalam-minggu.

# %%
days = pd.date_range("2023-01-01", periods=730, freq="D")
level = np.cumsum(rng.normal(0, 3, len(days)))                # random walk -> autokorelasi
sales = 200 + level + 25 * (days.dayofweek >= 5) + rng.normal(0, 5, len(days))
ts = pd.DataFrame({"t": np.arange(len(days)), "dow": days.dayofweek, "sales": sales})
Xt, yt = ts[["t", "dow"]], ts["sales"]

from sklearn.ensemble import RandomForestRegressor  # model fleksibel (dibahas Week 11)
rf = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
random_cv = -cross_val_score(rf, Xt, yt, cv=KFold(5, shuffle=True, random_state=0), scoring="neg_mean_absolute_error")
time_cv = -cross_val_score(rf, Xt, yt, cv=TimeSeriesSplit(5), scoring="neg_mean_absolute_error")
print(f"KFold acak      MAE = {random_cv.mean():.1f}  <- terlalu optimis")
print(f"TimeSeriesSplit MAE = {time_cv.mean():.1f}  <- realistis untuk memprediksi masa depan")
# Dengan split acak, untuk menebak hari ke-100 model sudah melihat hari ke-99 dan ke-101.
# Itu "melihat masa depan" — tidak mungkin terjadi saat produksi.

for i, (tr, va) in enumerate(TimeSeriesSplit(5).split(Xt)):
    print(f"fold {i}: train hari {tr[0]}-{tr[-1]}  ->  validasi hari {va[0]}-{va[-1]}")

# %% Data rumah: random split vs split waktu
mask_tr = house["sold_date"] < "2025-01-01"
m_time = model.fit(X[mask_tr], y_h[mask_tr])
mae_time = np.mean(np.abs(y_h[~mask_tr] - m_time.predict(X[~mask_tr])))
print(f"Rumah: CV acak MAE={-cv['test_score'].mean():.1f} | latih 2022-2024, uji 2025 MAE={mae_time:.1f}")
