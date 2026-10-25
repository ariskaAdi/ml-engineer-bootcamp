# %% [markdown]
# # Week 9 — Preprocessing, Pipeline, dan Feature Engineering
#
# Masalah Week 7: kita mengisi nilai kosong, one-hot, dan scaling "dengan tangan" di luar model.
# Mudah lupa, mudah bocor (leakage), dan susah dipakai ulang saat produksi.
# Solusi: Pipeline = preprocessing + model menjadi SATU objek yang di-fit hanya di train.

# %%
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from mlbootcamp.datasets import make_house_prices
from mlbootcamp.paths import MODELS_DIR

df = make_house_prices()                     # SEKARANG dengan nilai kosong
print(df.isna().sum()[lambda s: s > 0])

# %% 1. Imputer, scaler, one-hot — satu per satu dulu
imp = SimpleImputer(strategy="median")
demo = pd.DataFrame({"a": [1.0, np.nan, 3.0, 100.0]})
print("median imputer:", imp.fit_transform(demo).ravel())   # nan -> median(1,3,100)=3

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
cities = pd.DataFrame({"city": ["Jakarta", "Bandung", "Jakarta"]})
ohe.fit(cities)
print(ohe.get_feature_names_out(), ohe.transform(pd.DataFrame({"city": ["Bandung", "Bali"]})))
# "Bali" tidak ada di train -> semua 0 (tidak error) berkat handle_unknown="ignore"

# %% 2. Split (berdasarkan waktu — alasannya dibahas Week 10)
target = "price"
train_df = df[df["sold_date"] < "2025-01-01"]
test_df = df[df["sold_date"] >= "2025-01-01"]
tr_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)
print(f"train={len(tr_df)} val={len(val_df)} test(2025)={len(test_df)}")

# %% 3. Pipeline baseline = rekonstruksi Week 7 (sekarang tanpa membuang baris kosong)
num_cols = ["land_area", "building_area", "bedrooms", "bathrooms", "age_years", "distance_km", "has_garage"]
cat_cols = ["city", "certificate"]

def make_preprocessor(num_cols, cat_cols):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),  # + kolom "apakah tadinya kosong"
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])

base_pipe = Pipeline([("prep", make_preprocessor(num_cols, cat_cols)), ("model", LinearRegression())])
base_pipe.fit(tr_df, tr_df[target])           # SEMUA statistik (median, mean, std, kategori) dari train saja
pred = base_pipe.predict(val_df)
print(f"Baseline pipeline  val MAE={mean_absolute_error(val_df[target], pred):.1f} "
      f"RMSE={root_mean_squared_error(val_df[target], pred):.1f}")

# %% 4. Feature engineering: buat fitur baru, lalu UJI apakah membantu
def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["years_since_2022"] = (X["sold_date"] - pd.Timestamp("2022-01-01")).dt.days / 365  # bagian tanggal -> angka
    X["sold_month"] = X["sold_date"].dt.month
    X["building_to_land"] = X["building_area"] / X["land_area"]                         # rasio
    X["log_land"] = np.log(X["land_area"])
    X["age_group"] = pd.cut(X["age_years"], [-1, 5, 15, 30, 100],
                            labels=["baru", "muda", "sedang", "tua"]).astype(str)        # binning
    X.loc[X["age_years"].isna(), "age_group"] = np.nan
    X["land_x_jakarta"] = X["land_area"] * (X["city"] == "Jakarta")                     # interaksi
    return X

fe_num = num_cols + ["years_since_2022", "building_to_land", "log_land", "land_x_jakarta"]
fe_cat = cat_cols + ["age_group"]

def evaluate(name, pipe):
    pipe.fit(tr_df, tr_df[target])
    p = pipe.predict(val_df)
    mae, rmse = mean_absolute_error(val_df[target], p), root_mean_squared_error(val_df[target], p)
    print(f"{name:<40} MAE={mae:7.1f} RMSE={rmse:7.1f}")
    return mae

feature_step = FunctionTransformer(add_features)   # fungsi biasa dibungkus jadi langkah pipeline
evaluate("+ semua fitur baru", Pipeline([
    ("features", feature_step), ("prep", make_preprocessor(fe_num, fe_cat)), ("model", LinearRegression())]))

# Uji satu per satu (ablation): mana yang benar-benar membantu?
for extra in ["years_since_2022", "building_to_land", "log_land", "land_x_jakarta"]:
    evaluate(f"  baseline + {extra}", Pipeline([
        ("features", feature_step), ("prep", make_preprocessor(num_cols + [extra], cat_cols)),
        ("model", LinearRegression())]))
evaluate("  baseline + age_group (binning)", Pipeline([
    ("features", feature_step), ("prep", make_preprocessor(num_cols, cat_cols + ["age_group"])),
    ("model", LinearRegression())]))

# %% 5. Target log — TransformedTargetRegressor otomatis log saat fit, exp saat predict
log_pipe = Pipeline([
    ("features", feature_step),
    ("prep", make_preprocessor(fe_num, fe_cat)),
    ("model", TransformedTargetRegressor(LinearRegression(), func=np.log, inverse_func=np.exp)),
])
evaluate("semua fitur + log(target)", log_pipe)

# %% 6. Simpan pipeline utuh (preprocessing ikut tersimpan!) lalu muat ulang
final_pipe = log_pipe.fit(train_df, train_df[target])
path = MODELS_DIR / "week09_house_pipeline.joblib"
joblib.dump(final_pipe, path)
loaded = joblib.load(path)
new_house = pd.DataFrame([{
    "sold_date": pd.Timestamp("2025-06-01"), "city": "Bandung", "land_area": 150, "building_area": np.nan,
    "bedrooms": 3, "bathrooms": 2, "age_years": 5, "distance_km": 7.0, "has_garage": 1, "certificate": "SHM"}])
print("Prediksi rumah baru (building_area kosong pun bisa):", loaded.predict(new_house).round(1), "juta")
p_test = loaded.predict(test_df)
print(f"TEST 2025: MAE={mean_absolute_error(test_df[target], p_test):.1f} RMSE={root_mean_squared_error(test_df[target], p_test):.1f}")

# %% [markdown]
# ## 7. Demo leakage: memilih fitur dengan SELURUH data
# 200 fitur NOISE murni, target ACAK. Seharusnya akurasi ~50% (tebak koin).

# %%
rng = np.random.default_rng(0)
X_noise = rng.normal(size=(200, 2000))
y_noise = rng.integers(0, 2, 200)

# SALAH: pilih 20 fitur "terbaik" pakai semua data, BARU cross-validation
X_selected = SelectKBest(f_classif, k=20).fit_transform(X_noise, y_noise)
leaky = cross_val_score(LogisticRegression(max_iter=1000), X_selected, y_noise, cv=5).mean()

# BENAR: seleksi fitur di DALAM pipeline -> diulang di setiap fold hanya dengan data train fold itu
honest = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000)),
                         X_noise, y_noise, cv=5).mean()
print(f"Akurasi data acak - bocor: {leaky:.2f} | jujur (pipeline): {honest:.2f}")
