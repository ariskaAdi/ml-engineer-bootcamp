# %% [markdown]
# # Week 11 — Model berbasis pohon: Decision Tree, Random Forest, Gradient Boosting
#
# Dua dataset: harga rumah (regresi) dan churn (klasifikasi).
# Setiap model dibandingkan dengan 5-fold CV pada data dev; test set disimpan untuk Week 12.

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mlbootcamp.datasets import make_churn, make_house_prices
from mlbootcamp.features import house_pipeline
from mlbootcamp.paths import out


def cv_report(name, model, X, y, cv, scoring):
    t0 = time.perf_counter()
    r = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
    sign = -1 if scoring.startswith("neg_") else 1
    row = {"model": name, "train": sign * r["train_score"].mean(), "val": sign * r["test_score"].mean(),
           "val_std": r["test_score"].std(), "detik": time.perf_counter() - t0}
    print(f"{name:<34} train={row['train']:8.3f}  val={row['val']:8.3f} +- {row['val_std']:.3f}  ({row['detik']:.1f}s)")
    return row

# %% [markdown]
# ## A. Regresi harga rumah (metrik: MAE, makin kecil makin baik)

# %%
house = make_house_prices()
Xh, yh = house.drop(columns=["price"]), house["price"]
Xh_dev, Xh_test, yh_dev, yh_test = train_test_split(Xh, yh, test_size=0.2, random_state=42)
cv5 = KFold(5, shuffle=True, random_state=42)
log_t = lambda m: TransformedTargetRegressor(m, func=np.log, inverse_func=np.exp)

rows = [
    cv_report("Ridge + log (baseline Week 10)", house_pipeline(log_t(Ridge(1.0))), Xh_dev, yh_dev, cv5, "neg_mean_absolute_error"),
]
for depth in [3, 6, 10, None]:
    rows.append(cv_report(f"DecisionTree depth={depth}", house_pipeline(DecisionTreeRegressor(max_depth=depth, random_state=0), scale=False),
                          Xh_dev, yh_dev, cv5, "neg_mean_absolute_error"))
rows.append(cv_report("RandomForest (300 pohon)", house_pipeline(RandomForestRegressor(300, min_samples_leaf=2, random_state=0, n_jobs=1), scale=False),
                      Xh_dev, yh_dev, cv5, "neg_mean_absolute_error"))
rows.append(cv_report("HistGradientBoosting", house_pipeline(HistGradientBoostingRegressor(random_state=0), scale=False),
                      Xh_dev, yh_dev, cv5, "neg_mean_absolute_error"))
rows.append(cv_report("HistGradientBoosting + log", house_pipeline(log_t(HistGradientBoostingRegressor(random_state=0)), scale=False),
                      Xh_dev, yh_dev, cv5, "neg_mean_absolute_error"))
house_results = pd.DataFrame(rows).set_index("model").round(3)

# %% [markdown]
# ## B. Klasifikasi churn (metrik: ROC-AUC, makin besar makin baik)

# %%
churn = make_churn()
Xc, yc = churn.drop(columns=["customer_id", "churn"]), churn["churn"]
Xc_dev, Xc_test, yc_dev, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
num = ["senior_citizen", "tenure_months", "has_tech_support", "paperless_billing",
       "monthly_charges", "total_charges", "support_calls"]
cat = ["contract", "internet_service", "payment_method"]

def churn_pipe(model, scale=True):
    num_steps = [("imp", SimpleImputer(strategy="constant", fill_value=0))] + ([("sc", StandardScaler())] if scale else [])
    prep = ColumnTransformer([("num", Pipeline(num_steps), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return Pipeline([("prep", prep), ("model", model)])

skf = StratifiedKFold(5, shuffle=True, random_state=42)
rows = [cv_report("LogisticRegression (baseline)", churn_pipe(LogisticRegression(max_iter=1000)), Xc_dev, yc_dev, skf, "roc_auc")]
for depth in [3, 5, None]:
    rows.append(cv_report(f"DecisionTree depth={depth}", churn_pipe(DecisionTreeClassifier(max_depth=depth, random_state=0), False), Xc_dev, yc_dev, skf, "roc_auc"))
rows.append(cv_report("RandomForest (300 pohon)", churn_pipe(RandomForestClassifier(300, min_samples_leaf=5, random_state=0, n_jobs=1), False), Xc_dev, yc_dev, skf, "roc_auc"))
rows.append(cv_report("HistGradientBoosting", churn_pipe(HistGradientBoostingClassifier(random_state=0), False), Xc_dev, yc_dev, skf, "roc_auc"))
rows.append(cv_report("HistGradientBoosting (lr=0.05, depth=3)", churn_pipe(HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, random_state=0), False),
                      Xc_dev, yc_dev, skf, "roc_auc"))
churn_results = pd.DataFrame(rows).set_index("model").round(3)

# %% Ringkasan
print("\nRUMAH (MAE):\n", house_results)
print("\nCHURN (ROC-AUC):\n", churn_results)

# %% [markdown]
# ## C. Feature importance — hati-hati!
# Tambahkan 2 kolom ACAK yang sama sekali tidak berhubungan dengan harga.

# %%
rng = np.random.default_rng(0)
Xh_noise = Xh_dev.assign(random_id=rng.integers(0, 10_000, len(Xh_dev)).astype(float),   # banyak nilai unik
                         random_coin=rng.integers(0, 2, len(Xh_dev)).astype(float))      # hanya 2 nilai
feat_cols = ["land_area", "building_area", "bedrooms", "bathrooms", "age_years", "distance_km",
             "has_garage", "random_id", "random_coin"]
Xn = Xh_noise[feat_cols].assign(city_jakarta=(Xh_noise["city"] == "Jakarta").astype(float),
                                years=(Xh_noise["sold_date"] - pd.Timestamp("2022-01-01")).dt.days / 365)
Xn = Xn.fillna(Xn.median())
Xn_tr, Xn_va, yn_tr, yn_va = train_test_split(Xn, yh_dev, test_size=0.3, random_state=0)
rf = RandomForestRegressor(300, random_state=0, n_jobs=-1).fit(Xn_tr, yn_tr)

imp = pd.DataFrame({
    "impurity (bawaan RF)": rf.feature_importances_,
    "permutation (di VALIDATION)": permutation_importance(rf, Xn_va, yn_va, n_repeats=10, random_state=0,
                                                         scoring="neg_mean_absolute_error").importances_mean,
}, index=Xn.columns)
imp["permutation (di VALIDATION)"] /= imp["permutation (di VALIDATION)"].clip(lower=0).sum()
print(imp.sort_values("impurity (bawaan RF)", ascending=False).round(3))
imp.sort_values("impurity (bawaan RF)").plot.barh(figsize=(8, 5), title="Kolom acak 'random_id' terlihat penting di impurity importance")
plt.tight_layout(); plt.savefig(out("week11_1_importance.png")); plt.show()
