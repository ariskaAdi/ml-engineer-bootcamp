# %% [markdown]
# # Week 12 — Hyperparameter Tuning (churn)
#
# Parameter   = dipelajari model dari data (bobot, split pohon).
# Hyperparameter = KITA yang memilih sebelum training (C, max_depth, learning_rate).
# Tuning = mencoba banyak kombinasi hyperparameter dengan CV, pilih yang terbaik.
#
# Aturan: test set hanya dipakai SEKALI di akhir.

# %%
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from mlbootcamp.datasets import make_churn
from mlbootcamp.paths import MODELS_DIR

SEED = 42
df = make_churn()
X, y = df.drop(columns=["customer_id", "churn"]), df["churn"]
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
num = ["senior_citizen", "tenure_months", "has_tech_support", "paperless_billing",
       "monthly_charges", "total_charges", "support_calls"]
cat = ["contract", "internet_service", "payment_method"]


def make_pipe(model, scale=True):
    num_steps = [("imp", SimpleImputer(strategy="constant", fill_value=0))] + ([("sc", StandardScaler())] if scale else [])
    prep = ColumnTransformer([("num", Pipeline(num_steps), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return Pipeline([("prep", prep), ("model", model)])


results = {}

# %% 0. Baseline tanpa tuning
for name, pipe in {"dummy": make_pipe(DummyClassifier()),
                   "logreg_default": make_pipe(LogisticRegression(max_iter=2000)),
                   "hgb_default": make_pipe(HistGradientBoostingClassifier(random_state=SEED), scale=False)}.items():
    s = cross_val_score(pipe, X_dev, y_dev, cv=cv, scoring="roc_auc")
    results[name] = {"cv_auc": s.mean(), "cv_std": s.std(), "params": "default"}
    print(f"{name:<16} CV AUC = {s.mean():.4f} +- {s.std():.4f}")

# %% 1. GridSearchCV — coba SEMUA kombinasi (cocok kalau ruangnya kecil)
grid = GridSearchCV(
    make_pipe(LogisticRegression(max_iter=5000)),
    param_grid={
        "model__C": [0.001, 0.01, 0.1, 1, 10],           # nama: <langkah>__<parameter>
        "model__class_weight": [None, "balanced"],
    },
    cv=cv, scoring="roc_auc", n_jobs=-1, return_train_score=True,
)
t0 = time.perf_counter(); grid.fit(X_dev, y_dev)
print(f"\nGrid LogReg: {len(grid.cv_results_['params'])} kombinasi x 5 fold, {time.perf_counter() - t0:.1f}s")
print("terbaik:", grid.best_params_, f"CV AUC={grid.best_score_:.4f}")
cvr = pd.DataFrame(grid.cv_results_)[["param_model__C", "param_model__class_weight", "mean_train_score", "mean_test_score", "std_test_score"]]
print(cvr.sort_values("mean_test_score", ascending=False).round(4).to_string(index=False))
results["logreg_grid"] = {"cv_auc": grid.best_score_, "cv_std": cvr["std_test_score"].iloc[grid.best_index_], "params": grid.best_params_}

# %% 2. RandomizedSearchCV — ambil N kombinasi acak dari distribusi (cocok untuk banyak hyperparameter)
rand = RandomizedSearchCV(
    make_pipe(HistGradientBoostingClassifier(random_state=SEED, early_stopping=False), scale=False),
    param_distributions={
        "model__learning_rate": loguniform(0.01, 0.3),       # skala log: 0.01 dan 0.1 sama "jauhnya"
        "model__max_depth": [2, 3, 4, 6, None],
        "model__max_iter": randint(50, 400),
        "model__min_samples_leaf": randint(10, 200),
        "model__l2_regularization": loguniform(1e-3, 10),
    },
    n_iter=40, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=SEED, return_train_score=True,
)
t0 = time.perf_counter(); rand.fit(X_dev, y_dev)
print(f"\nRandom HGB: 40 kombinasi x 5 fold, {time.perf_counter() - t0:.1f}s")
print("terbaik:", {k: (float(round(v, 4)) if isinstance(v, float) else v) for k, v in rand.best_params_.items()},
      f"CV AUC={rand.best_score_:.4f}")
rcv = pd.DataFrame(rand.cv_results_)
results["hgb_random"] = {"cv_auc": rand.best_score_, "cv_std": rcv["std_test_score"].iloc[rand.best_index_],
                         "params": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in rand.best_params_.items()}}
gap = rcv["mean_train_score"].iloc[rand.best_index_] - rand.best_score_
print(f"selisih train-val kombinasi terbaik: {gap:.4f} (default tadi jauh lebih besar)")

# %% 3. Ide tambahan: LogReg + interaksi fitur (polinomial derajat 2)
poly = GridSearchCV(
    Pipeline([("prep", make_pipe(LogisticRegression()).named_steps["prep"]),
              ("poly", PolynomialFeatures(2, interaction_only=True, include_bias=False)),
              ("sc", StandardScaler(with_mean=False)),
              ("model", LogisticRegression(max_iter=5000))]),
    {"model__C": [0.001, 0.003, 0.01, 0.03]}, cv=cv, scoring="roc_auc", n_jobs=-1)
poly.fit(X_dev, y_dev)
results["logreg_poly_grid"] = {"cv_auc": poly.best_score_, "cv_std": pd.DataFrame(poly.cv_results_)["std_test_score"].iloc[poly.best_index_],
                               "params": poly.best_params_}
print(f"\nLogReg + interaksi: CV AUC={poly.best_score_:.4f} {poly.best_params_}")

# %% 4. Pilih model berdasarkan CV (BUKAN test)
summary = pd.DataFrame(results).T
summary["cv_auc"] = summary["cv_auc"].astype(float); summary["cv_std"] = summary["cv_std"].astype(float)
summary = summary.sort_values("cv_auc", ascending=False)
print("\n", summary[["cv_auc", "cv_std"]].round(4))
best_name = summary.index[0]
# Aturan "satu standard error": kalau model yang lebih sederhana masih dalam 1 std dari yang terbaik, pilih yang sederhana.
simple_order = ["logreg_grid", "logreg_default", "logreg_poly_grid", "hgb_random", "hgb_default"]
threshold = summary.loc[best_name, "cv_auc"] - summary.loc[best_name, "cv_std"]
chosen = next(n for n in simple_order if n in summary.index and summary.loc[n, "cv_auc"] >= threshold)
print(f"terbaik CV: {best_name} | dipilih (aturan 1-SE, utamakan sederhana): {chosen}")

# %% 5. Evaluasi FINAL di test — sekali saja
final_model = {"logreg_grid": grid, "hgb_random": rand, "logreg_poly_grid": poly}.get(chosen, grid).best_estimator_
# GridSearchCV (refit=True) sudah melatih ulang model terbaik di SELURUH data dev
proba = final_model.predict_proba(X_test)[:, 1]
test_auc, test_ap = roc_auc_score(y_test, proba), average_precision_score(y_test, proba)
print(f"TEST: ROC-AUC={test_auc:.4f}  PR-AUC={test_ap:.4f}")

# %% 6. Simpan model + catatan eksperimen supaya bisa direproduksi
joblib.dump(final_model, MODELS_DIR / "week12_churn_model.joblib")
record = {"seed": SEED, "chosen": chosen, "cv_results": summary[["cv_auc", "cv_std"]].round(4).to_dict(),
          "chosen_params": str(results[chosen]["params"]), "test_roc_auc": round(test_auc, 4), "test_pr_auc": round(test_ap, 4)}
(MODELS_DIR / "week12_experiment.json").write_text(json.dumps(record, indent=2, default=str))
print("tersimpan: models/week12_churn_model.joblib, models/week12_experiment.json")
