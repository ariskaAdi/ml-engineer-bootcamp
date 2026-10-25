# %% [markdown]
# # Week 16 — Interpretabilitas dan Eksperimen Terkontrol
#
# Tiga pertanyaan yang SERING tertukar:
# 1. Asosiasi   : fitur X bergerak bersama target y di data.
# 2. Penjelasan : model memakai fitur X untuk memprediksi y (permutation importance, SHAP).
# 3. Kausalitas : kalau kita MENGUBAH X, y ikut berubah.
# Alat interpretabilitas menjawab (2). Hanya eksperimen (atau asumsi kuat) yang menjawab (3).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mlbootcamp.datasets import make_churn
from mlbootcamp.experiments import log_experiment
from mlbootcamp.paths import out

SEED = 42
df = make_churn()
df[["senior_citizen", "tenure_months", "has_tech_support", "paperless_billing", "support_calls"]] = df[
    ["senior_citizen", "tenure_months", "has_tech_support", "paperless_billing", "support_calls"]].astype(float)  # PDP butuh float
num = ["senior_citizen", "tenure_months", "has_tech_support", "paperless_billing",
       "monthly_charges", "total_charges", "support_calls"]
cat = ["contract", "internet_service", "payment_method"]
X, y = df[num + cat], df["churn"]
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)


def make_model(num_cols=num, cat_cols=cat, **hgb):
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="constant", fill_value=0), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ], verbose_feature_names_out=False).set_output(transform="pandas")
    model = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, random_state=SEED, **hgb)
    return Pipeline([("prep", prep), ("model", model)])


pipe = make_model().fit(X_tr, y_tr)
print(f"val ROC-AUC = {roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1]):.4f}")

# %% 1. Permutation importance (di VALIDATION, pada kolom ASLI sebelum preprocessing)
pi = permutation_importance(pipe, X_val, y_val, scoring="roc_auc", n_repeats=20, random_state=SEED)
imp = pd.DataFrame({"mean": pi.importances_mean, "std": pi.importances_std}, index=X.columns).sort_values("mean")
print(imp.round(4).iloc[::-1])
imp["mean"].plot.barh(xerr=imp["std"], figsize=(7, 4), title="Permutation importance (turunnya ROC-AUC)")
plt.tight_layout(); plt.savefig(out("week16_1_permutation.png")); plt.show()

# %% Fitur berkorelasi berbagi importance: tenure dan total_charges
print("korelasi tenure vs total_charges:", round(X["tenure_months"].corr(X["total_charges"]), 2))
# Mengacak tenure saja tidak terlalu merusak, karena total_charges masih menyimpan sebagian infonya.

# %% 2. Partial dependence: rata-rata prediksi kalau satu fitur diubah, fitur lain tetap
fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
PartialDependenceDisplay.from_estimator(pipe, X_val, ["tenure_months", "monthly_charges", "support_calls"], ax=ax)
plt.tight_layout(); plt.savefig(out("week16_2_pdp.png")); plt.show()

# %% 3. SHAP: kontribusi setiap fitur untuk SATU prediksi
Xt_val = pipe.named_steps["prep"].transform(X_val)
explainer = shap.TreeExplainer(pipe.named_steps["model"])
sv = explainer(Xt_val)                                # satuan: log-odds
shap.plots.beeswarm(sv, show=False, max_display=10)
plt.title("SHAP: setiap titik = satu pelanggan"); plt.tight_layout(); plt.savefig(out("week16_3_shap_beeswarm.png")); plt.show()

i = int(np.argmax(pipe.predict_proba(X_val)[:, 1]))   # pelanggan paling berisiko
print("\nPelanggan paling berisiko:\n", X_val.iloc[i])
contrib = pd.Series(sv.values[i], index=Xt_val.columns).sort_values(key=abs, ascending=False)
print(f"base value (rata-rata log-odds) = {sv.base_values[i]:.3f}")
print("kontribusi SHAP (log-odds):\n", contrib.round(3))
p_i = pipe.predict_proba(X_val.iloc[[i]])[0, 1]
print(f"base + jumlah kontribusi = {sv.base_values[i] + sv.values[i].sum():.3f}  |  logit(p) = {np.log(p_i / (1 - p_i)):.3f}  (harus sama)")

# %% [markdown]
# ## 4. Asosiasi vs penjelasan vs kausalitas — demo
# Tambahkan fitur `visited_cancel_page`: pelanggan membuka halaman "berhenti langganan" bulan ini.
# Fitur ini adalah AKIBAT dari niat churn, bukan penyebab.

# %%
rng = np.random.default_rng(SEED)
df2 = df.copy()
df2["visited_cancel_page"] = np.where(df2["churn"] == 1, rng.random(len(df2)) < 0.7, rng.random(len(df2)) < 0.05).astype(int)
X2 = df2[num + ["visited_cancel_page"] + cat]
X2_tr, X2_val = X2.loc[X_tr.index], X2.loc[X_val.index]
pipe2 = make_model(num + ["visited_cancel_page"]).fit(X2_tr, y_tr)
print(f"val ROC-AUC dengan visited_cancel_page = {roc_auc_score(y_val, pipe2.predict_proba(X2_val)[:, 1]):.4f}")
pi2 = permutation_importance(pipe2, X2_val, y_val, scoring="roc_auc", n_repeats=10, random_state=SEED)
print(pd.Series(pi2.importances_mean, index=X2.columns).sort_values(ascending=False).head(4).round(4))

# "Intervensi": blokir halaman cancel untuk semua pelanggan -> fitur = 0.
# MODEL akan memprediksi churn turun drastis...
p_before = pipe2.predict_proba(X2_val)[:, 1].mean()
p_after = pipe2.predict_proba(X2_val.assign(visited_cancel_page=0))[:, 1].mean()
print(f"Prediksi model: rata-rata peluang churn {p_before:.3f} -> {p_after:.3f} setelah halaman diblokir")
print(f"Kenyataan (dari generator): churn tetap {y_val.mean():.3f}, karena niat berhenti tidak berubah.")
# Model menjelaskan PREDIKSI-nya dengan benar (association + explanation), tapi
# memakai penjelasan itu sebagai saran tindakan (causation) akan salah total.
# Fitur seperti ini juga sering merupakan LEAKAGE: apakah tersedia SEBELUM saat kita perlu memprediksi?

# %% [markdown]
# ## 5. Eksperimen terkontrol: ubah SATU faktor, catat semuanya

# %%
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
baseline_auc = cross_val_score(make_model(), X, y, cv=cv, scoring="roc_auc")
experiments = [
    ("Hapus total_charges (redundan dengan tenure x monthly?)", dict(num_cols=[c for c in num if c != "total_charges"]), {}),
    ("Hapus support_calls (penting menurut permutation?)", dict(num_cols=[c for c in num if c != "support_calls"]), {}),
    ("max_iter 100 -> 300 (lebih banyak pohon)", {}, dict(max_iter=300)),
]
for hypothesis, cols, params in experiments:
    use_num = cols.get("num_cols", num)
    scores = cross_val_score(make_model(use_num, cat, **params), df[use_num + cat], y, cv=cv, scoring="roc_auc")
    diffs = scores - baseline_auc            # fold yang SAMA -> perbandingan berpasangan
    if (diffs > 0).all():
        conclusion = "lebih baik di semua fold"
    elif (diffs < 0).all():
        conclusion = "lebih buruk di semua fold"
    else:
        conclusion = "tidak konsisten -> tidak berbeda berarti"
    log_experiment("week16", hypothesis=hypothesis, changed=cols or params,
                   metrics={"cv_auc": round(scores.mean(), 4), "baseline_auc": round(baseline_auc.mean(), 4),
                            "fold_diffs": np.round(diffs, 4).tolist()},
                   conclusion=conclusion)
    print(f"{hypothesis:<58} AUC={scores.mean():.4f} delta={diffs.mean():+.4f} folds={np.round(diffs, 3)} -> {conclusion}")
print("log tersimpan di outputs/experiments_week16.jsonl")
