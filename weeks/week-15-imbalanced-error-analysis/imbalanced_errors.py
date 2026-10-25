# %% [markdown]
# # Week 15 — Data Tidak Seimbang dan Analisis Error
#
# Fraud hanya ~2% transaksi. Tujuan: tangkap fraud sebanyak mungkin, tapi tim investigasi
# hanya sanggup memeriksa transaksi yang ditandai kalau precision-nya >= 30%.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlbootcamp.datasets import make_fraud
from mlbootcamp.paths import out

SEED = 0
df = make_fraud()
print(df["is_fraud"].value_counts(), f"\nfraud rate = {df['is_fraud'].mean():.2%}")
features = ["amount", "hour", "km_from_home", "new_merchant", "tx_last_24h", "online"]
X, y = df[features], df["is_fraud"]
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
X_tr, X_val, y_tr, y_val = train_test_split(X_dev, y_dev, test_size=0.33, random_state=SEED, stratify=y_dev)


def recall_at_precision(y_true, proba, min_precision=0.30):
    """Recall terbaik yang bisa dicapai dengan precision >= min_precision (+ threshold-nya)."""
    p, r, t = precision_recall_curve(y_true, proba)
    ok = p[:-1] >= min_precision
    if not ok.any():
        return 0.0, None
    i = np.argmax(np.where(ok, r[:-1], -1))
    return r[i], t[i]


def evaluate(name, proba, y_true=y_val):
    rec, thr = recall_at_precision(y_true, proba)
    pred05 = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred05).ravel()
    print(f"{name:<34} PR-AUC={average_precision_score(y_true, proba):.3f}  "
          f"recall@P>=30%={rec:.3f}  | @0.5: TP={tp} FP={fp} FN={fn}")
    return {"model": name, "pr_auc": average_precision_score(y_true, proba), "recall_at_p30": rec, "thr": thr}

# %% 1. Akurasi menipu
print(f"Model 'semua normal' accuracy = {1 - y_val.mean():.2%} dan menangkap 0 fraud.\n")

# %% 2. Bandingkan cara menangani ketidakseimbangan (satu faktor diubah setiap kali)
lr = lambda **kw: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **kw))
rows = []
m_base = lr().fit(X_tr, y_tr); rows.append(evaluate("LogReg", m_base.predict_proba(X_val)[:, 1]))
m_bal = lr(class_weight="balanced").fit(X_tr, y_tr); rows.append(evaluate("LogReg class_weight=balanced", m_bal.predict_proba(X_val)[:, 1]))

# Random oversampling: duplikasi fraud di TRAIN saja (jangan pernah di val/test!)
rng = np.random.default_rng(SEED)
pos = np.flatnonzero(y_tr.to_numpy() == 1)
extra = rng.choice(pos, size=len(y_tr) - 2 * len(pos), replace=True)
X_over = pd.concat([X_tr, X_tr.iloc[extra]]); y_over = pd.concat([y_tr, y_tr.iloc[extra]])
m_over = lr().fit(X_over, y_over); rows.append(evaluate("LogReg + oversampling 50:50", m_over.predict_proba(X_val)[:, 1]))

# Random undersampling: buang sebagian besar transaksi normal di TRAIN
neg = np.flatnonzero(y_tr.to_numpy() == 0)
keep = np.concatenate([pos, rng.choice(neg, size=len(pos) * 5, replace=False)])
m_under = lr().fit(X_tr.iloc[keep], y_tr.iloc[keep]); rows.append(evaluate("LogReg + undersampling 1:5", m_under.predict_proba(X_val)[:, 1]))

hgb = HistGradientBoostingClassifier(random_state=SEED).fit(X_tr, y_tr)
rows.append(evaluate("HistGradientBoosting", hgb.predict_proba(X_val)[:, 1]))
print("\n", pd.DataFrame(rows).set_index("model").round(3))
# Pengamatan: class_weight/resampling hampir tidak mengubah PR-AUC (ranking sama),
# tapi MENGGESER peluang -> pada threshold 0.5 jauh lebih banyak TP DAN FP.

# %% 3. Resampling sebelum split = LEAKAGE
X_over_all = pd.concat([X_dev, X_dev[y_dev == 1].sample(3000, replace=True, random_state=SEED)])
y_over_all = y_dev.loc[X_over_all.index]
a, b, ya, yb = train_test_split(X_over_all, y_over_all, test_size=0.3, random_state=SEED, stratify=y_over_all)
leaky = HistGradientBoostingClassifier(random_state=SEED).fit(a, ya)
print(f"\nOversampling SEBELUM split -> PR-AUC 'validasi' = {average_precision_score(yb, leaky.predict_proba(b)[:, 1]):.3f} (palsu!)")
print("Penyebab: salinan fraud yang sama ada di train DAN validasi, jadi model cukup menghafal.")

# %% 4. Precision-recall trade-off
fig, ax = plt.subplots(figsize=(7, 4))
for name, model in [("LogReg", m_base), ("HGB", hgb)]:
    p, r, _ = precision_recall_curve(y_val, model.predict_proba(X_val)[:, 1])
    ax.plot(r, p, label=name)
ax.axhline(0.30, color="gray", ls="--", label="batas precision 30%")
ax.axhline(y_val.mean(), color="black", ls=":", label="baseline acak")
ax.set(xlabel="recall", ylabel="precision", title="Precision-recall: fraud 2%"); ax.legend()
plt.tight_layout(); plt.savefig(out("week15_1_pr.png")); plt.show()

# %% 5. Analisis error: lihat FN dan FP model LogReg pada threshold operasional
_, thr = recall_at_precision(y_val, m_base.predict_proba(X_val)[:, 1])
val = X_val.copy(); val["y"] = y_val; val["proba"] = m_base.predict_proba(X_val)[:, 1]
val["pred"] = (val["proba"] >= thr).astype(int)
val["kind"] = np.select([(val.y == 1) & (val.pred == 1), (val.y == 0) & (val.pred == 1), (val.y == 1) & (val.pred == 0)],
                        ["TP", "FP", "FN"], "TN")
print(f"\nthreshold operasional LogReg = {thr:.3f}")
print(val.groupby("kind")[features].median().round(1))

# Fraud yang lolos (FN) jam berapa?
val["night"] = (val["hour"] <= 4) | (val["hour"] >= 23)
print("\nProporsi transaksi malam (23:00-04:59):")
print(val.groupby("kind")["night"].mean().round(2))
print("\nRecall fraud malam vs siang:")
fr = val[val.y == 1]
print(fr.groupby("night")["pred"].mean().round(3))

# %% [markdown]
# **Temuan:** 33% fraud yang LOLOS (FN) terjadi malam hari, padahal hanya 4% transaksi normal (TN)
# yang terjadi malam. Artinya "malam" adalah sinyal kuat yang belum dimanfaatkan penuh.
# **Hipotesis:** LogReg melihat `hour` sebagai angka lurus 0..23, sehingga jam 23 dan jam 2 dianggap
# berjauhan. Model linear hanya bisa bilang "makin besar jam, makin (tidak) fraud" — tidak bisa "malam".
#
# **Eksperimen terkontrol:** ubah SATU hal — tambahkan fitur `is_night`. Model, data, split, seed sama.

# %% 6. Eksperimen terkontrol
add_night = lambda d: d.assign(is_night=((d["hour"] <= 4) | (d["hour"] >= 23)).astype(int))
m_night = lr().fit(add_night(X_tr), y_tr)
before = evaluate("LogReg (sebelum)", m_base.predict_proba(X_val)[:, 1])
after = evaluate("LogReg + is_night (sesudah)", m_night.predict_proba(add_night(X_val))[:, 1])

# Cek kestabilan dengan 5-fold CV (bukan hanya satu split)
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
diffs = []
for tr_i, va_i in skf.split(X_dev, y_dev):
    a_, b_ = X_dev.iloc[tr_i], X_dev.iloc[va_i]
    ya_, yb_ = y_dev.iloc[tr_i], y_dev.iloc[va_i]
    s0 = average_precision_score(yb_, lr().fit(a_, ya_).predict_proba(b_)[:, 1])
    s1 = average_precision_score(yb_, lr().fit(add_night(a_), ya_).predict_proba(add_night(b_))[:, 1])
    diffs.append(s1 - s0)
print(f"Selisih PR-AUC per fold: {np.round(diffs, 3)} -> konsisten positif di {sum(d > 0 for d in diffs)}/5 fold")

# %% 7. Test set: sekali, untuk model akhir
_, thr_night = recall_at_precision(y_val, m_night.predict_proba(add_night(X_val))[:, 1])  # threshold dipilih di VALIDATION
final = lr().fit(add_night(X_dev), y_dev)
proba_test = final.predict_proba(add_night(X_test))[:, 1]
rec_t, _ = recall_at_precision(y_test, proba_test)
pred_t = (proba_test >= thr_night).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred_t).ravel()
print(f"\nTEST: PR-AUC={average_precision_score(y_test, proba_test):.3f}  "
      f"pada threshold val ({thr_night:.3f}): precision={tp / max(tp + fp, 1):.3f} recall={tp / (tp + fn):.3f}")
