# %% [markdown]
# # Week 8 — Klasifikasi: prediksi customer churn
#
# Tujuan bisnis: cari pelanggan yang akan berhenti, lalu beri mereka penawaran retensi.
# - Biaya penawaran (dikirim ke pelanggan yang ternyata tidak churn = False Positive): Rp 50 ribu
# - Kehilangan pelanggan yang tidak terdeteksi (False Negative): Rp 600 ribu (pendapatan hilang)

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from mlbootcamp.datasets import make_churn
from mlbootcamp.paths import out

df = make_churn()
print(df.shape, "| churn rate:", round(df["churn"].mean(), 3))

# %% Persiapan sederhana (Week 9 akan merapikannya dengan Pipeline)
X = df.drop(columns=["customer_id", "churn"])
X["total_charges"] = X["total_charges"].fillna(0)        # pelanggan baru belum ditagih
X = pd.get_dummies(X, drop_first=True, dtype=float)
y = df["churn"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
# stratify=y -> proporsi churn sama di setiap split

scaler = StandardScaler().fit(X_train)           # fit HANYA di train
Xs_train, Xs_val, Xs_test = (scaler.transform(d) for d in (X_train, X_val, X_test))

# %% Melatih 4 model
models = {
    "Dummy (most_frequent)": DummyClassifier(strategy="most_frequent"),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNN (k=15)": KNeighborsClassifier(n_neighbors=15),
    "DecisionTree (depth=4)": DecisionTreeClassifier(max_depth=4, random_state=42),
}
rows = []
for name, model in models.items():
    model.fit(Xs_train, y_train)
    proba = model.predict_proba(Xs_val)[:, 1]
    pred = (proba >= 0.5).astype(int)
    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_val, pred),
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall": recall_score(y_val, pred),
        "f1": f1_score(y_val, pred),
        "roc_auc": roc_auc_score(y_val, proba),
        "pr_auc": average_precision_score(y_val, proba),
    })
print(pd.DataFrame(rows).set_index("model").round(3))
# Dummy: accuracy ~72% tanpa menangkap SATU pun churn. Accuracy menyesatkan.

# %% Confusion matrix — hitung metrik dengan tangan
logreg = models["LogisticRegression"]
proba_val = logreg.predict_proba(Xs_val)[:, 1]
pred_val = (proba_val >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_val, pred_val).ravel()
print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"accuracy  = (TP+TN)/semua = {(tp + tn) / (tp + tn + fp + fn):.3f}")
print(f"precision = TP/(TP+FP)    = {tp / (tp + fp):.3f}   <- dari yang DITEBAK churn, berapa yang benar")
print(f"recall    = TP/(TP+FN)    = {tp / (tp + fn):.3f}   <- dari yang BENAR churn, berapa yang tertangkap")
p, r = tp / (tp + fp), tp / (tp + fn)
print(f"F1        = 2PR/(P+R)     = {2 * p * r / (p + r):.3f}")
ConfusionMatrixDisplay.from_predictions(y_val, pred_val, display_labels=["stay", "churn"])
plt.title("LogReg, threshold 0.5"); plt.savefig(out("week08_1_confusion.png")); plt.show()

# %% Threshold: 0.5 bukan angka keramat
FP_COST, FN_COST = 50, 600   # ribu Rupiah
thresholds = np.linspace(0.05, 0.95, 91)
costs, precs, recs = [], [], []
for t in thresholds:
    pr = (proba_val >= t).astype(int)
    tn_, fp_, fn_, tp_ = confusion_matrix(y_val, pr).ravel()
    costs.append(fp_ * FP_COST + fn_ * FN_COST)
    precs.append(tp_ / max(tp_ + fp_, 1)); recs.append(tp_ / (tp_ + fn_))
best_t = thresholds[int(np.argmin(costs))]
print(f"Biaya @0.5 = Rp {costs[45]:,} rb | threshold terbaik = {best_t:.2f}, biaya = Rp {min(costs):,} rb")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(thresholds, precs, label="precision"); axes[0].plot(thresholds, recs, label="recall")
axes[0].axvline(best_t, color="gray", ls="--"); axes[0].set(xlabel="threshold", title="Trade-off precision vs recall"); axes[0].legend()
axes[1].plot(thresholds, costs); axes[1].axvline(best_t, color="gray", ls="--")
axes[1].set(xlabel="threshold", ylabel="total biaya (ribu Rp)", title="Biaya bisnis per threshold")
plt.tight_layout(); plt.savefig(out("week08_2_threshold.png")); plt.show()

# %% Evaluasi akhir di TEST dengan threshold yang dipilih di VALIDATION
proba_test = logreg.predict_proba(Xs_test)[:, 1]
for t in [0.5, best_t]:
    pr = (proba_test >= t).astype(int)
    tn_, fp_, fn_, tp_ = confusion_matrix(y_test, pr).ravel()
    print(f"TEST t={t:.2f}: precision={tp_ / (tp_ + fp_):.3f} recall={tp_ / (tp_ + fn_):.3f} "
          f"biaya=Rp {fp_ * FP_COST + fn_ * FN_COST:,} rb")

# %% Precision-recall curve (lebih informatif daripada ROC kalau kelas positif sedikit)
prec_c, rec_c, _ = precision_recall_curve(y_val, proba_val)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(rec_c, prec_c); ax.axhline(y_val.mean(), color="gray", ls="--", label="baseline (tebak acak)")
ax.set(xlabel="recall", ylabel="precision", title="Precision-Recall curve (LogReg)"); ax.legend()
plt.tight_layout(); plt.savefig(out("week08_3_pr_curve.png")); plt.show()

# %% Apa yang dipelajari model?
coef = pd.Series(logreg.coef_[0], index=X.columns).sort_values()
print("Koefisien LogReg (fitur sudah distandarkan; + = menaikkan peluang churn):")
print(coef.round(2))

# %% Decision tree mudah dibaca manusia
fig, ax = plt.subplots(figsize=(18, 7))
plot_tree(models["DecisionTree (depth=4)"], feature_names=list(X.columns), class_names=["stay", "churn"],
          filled=True, max_depth=2, fontsize=8, ax=ax)
plt.savefig(out("week08_4_tree.png")); plt.show()

# %% KNN: pengaruh k dan pentingnya scaling
for k in [1, 5, 15, 51]:
    knn = KNeighborsClassifier(n_neighbors=k).fit(Xs_train, y_train)
    knn_raw = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    print(f"k={k:>2}: ROC-AUC scaled={roc_auc_score(y_val, knn.predict_proba(Xs_val)[:, 1]):.3f} "
          f"| tanpa scaling={roc_auc_score(y_val, knn_raw.predict_proba(X_val)[:, 1]):.3f}")
# Tanpa scaling, jarak didominasi total_charges (ribuan) sehingga fitur lain diabaikan.
