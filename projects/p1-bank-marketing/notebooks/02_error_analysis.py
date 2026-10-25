# %% [markdown]
# # P1 — Analisis error
# Jalankan setelah `train.py`. Memakai `artifacts/test_predictions.csv`.

# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(PROJECT / "src"))
from bank_marketing.config import ARTIFACTS_DIR, SEED, TARGET  # noqa: E402
from bank_marketing.data import load_dataset  # noqa: E402
from bank_marketing.train import candidates  # noqa: E402

meta = json.loads((ARTIFACTS_DIR / "metadata.json").read_text())
pred = pd.read_csv(ARTIFACTS_DIR / "test_predictions.csv")
thr = meta["threshold"]
pred["call"] = pred["proba"] >= thr
pred["kind"] = np.select([pred.y.eq(1) & pred.call, pred.y.eq(0) & pred.call, pred.y.eq(1) & ~pred.call],
                         ["TP", "FP", "FN"], "TN")
print(pred["kind"].value_counts())

# %% 1. Performa per segmen: di mana model lemah?
def segment_report(col):
    rows = []
    for val, g in pred.groupby(col):
        if g["y"].nunique() < 2 or len(g) < 100:
            continue
        rows.append({col: val, "n": len(g), "base_rate": g.y.mean(), "pr_auc": average_precision_score(g.y, g.proba),
                     "recall": (g.call & g.y.eq(1)).sum() / g.y.sum(), "called": g.call.mean()})
    return pd.DataFrame(rows).set_index(col).round(3)

for col in ["contact", "poutcome", "month", "job"]:
    print(f"\n--- per {col} ---\n", segment_report(col))

# %% 2. Fraksi FN: siapa nasabah 'yes' yang tidak kita telepon?
fn, tp = pred[pred.kind == "FN"], pred[pred.kind == "TP"]
for col in ["contact", "poutcome", "month"]:
    print(f"\n{col}: proporsi di FN vs TP")
    print(pd.concat([fn[col].value_counts(normalize=True).rename("FN"),
                     tp[col].value_counts(normalize=True).rename("TP")], axis=1).fillna(0).round(2).head(6))
print("\nmedian numerik FN vs TP:\n", pd.concat([fn[["age", "campaign", "euribor3m", "nr.employed"]].median().rename("FN"),
                                              tp[["age", "campaign", "euribor3m", "nr.employed"]].median().rename("TP")], axis=1))

# %% 3. Uji drift waktu: data UCI terurut Mei 2008 - Nov 2010.
# Latih di 80% awal, uji di 20% akhir (kondisi ekonomi berbeda!)
df = load_dataset()
cut = int(len(df) * 0.8)
early, late = df.iloc[:cut], df.iloc[cut:]
m = candidates()[meta["model_name"]].fit(early.drop(columns=[TARGET]), early[TARGET])
p_late = m.predict_proba(late.drop(columns=[TARGET]))[:, 1]
print(f"\nSplit waktu: base rate awal={early[TARGET].mean():.3f}, akhir={late[TARGET].mean():.3f}")
print(f"PR-AUC acak (test)={meta['test']['pr_auc']:.3f} | PR-AUC 20% data TERAKHIR={average_precision_score(late[TARGET], p_late):.3f} "
      f"| ROC-AUC terakhir={roc_auc_score(late[TARGET], p_late):.3f}")
print(f"rata-rata prediksi di periode akhir={p_late.mean():.3f} vs kenyataan={late[TARGET].mean():.3f} -> kalibrasi bergeser")
