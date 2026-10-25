# %% [markdown]
# # P3 — Analisis error (jalankan setelah train.py)

# %%
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

PROJECT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(PROJECT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from news_topics.data import ARTIFACTS, LABELS  # noqa: E402

results = json.loads((ARTIFACTS / "results.json").read_text())
print(pd.DataFrame(results).T.to_string())
df = pd.read_csv(ARTIFACTS / "test_predictions.csv")
models = [c.removeprefix("pred_") for c in df.columns if c.startswith("pred_")]

# %% Laporan per kelas untuk setiap model
for m in models:
    print(f"\n=== {m} ===")
    print(classification_report(df.label, df[f"pred_{m}"], target_names=LABELS, digits=3))
    print(pd.DataFrame(confusion_matrix(df.label, df[f"pred_{m}"]), index=LABELS, columns=LABELS))

# %% Contoh benar dan salah dari model terbaik
best = max(models, key=lambda m: results[m]["accuracy"])
wrong = df[df[f"pred_{best}"] != df.label]
print(f"\nModel terbaik: {best}. Salah: {len(wrong)} dari {len(df)}")
for r in wrong.sample(8, random_state=0).itertuples():
    print(f"  benar={LABELS[r.label]:<9} tebak={LABELS[getattr(r, 'pred_' + best)]:<9} | {r.text[:130]}")

# %% Kapan model sederhana dan transformer tidak setuju?
if "pred_tfidf_logreg_full" in df and len(models) > 1:
    other = [m for m in models if m != "tfidf_logreg_full"][-1]
    dis = df[df.pred_tfidf_logreg_full != df[f"pred_{other}"]]
    print(f"\nTF-IDF vs {other} tidak setuju pada {len(dis)} berita "
          f"({len(dis) / len(df):.1%}); TF-IDF benar {(dis.pred_tfidf_logreg_full == dis.label).mean():.0%}, "
          f"{other} benar {(dis[f'pred_{other}'] == dis.label).mean():.0%}")
