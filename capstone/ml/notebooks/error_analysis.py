# %% [markdown]
# # Capstone — Analisis error (jalankan setelah train.py)

# %%
import json
import sys
from pathlib import Path

import pandas as pd

ML = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
meta = json.loads((ML / "artifacts/metadata.json").read_text())
df = pd.read_csv(ML / "artifacts/test_predictions.csv")
thr = meta["threshold"]
print(f"model={meta['model_name']} threshold={thr:.3f}")

# %% False positive: pesan normal yang akan DITAHAN (paling merugikan pengguna)
fp = df[(df.y == 0) & (df.pred == 1)].sort_values("proba", ascending=False)
print(f"\nFALSE POSITIVE ({len(fp)}):")
for r in fp.itertuples():
    print(f"  p={r.proba:.2f}  {r.text[:160]}")

# %% False negative: spam yang lolos
fn = df[(df.y == 1) & (df.pred == 0)].sort_values("proba")
print(f"\nFALSE NEGATIVE ({len(fn)}):")
for r in fn.itertuples():
    print(f"  p={r.proba:.2f}  {r.text[:160]}")

# %% Pesan di zona abu-abu (dekat threshold) -> kandidat "review" alih-alih keputusan otomatis
grey = df[(df.proba > thr * 0.5) & (df.proba < min(0.95, thr * 2))]
print(f"\nZona abu-abu ({thr * 0.5:.2f}-{min(0.95, thr * 2):.2f}): {len(grey)} pesan, spam={grey.y.mean():.2f}")

# %% Error per panjang pesan
df["len_bucket"] = pd.cut(df.text.str.len(), [0, 40, 80, 160, 1000])
print(df.assign(err=df.y != df.pred).groupby("len_bucket", observed=True)["err"].agg(["mean", "size"]).round(3))
