"""Week 36 — kirim traffic ke API, lalu buat laporan monitoring dari database log prediksi.

Prasyarat: stack Docker berjalan (lihat services/docker-compose.yml).
    uv run python services/monitoring/simulate_and_monitor.py --n 300

Skenario:
  Fase A "normal" : nasabah dari periode AWAL data (seperti data training)
  Fase B "drift"  : nasabah dari periode AKHIR data (kondisi ekonomi berubah: euribor turun drastis)
  + beberapa request rusak (untuk error rate) — dan semua dihitung latensinya dari sisi klien.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "projects/p1-bank-marketing/src"))
from bank_marketing.data import load_dataset  # noqa: E402

API = os.getenv("API_URL", "http://localhost:8080")
DB_URL = os.getenv("MONITOR_DATABASE_URL", "postgresql+psycopg://bootcamp:local-dev-only@localhost:5433/predictions")
OUT = ROOT / "outputs"
RENAME = {"emp.var.rate": "emp_var_rate", "cons.price.idx": "cons_price_idx",
          "cons.conf.idx": "cons_conf_idx", "nr.employed": "nr_employed"}
NUMERIC = ["age", "campaign", "euribor3m", "nr_employed", "emp_var_rate", "cons_conf_idx"]
CATEGORICAL = ["contact", "month", "poutcome"]


def post(payload: dict) -> tuple[int, float]:
    req = urllib.request.Request(f"{API}/v1/predictions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
            status = r.status
    except urllib.error.HTTPError as e:
        status = e.code
    except OSError:
        status = 599
    return status, (time.perf_counter() - t0) * 1e3


def to_request(row: pd.Series, i: int) -> dict:
    d = row.rename(RENAME).to_dict()
    d = {k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()}
    d["client_id"] = f"SIM{i:05d}"
    return d


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population Stability Index. < 0.1 stabil, 0.1-0.2 waspada, > 0.2 drift berarti."""
    if ref.dtype == object or isinstance(ref.dtype, pd.CategoricalDtype) or ref.dtype.kind in "OUS":
        cats = sorted(set(ref) | set(cur))
        r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref)
        c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def main(n: int) -> None:
    df = load_dataset()
    cols = ["age", "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week",
            "campaign", "pdays", "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
            "euribor3m", "nr.employed"]
    cut = int(len(df) * 0.8)
    reference = df.iloc[:cut][cols].rename(columns=RENAME)          # "data training"
    phase_a = df.iloc[:cut].sample(n, random_state=1)
    phase_b = df.iloc[cut:].sample(n, random_state=2)

    # 1) Kirim traffic
    client_log = []
    for phase, rows in [("A_normal", phase_a), ("B_drift", phase_b)]:
        for i, (_, row) in enumerate(rows[cols].iterrows()):
            status, ms = post(to_request(row, i))
            client_log.append({"phase": phase, "status": status, "latency_ms": ms})
        for _ in range(max(1, n // 50)):                           # request rusak -> 422
            status, ms = post({"client_id": "BAD", "age": 5})
            client_log.append({"phase": phase, "status": status, "latency_ms": ms})
    clog = pd.DataFrame(client_log)

    # 2) Baca log prediksi dari database
    engine = create_engine(DB_URL)
    with engine.connect() as con:
        logged = pd.read_sql(text("SELECT * FROM predictions ORDER BY id DESC LIMIT :n"), con,
                             params={"n": 2 * n}).iloc[::-1].reset_index(drop=True)
    feats = pd.json_normalize(logged["features"].map(json.loads))
    logged["phase"] = np.where(np.arange(len(logged)) < n, "A_normal", "B_drift")

    # 3) Laporan
    report = {"api_latency_ms (klien)": clog.groupby("phase")["latency_ms"].quantile([0.5, 0.95, 0.99]).unstack().round(1).to_dict("index"),
              "model_latency_ms (server)": logged.groupby("phase")["latency_ms"].quantile([0.5, 0.95]).unstack().round(2).to_dict("index"),
              "error_rate": clog.assign(err=clog.status >= 400).groupby("phase")["err"].mean().round(3).to_dict(),
              "status_counts": clog.groupby(["phase", "status"]).size().unstack(fill_value=0).to_dict("index"),
              "prediction": logged.groupby("phase").agg(mean_proba=("probability", "mean"),
                                                        pct_should_call=("should_call", "mean")).round(3).to_dict("index")}
    drift = {}
    for col in NUMERIC + CATEGORICAL:
        drift[col] = {ph: round(psi(reference[col], feats.loc[logged.phase == ph, col]), 3) for ph in ["A_normal", "B_drift"]}
    report["psi_vs_training"] = drift
    print(json.dumps(report, indent=2, default=str))
    (OUT / "week36_monitoring_report.json").write_text(json.dumps(report, indent=2, default=str))

    # 4) Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    logged["probability"].rolling(25).mean().plot(ax=axes[0])
    axes[0].axvline(n, color="red", ls="--"); axes[0].set(title="Rata-rata peluang (rolling 25)", xlabel="request ke-")
    pd.DataFrame(drift).T.plot.barh(ax=axes[1]); axes[1].axvline(0.2, color="red", ls="--")
    axes[1].set(title="PSI vs data training (>0,2 = drift)")
    for ph in ["A_normal", "B_drift"]:
        axes[2].hist(feats.loc[logged.phase == ph, "euribor3m"], bins=30, alpha=0.6, label=ph)
    axes[2].hist(reference["euribor3m"].sample(n, random_state=0), bins=30, alpha=0.4, label="training")
    axes[2].set(title="euribor3m"); axes[2].legend()
    plt.tight_layout(); plt.savefig(OUT / "week36_monitoring.png")
    print(f"tersimpan: {OUT / 'week36_monitoring_report.json'}, {OUT / 'week36_monitoring.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    main(ap.parse_args().n)
