"""Latih dan bandingkan model, pilih dengan CV, evaluasi sekali di test, simpan artefak.

Jalankan dari root repo:
    uv run python projects/p1-bank-marketing/src/bank_marketing/train.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split

if __package__ in (None, ""):  # dijalankan sebagai script -> tambahkan src/ ke path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bank_marketing.config import ARTIFACTS_DIR, RAW_CSV, SEED, TARGET  # noqa: E402
from bank_marketing.data import load_dataset  # noqa: E402
from bank_marketing.features import build_pipeline  # noqa: E402

TOP_FRACTION = 0.20  # bisnis: tim call center sanggup menelepon 20% nasabah dengan skor tertinggi


def candidates() -> dict:
    return {
        "dummy": build_pipeline(DummyClassifier(strategy="prior")),
        "logreg": build_pipeline(LogisticRegression(max_iter=2000, C=0.5)),
        "random_forest": build_pipeline(
            RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=SEED), scale=False),
        "hist_gb": build_pipeline(
            HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=4, min_samples_leaf=50,
                                           l2_regularization=1.0, random_state=SEED), scale=False),
    }


def capture_at_top(y_true, proba, fraction: float = TOP_FRACTION) -> float:
    """Proporsi semua 'yes' yang tertangkap kalau kita hanya menelepon `fraction` teratas."""
    y_true = np.asarray(y_true)
    k = int(len(proba) * fraction)
    top = np.argsort(-proba)[:k]
    return y_true[top].sum() / y_true.sum()


def main() -> dict:
    t0 = time.perf_counter()
    df = load_dataset()
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    print(f"data: {df.shape}, positive rate={y.mean():.3f}")

    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    # 1) Bandingkan kandidat dengan CV yang sama
    cv_rows = []
    for name, pipe in candidates().items():
        r = cross_validate(pipe, X_dev, y_dev, cv=cv, scoring=["average_precision", "roc_auc"], n_jobs=1)
        row = {"model": name,
               "pr_auc": r["test_average_precision"].mean(), "pr_auc_std": r["test_average_precision"].std(),
               "roc_auc": r["test_roc_auc"].mean(), "fit_seconds": r["fit_time"].mean()}
        cv_rows.append(row)
        print(f"  {name:<14} PR-AUC={row['pr_auc']:.4f} +- {row['pr_auc_std']:.4f}  ROC-AUC={row['roc_auc']:.4f}")
    cv_table = pd.DataFrame(cv_rows).set_index("model")

    # 2) Pilih: PR-AUC tertinggi; kalau model lebih sederhana masih dalam 1 std, pilih yang sederhana
    best = cv_table["pr_auc"].idxmax()
    cutoff = cv_table.loc[best, "pr_auc"] - cv_table.loc[best, "pr_auc_std"]
    chosen = next(m for m in ["logreg", "random_forest", "hist_gb"] if cv_table.loc[m, "pr_auc"] >= cutoff)
    print(f"terbaik CV: {best} | dipilih: {chosen}")

    # 3) Threshold dari prediksi out-of-fold di dev (bukan test): skor batas 20% teratas
    pipe = candidates()[chosen]
    oof = cross_val_predict(clone(pipe), X_dev, y_dev, cv=cv, method="predict_proba")[:, 1]
    threshold = float(np.quantile(oof, 1 - TOP_FRACTION))

    # 4) Latih ulang di seluruh dev, evaluasi SEKALI di test
    model = pipe.fit(X_dev, y_dev)
    proba = model.predict_proba(X_test)[:, 1]
    pred = proba >= threshold
    tp = int((pred & (y_test == 1)).sum())
    test = {
        "pr_auc": average_precision_score(y_test, proba),
        "roc_auc": roc_auc_score(y_test, proba),
        "capture_at_top20": float(capture_at_top(y_test, proba)),
        "precision_at_threshold": tp / max(int(pred.sum()), 1),
        "recall_at_threshold": tp / int(y_test.sum()),
        "base_rate": float(y_test.mean()),
    }
    print("TEST:", {k: round(v, 4) for k, v in test.items()})

    # 5) Simpan artefak + metadata untuk reproduksi
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")
    metadata = {
        "model_name": chosen,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": threshold,
        "top_fraction": TOP_FRACTION,
        "seed": SEED,
        "data_file": RAW_CSV.name,
        "data_sha256": hashlib.sha256(RAW_CSV.read_bytes()).hexdigest(),
        "n_rows": len(df),
        "cv": cv_table.round(4).reset_index().to_dict(orient="records"),
        "test": {k: round(v, 4) for k, v in test.items()},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__},
        "train_seconds": round(time.perf_counter() - t0, 1),
    }
    (ARTIFACTS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    X_test.assign(y=y_test, proba=proba).to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False)
    print(f"artefak tersimpan di {ARTIFACTS_DIR}")
    return metadata


if __name__ == "__main__":
    main()
