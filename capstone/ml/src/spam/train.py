"""Capstone — latih, bandingkan, pilih, evaluasi, simpan.

    uv run python capstone/ml/src/spam/train.py
    uv run python capstone/ml/src/spam/train.py --with-embeddings     # + kandidat e5 (lambat di CPU)

Aturan pemilihan (ditulis SEBELUM eksperimen, lihat docs/01_problem_and_plan.md):
  - Pemilihan model: PR-AUC rata-rata 5-fold CV di data dev (stabil, tidak bergantung threshold).
    Kalau selisih < 1 std dari yang terbaik, pilih yang lebih sederhana/cepat.
  - Titik operasi: threshold dengan recall spam maksimum pada precision >= 98% (pesan normal yang
    ditahan sangat merugikan pengguna), dipilih dari prediksi out-of-fold data dev.
  - Test dipakai sekali di akhir.
"""

from __future__ import annotations

import argparse
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spam.data import ARTIFACTS, RAW, SEED, data_hash, dedupe, load, split  # noqa: E402

TARGET_PRECISION = 0.98


def candidates() -> dict:
    word = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    return {
        "dummy": make_pipeline(word(), DummyClassifier(strategy="prior")),
        "nb_word": make_pipeline(word(), ComplementNB(alpha=0.3)),
        "logreg_word": make_pipeline(word(), LogisticRegression(C=10, max_iter=2000, class_weight="balanced")),
        "logreg_word_char": make_pipeline(FeatureUnion([("w", word()), ("c", char())]),
                                          LogisticRegression(C=10, max_iter=3000, class_weight="balanced")),
        "svm_word_char": make_pipeline(FeatureUnion([("w", word()), ("c", char())]),
                                       CalibratedClassifierCV(LinearSVC(C=0.5), cv=3)),
    }


SIMPLICITY = ["nb_word", "logreg_word", "logreg_word_char", "svm_word_char", "e5_logreg"]


def recall_at_precision(y, p, target=TARGET_PRECISION) -> tuple[float, float]:
    prec, rec, thr = precision_recall_curve(y, p)
    ok = prec[:-1] >= target
    if not ok.any():
        return 0.0, 1.0
    i = np.argmax(np.where(ok, rec[:-1], -1))
    return float(rec[i]), float(thr[i])


def main(with_embeddings: bool = False) -> dict:
    t_start = time.perf_counter()
    raw = load()
    df = dedupe(raw)
    dev, test = split(df)
    print(f"raw={len(raw)} dedup={len(df)} (dibuang {len(raw) - len(df)}) dev={len(dev)} test={len(test)} spam={df.y.mean():.3f}")

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    models = candidates()
    if with_embeddings:
        from spam.embed import E5Vectorizer
        models["e5_logreg"] = make_pipeline(E5Vectorizer(), LogisticRegression(C=4, max_iter=2000, class_weight="balanced"))

    rows, oof_store = [], {}
    for name, pipe in models.items():
        t0 = time.perf_counter()
        fold_scores, fold_recalls = [], []
        oof = np.zeros(len(dev))
        for tr, va in cv.split(dev["text"], dev["y"]):
            m = clone(pipe).fit(dev["text"].iloc[tr], dev["y"].iloc[tr])
            p = m.predict_proba(dev["text"].iloc[va])[:, 1]
            oof[va] = p
            fold_scores.append(average_precision_score(dev["y"].iloc[va], p))
            fold_recalls.append(recall_at_precision(dev["y"].iloc[va], p)[0])
        oof_store[name] = oof
        rows.append({"model": name, "pr_auc": np.mean(fold_scores), "pr_auc_std": np.std(fold_scores),
                     "recall_at_p98": np.mean(fold_recalls), "recall_at_p98_std": np.std(fold_recalls),
                     "cv_seconds": time.perf_counter() - t0})
        print(f"  {name:<18} PR-AUC={rows[-1]['pr_auc']:.4f}+-{rows[-1]['pr_auc_std']:.4f}  "
              f"recall@P98={rows[-1]['recall_at_p98']:.3f}+-{rows[-1]['recall_at_p98_std']:.3f}  ({rows[-1]['cv_seconds']:.0f}s)")
    table = pd.DataFrame(rows).set_index("model")

    best = table["pr_auc"].idxmax()
    cutoff = table.loc[best, "pr_auc"] - table.loc[best, "pr_auc_std"]
    chosen = next(m for m in SIMPLICITY if m in table.index and table.loc[m, "pr_auc"] >= cutoff)
    # Threshold dari prediksi out-of-fold di dev (BUKAN test)
    _, threshold = recall_at_precision(dev["y"], oof_store[chosen])
    print(f"terbaik={best} dipilih={chosen} threshold={threshold:.3f}")

    final = clone(models[chosen]).fit(dev["text"], dev["y"])
    t0 = time.perf_counter()
    p_test = final.predict_proba(test["text"])[:, 1]
    infer_ms = (time.perf_counter() - t0) / len(test) * 1e3
    pred = (p_test >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test["y"], pred).ravel()
    test_metrics = {"pr_auc": round(float(average_precision_score(test["y"], p_test)), 4),
                    "precision": round(float(tp / max(tp + fp, 1)), 4), "recall": round(float(tp / (tp + fn)), 4),
                    "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                    "infer_ms_per_message": round(infer_ms, 3)}
    print("TEST:", test_metrics)

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(final, ARTIFACTS / "model.joblib")
    meta = {"model_name": chosen, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "threshold": threshold, "target_precision": TARGET_PRECISION, "seed": SEED,
            "data_sha256": data_hash(RAW), "n_raw": len(raw), "n_dedup": len(df),
            "cv": table.round(4).reset_index().to_dict(orient="records"), "test": test_metrics,
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__},
            "train_seconds": round(time.perf_counter() - t_start, 1)}
    (ARTIFACTS / "metadata.json").write_text(json.dumps(meta, indent=2))
    test.assign(proba=p_test, pred=pred).to_csv(ARTIFACTS / "test_predictions.csv", index=False)

    # Bukti leakage duplikat: evaluasi model yang sama kalau duplikat TIDAK dibuang
    dev_raw, test_raw = split(raw)
    leak = clone(models[chosen]).fit(dev_raw["text"], dev_raw["y"])
    leaked_in_test = test_raw["text"].isin(dev_raw["text"]).mean()
    meta["duplicate_leakage_check"] = {
        "test_rows_also_in_train": round(float(leaked_in_test), 4),
        "pr_auc_without_dedup": round(float(average_precision_score(test_raw["y"], leak.predict_proba(test_raw["text"])[:, 1])), 4)}
    (ARTIFACTS / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("tanpa dedup:", meta["duplicate_leakage_check"])
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-embeddings", action="store_true")
    main(ap.parse_args().with_embeddings)
