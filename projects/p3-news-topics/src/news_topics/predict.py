"""Inference topik berita.

    uv run python projects/p3-news-topics/src/news_topics/predict.py "Apple unveils new chip for laptops"
    uv run python projects/p3-news-topics/src/news_topics/predict.py --model distilbert "..."
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from news_topics.data import ARTIFACTS, LABELS, clean_text  # noqa: E402


@lru_cache(maxsize=2)
def _load(kind: str):
    if kind == "tfidf":
        return joblib.load(ARTIFACTS / "tfidf_logreg.joblib")
    from transformers import pipeline
    return pipeline("text-classification", model=str(ARTIFACTS / "distilbert"), top_k=None)


def predict(texts: list[str], kind: str = "tfidf") -> list[dict]:
    if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("setiap teks harus string yang tidak kosong")
    texts = [clean_text(t) for t in texts]
    model = _load(kind)
    if kind == "tfidf":
        probs = model.predict_proba(texts)
    else:
        out = model(texts, truncation=True, max_length=96)
        probs = np.array([[next(d["score"] for d in row if d["label"] == lab) for lab in LABELS] for row in out])
    return [{"label": LABELS[int(p.argmax())], "confidence": round(float(p.max()), 4),
             "scores": {lab: round(float(s), 4) for lab, s in zip(LABELS, p)}} for p in probs]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="+")
    ap.add_argument("--model", choices=["tfidf", "distilbert"], default="tfidf")
    a = ap.parse_args()
    for t, r in zip(a.texts, predict(a.texts, a.model)):
        print(f"{r['label']:<9} {r['confidence']:.2f}  {t}")
