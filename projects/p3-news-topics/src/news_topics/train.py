"""Latih dan bandingkan 3 pendekatan pada AG News, lalu simpan model + hasil.

    uv run python projects/p3-news-topics/src/news_topics/train.py            # semua (+-20 menit di CPU)
    uv run python projects/p3-news-topics/src/news_topics/train.py --skip-bert

1. TF-IDF + Logistic Regression      (baseline klasik, seluruh 115k data train)
2. Sentence embedding (e5-small) + LR (8k data train, model embedding beku)
3. Fine-tune DistilBERT               (8k data train, 1 epoch)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from news_topics.data import ARTIFACTS, LABELS, SEED, load_splits  # noqa: E402

N_SMALL = 8000
BERT_NAME = "distilbert-base-uncased"
EMB_NAME = "intfloat/multilingual-e5-small"


def score(name, y_true, y_pred, fit_s, infer_ms, results, extra=None):
    results[name] = {"accuracy": round(accuracy_score(y_true, y_pred), 4),
                     "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
                     "fit_seconds": round(fit_s, 1), "infer_ms_per_text": round(infer_ms, 3), **(extra or {})}
    print(f"{name:<34} {results[name]}")


def run_tfidf(splits, results, preds):
    for name, train in [("tfidf_logreg_full", splits.train), ("tfidf_logreg_8k", splits.train.groupby("label", group_keys=False).sample(N_SMALL // 4, random_state=SEED))]:
        pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300_000),
                             LogisticRegression(max_iter=1000, C=8.0))
        t0 = time.perf_counter(); pipe.fit(train["text"], train["label"]); fit_s = time.perf_counter() - t0
        val_acc = accuracy_score(splits.val["label"], pipe.predict(splits.val["text"]))
        t0 = time.perf_counter(); p = pipe.predict(splits.test["text"]); infer = (time.perf_counter() - t0) / len(p) * 1e3
        score(name, splits.test["label"], p, fit_s, infer, results, {"val_accuracy": round(val_acc, 4), "n_train": len(train)})
        preds[name] = p
        if name == "tfidf_logreg_full":
            joblib.dump(pipe, ARTIFACTS / "tfidf_logreg.joblib")


def run_embeddings(splits, results, preds):
    from sentence_transformers import SentenceTransformer

    small = splits.train.groupby("label", group_keys=False).sample(N_SMALL // 4, random_state=SEED)
    enc = SentenceTransformer(EMB_NAME)
    t0 = time.perf_counter()
    X_tr = enc.encode(["query: " + t for t in small["text"]], batch_size=64, normalize_embeddings=True)
    clf = LogisticRegression(max_iter=2000, C=4.0).fit(X_tr, small["label"])
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    X_te = enc.encode(["query: " + t for t in splits.test["text"]], batch_size=64, normalize_embeddings=True)
    p = clf.predict(X_te); infer = (time.perf_counter() - t0) / len(p) * 1e3
    score("e5_embeddings_logreg_8k", splits.test["label"], p, fit_s, infer, results, {"n_train": len(small)})
    preds["e5_embeddings_logreg_8k"] = p


def run_bert(splits, results, preds, epochs=1, max_len=96, batch=16, lr=5e-5):
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    small = splits.train.groupby("label", group_keys=False).sample(N_SMALL // 4, random_state=SEED)
    tok = AutoTokenizer.from_pretrained(BERT_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_NAME, num_labels=4, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)}).to(device)

    def loader(df, shuffle):
        e = tok(df["text"].tolist(), truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
        data = list(zip(e["input_ids"], e["attention_mask"], torch.tensor(df["label"].to_numpy())))
        return DataLoader(data, batch_size=batch if shuffle else 128, shuffle=shuffle,
                          collate_fn=lambda b: tuple(torch.stack(t) for t in zip(*b)),
                          generator=torch.Generator().manual_seed(SEED))

    train_loader = loader(small, True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * total)) * max(0.0, (total - s) / total))
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for step, (ids, mask, y) in enumerate(train_loader, 1):
            loss = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device)).loss
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if step % 100 == 0:
                print(f"  bert epoch {epoch + 1} step {step}/{len(train_loader)} loss={loss.item():.4f}")
    fit_s = time.perf_counter() - t0

    @torch.no_grad()
    def predict(df):
        model.eval()
        out = []
        for ids, mask, _ in loader(df, False):
            out.append(model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.argmax(-1).cpu())
        return torch.cat(out).numpy()

    val_acc = accuracy_score(splits.val["label"][:1000], predict(splits.val[:1000]))
    t0 = time.perf_counter(); p = predict(splits.test); infer = (time.perf_counter() - t0) / len(p) * 1e3
    score("distilbert_finetuned_8k", splits.test["label"], p, fit_s, infer, results,
          {"val_accuracy_1k": round(val_acc, 4), "n_train": len(small), "device": str(device)})
    preds["distilbert_finetuned_8k"] = p
    model.save_pretrained(ARTIFACTS / "distilbert"); tok.save_pretrained(ARTIFACTS / "distilbert")


def main(skip_bert=False, skip_emb=False):
    ARTIFACTS.mkdir(exist_ok=True)
    splits = load_splits()
    print(f"train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}")
    results, preds = {}, {}
    run_tfidf(splits, results, preds)
    if not skip_emb:
        run_embeddings(splits, results, preds)
    if not skip_bert:
        run_bert(splits, results, preds)
    (ARTIFACTS / "results.json").write_text(json.dumps(results, indent=2))
    out = splits.test[["text", "label"]].copy()
    for k, v in preds.items():
        out[f"pred_{k}"] = v
    out.to_csv(ARTIFACTS / "test_predictions.csv", index=False)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bert", action="store_true")
    ap.add_argument("--skip-emb", action="store_true")
    a = ap.parse_args()
    main(a.skip_bert, a.skip_emb)
