# %% [markdown]
# # Week 27 (Track B) — Data Teks dan Baseline NLP Klasik
# Data: SST-2 (kalimat review film, label 0=negatif 1=positif). Diunduh otomatis dari Hugging Face.
# Label test resmi SST-2 disembunyikan, jadi split "validation" resmi (872 kalimat) kita pakai sebagai TEST.

# %%
import json
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from mlbootcamp.paths import MODELS_DIR

SEED = 42
ds = load_dataset("stanfordnlp/sst2")
train_full = ds["train"].to_pandas()
test = ds["validation"].to_pandas()
train, val = train_test_split(train_full, test_size=5000, random_state=SEED, stratify=train_full["label"])
print(f"train={len(train)} val={len(val)} test={len(test)} | positif={train.label.mean():.2f}")
print(train.sample(5, random_state=1)[["sentence", "label"]].to_string())

# %% [markdown]
# ## 1. Dari teks ke angka: cleaning -> tokenisasi -> vocabulary -> vektor

# %%
def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # buang tag HTML
    text = re.sub(r"http\S+", " URL ", text)       # ganti URL dengan token khusus
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+|[!?]", clean(text))

s = "This movie is NOT good... <br/> See http://x.com !!"
print(clean(s)); print(tokenize(s))

# Vocabulary = daftar token unik yang dikenal model
counts = Counter(tok for sent in train["sentence"] for tok in tokenize(sent))
print(f"vocabulary: {len(counts):,} token unik | 10 terbanyak: {counts.most_common(10)}")
print(f"token yang muncul sekali saja: {sum(c == 1 for c in counts.values()):,}")

# %% Bag-of-words: setiap kalimat -> vektor hitungan kata (urutan kata HILANG)
bow = CountVectorizer(tokenizer=tokenize, token_pattern=None)
X_demo = bow.fit_transform(["the film is good", "the film is not good", "good , not the film"])
print(pd.DataFrame(X_demo.toarray(), columns=bow.get_feature_names_out()))
# Kalimat 2 dan 3 punya vektor SAMA PERSIS, padahal artinya berbeda -> word order penting!

# %% TF-IDF: kata yang muncul di SEMUA dokumen ("the") diberi bobot kecil, kata khas diberi bobot besar
tfidf = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
tfidf.fit(train["sentence"])
idf = pd.Series(tfidf.idf_, index=tfidf.get_feature_names_out())
print("IDF terendah (kata umum):", idf.nsmallest(5).round(2).to_dict())
print("IDF contoh kata khas:", idf[["masterpiece", "boring", "the"]].round(2).to_dict())

# %% [markdown]
# ## 2. Baseline: TF-IDF + Logistic Regression

# %%
results = {}

def evaluate(name, pipe):
    t0 = time.perf_counter()
    pipe.fit(train["sentence"], train["label"])
    fit_s = time.perf_counter() - t0
    val_acc = accuracy_score(val["label"], pipe.predict(val["sentence"]))
    t0 = time.perf_counter()
    pred_test = pipe.predict(test["sentence"])
    infer_ms = (time.perf_counter() - t0) / len(test) * 1e3
    results[name] = {"val_acc": val_acc, "test_acc": accuracy_score(test["label"], pred_test),
                     "test_f1": f1_score(test["label"], pred_test), "fit_s": fit_s, "infer_ms_per_text": infer_ms}
    print(f"{name:<34} val acc={val_acc:.4f}  fit={fit_s:.1f}s")
    return pipe

lr = lambda: LogisticRegression(max_iter=2000, C=4.0)
evaluate("BoW unigram + LogReg", make_pipeline(CountVectorizer(tokenizer=tokenize, token_pattern=None), lr()))
evaluate("TF-IDF unigram + LogReg", make_pipeline(TfidfVectorizer(tokenizer=tokenize, token_pattern=None, sublinear_tf=True), lr()))
best = evaluate("TF-IDF uni+bigram + LogReg", make_pipeline(
    TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2), min_df=2, sublinear_tf=True), lr()))
# bigram: "not good" menjadi SATU fitur -> sebagian informasi urutan kembali

# %% Test (sekali) + laporan
print(pd.DataFrame(results).T.round(4).to_string())
# Kenapa val (0.92) jauh di atas test (0.82)? Train SST-2 berisi POTONGAN frasa pendek, test berisi kalimat utuh.
print("panjang kata median - train:", train.sentence.str.split().str.len().median(), "| test:", test.sentence.str.split().str.len().median())
print(classification_report(test["label"], best.predict(test["sentence"]), target_names=["neg", "pos"], digits=3))

# %% Apa yang dipelajari? Bobot terbesar/terkecil
vec, clf = best.named_steps["tfidfvectorizer"], best.named_steps["logisticregression"]
coef = pd.Series(clf.coef_[0], index=vec.get_feature_names_out())
print("paling POSITIF:", coef.nlargest(12).index.tolist())
print("paling NEGATIF:", coef.nsmallest(12).index.tolist())
print("bigram negasi:", coef[[c for c in ["not good", "not bad", "never boring", "not funny"] if c in coef.index]].round(2).to_dict())

# %% [markdown]
# ## 3. Kenapa urutan dan konteks penting — kasus yang membuat baseline gagal

# %%
probe = ["this movie is good", "this movie is not good", "this movie is not bad at all",
         "i expected it to be bad , but it was wonderful", "i expected it to be wonderful , but it was bad",
         "not the worst film ever", "a masterpiece of boredom"]
for s, p in zip(probe, best.predict_proba(probe)[:, 1]):
    print(f"P(pos)={p:.2f}  {s}")
# Dua kalimat "expected ... but ..." berisi kata yang SAMA -> BoW/TF-IDF hampir tidak bisa membedakan.

test_err = test.assign(pred=best.predict(test["sentence"]))
wrong = test_err[test_err.pred != test_err.label]
neg_words = r"\b(not|n't|no|never|nothing|but|though|although)\b"
has_neg = test_err["sentence"].str.contains(neg_words)
print(f"\nError rate kalimat dengan negasi/kontras: {(test_err[has_neg].pred != test_err[has_neg].label).mean():.3f} "
      f"vs tanpa: {(test_err[~has_neg].pred != test_err[~has_neg].label).mean():.3f}")
print(wrong.sample(6, random_state=0)[["sentence", "label"]].to_string())

# %% Simpan hasil baseline untuk dibandingkan di Week 28
(MODELS_DIR / "week27_baseline_results.json").write_text(json.dumps(
    {k: {m: round(v, 4) for m, v in d.items()} for k, d in results.items()}, indent=2))
