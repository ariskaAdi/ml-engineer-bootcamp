# %% [markdown]
# # Week 29 (Track B) — Embeddings, Semantic Search, dan RAG
#
# Korpus: semua `weeks/*/NOTES.md` bootcamp ini (bahasa Indonesia) -> "asisten belajar" pribadi.
# Embedding: intfloat/multilingual-e5-small (mendukung bahasa Indonesia, unduh +-470 MB sekali).
#   Model e5 dilatih dengan awalan: "query: " untuk pertanyaan dan "passage: " untuk dokumen.
# Generator (opsional): model lokal lewat Ollama (`ollama pull qwen2.5:3b`). Kalau Ollama tidak jalan,
# script hanya menampilkan prompt yang akan dikirim.

# %%
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from mlbootcamp.paths import ROOT

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # catatan berisi karakter non-ASCII
EMB_MODEL = "intfloat/multilingual-e5-small"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"

model = SentenceTransformer(EMB_MODEL)
print("dimensi embedding:", model.get_sentence_embedding_dimension(), "| max token:", model.max_seq_length)

# %% [markdown]
# ## 1. Embedding + cosine similarity
# Embedding = vektor yang mewakili MAKNA kalimat. Kalimat bermakna mirip -> vektor berdekatan.
# cosine(a, b) = (a . b) / (|a| |b|)  -> 1 = arah sama, 0 = tidak berhubungan

# %%
sents = ["Model menghafal data latih sehingga gagal di data baru.",
         "The model overfits the training set.",
         "Harga rumah di Jakarta paling mahal.",
         "Overfitting terjadi ketika skor train jauh lebih tinggi daripada validasi."]
E = model.encode(["query: " + s for s in sents])
cos_manual = (E[0] @ E[1]) / (np.linalg.norm(E[0]) * np.linalg.norm(E[1]))
print(f"cosine manual kalimat 0-1: {cos_manual:.3f}")
En = E / np.linalg.norm(E, axis=1, keepdims=True)          # normalisasi -> cosine = dot product biasa
print(pd.DataFrame(En @ En.T, index=range(4), columns=range(4)).round(2))
# Kalimat 0 (Indonesia) dan 1 (Inggris) mirip walau TIDAK berbagi satu kata pun.

# %% [markdown]
# ## 2. Korpus: pecah NOTES.md menjadi potongan per heading (chunking)

# %%
def chunk_markdown(path: Path, max_chars: int = 500) -> list[dict]:
    """Potong per heading '## ', lalu gabungkan paragraf sampai +-max_chars.
    Chunk kecil = lebih fokus, dan muat dalam batas token model (teks yang lebih panjang DIPOTONG diam-diam)."""
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"\n(?=## )", text)
    title = parts[0].splitlines()[0].lstrip("# ").strip()
    chunks = []
    for part in parts:
        heading = part.splitlines()[0].lstrip("# ").strip()
        buf = ""
        for para in (re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", part) if p.strip()):
            if buf and len(buf) + len(para) > max_chars:
                chunks.append({"source": f"{path.parent.name}/{path.name}", "title": title, "heading": heading, "text": buf})
                buf = ""
            buf = f"{buf} {para}".strip()
        if buf:
            chunks.append({"source": f"{path.parent.name}/{path.name}", "title": title, "heading": heading, "text": buf})
    return chunks

notes = sorted((ROOT / "weeks").glob("*/NOTES.md"))
corpus = pd.DataFrame([c for p in notes for c in chunk_markdown(p)])
print(f"{len(notes)} file -> {len(corpus)} chunk")
corpus.head(3)

# %% Index: embed semua chunk sekali, simpan sebagai matriks ternormalisasi
t0 = time.perf_counter()
docs = (corpus["title"] + " | " + corpus["heading"] + " | " + corpus["text"]).tolist()
doc_emb = model.encode(["passage: " + d for d in docs], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
print(f"index {doc_emb.shape} dibuat dalam {time.perf_counter() - t0:.1f}s")

tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(docs)
doc_tfidf = tfidf.transform(docs)


def search_semantic(query, k=3):
    q = model.encode(["query: " + query], normalize_embeddings=True)[0]
    scores = doc_emb @ q
    top = np.argsort(-scores)[:k]
    return corpus.iloc[top].assign(score=scores[top])


def search_keyword(query, k=3):
    scores = (doc_tfidf @ tfidf.transform([query]).T).toarray().ravel()
    top = np.argsort(-scores)[:k]
    return corpus.iloc[top].assign(score=scores[top])

def search_hybrid(query, k=3, c=60):
    """Reciprocal Rank Fusion: gabungkan RANKING semantic dan keyword (skornya beda skala, rank tidak)."""
    q = model.encode(["query: " + query], normalize_embeddings=True)[0]
    sem = doc_emb @ q
    kw = (doc_tfidf @ tfidf.transform([query]).T).toarray().ravel()
    rank_sem, rank_kw = np.argsort(np.argsort(-sem)), np.argsort(np.argsort(-kw))
    fused = 1 / (c + rank_sem) + 1 / (c + rank_kw)
    top = np.argsort(-fused)[:k]
    return corpus.iloc[top].assign(score=fused[top])

# %% Contoh: pertanyaan memakai kata yang TIDAK ada di catatan
q = "bagaimana supaya model tidak menghafal data latihan?"
print("SEMANTIC:\n", search_semantic(q)[["source", "heading", "score"]].to_string())
print("KEYWORD (TF-IDF):\n", search_keyword(q)[["source", "heading", "score"]].to_string())

# %% [markdown]
# ## 3. Evaluasi retrieval: pertanyaan + file yang seharusnya ditemukan

# %%
eval_set = [
    ("bagaimana supaya model tidak menghafal data latihan", "week-10"),
    ("kenapa scaler tidak boleh di-fit sebelum split data", "week-09"),
    ("apa bedanya where dan having", "week-02"),
    ("arti p-value yang benar", "week-04"),
    ("kenapa gradien harus dinolkan setiap langkah training", "week-21"),
    ("GPU tidak lebih cepat untuk perkalian matriks kecil", "week-20"),
    ("memilih ambang batas klasifikasi berdasarkan biaya kesalahan", "week-08"),
    ("cluster k-means belum tentu segmen pelanggan yang bermakna", "week-14"),
    ("fitur lama telepon bocor karena baru diketahui setelah menelepon", "week-17"),
    ("menangani data fraud yang sangat sedikit", "week-15"),
]

def retrieval_metrics(search_fn, k=3):
    hits, rr = [], []
    for query, expected in eval_set:
        res = search_fn(query, k=10)["source"].tolist()
        ranks = [i for i, s in enumerate(res) if s.startswith(expected)]
        hits.append(bool(ranks) and ranks[0] < k)
        rr.append(1 / (ranks[0] + 1) if ranks else 0)
    return {"hit@3": np.mean(hits), "MRR@10": np.mean(rr)}

for name, fn in [("semantic (e5)", search_semantic), ("keyword TF-IDF", search_keyword), ("hybrid RRF", search_hybrid)]:
    print(f"{name:<15}", {k: round(float(v), 3) for k, v in retrieval_metrics(fn).items()})

# %% [markdown]
# ## 4. RAG: Retrieve -> Augment (masukkan ke prompt) -> Generate

# %%
def build_prompt(question, chunks):
    context = "\n\n".join(f"[{i + 1}] ({c.source} - {c.heading})\n{c.text}" for i, c in enumerate(chunks.itertuples()))
    return (
        "Kamu asisten belajar machine learning. Jawab HANYA berdasarkan KONTEKS di bawah, dalam bahasa Indonesia, "
        "maksimal 4 kalimat, dan sebutkan nomor sumber seperti [1]. Kalau jawabannya tidak ada di konteks, "
        "katakan: 'Tidak ditemukan di catatan.'\n\n"
        f"KONTEKS:\n{context}\n\nPERTANYAAN: {question}\nJAWABAN:"
    )


def generate(prompt: str) -> str | None:
    body = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0}}).encode()
    try:
        req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())["response"].strip()
    except OSError as e:
        print(f"(Ollama tidak tersedia: {e}. Tampilkan prompt saja.)")
        return None


def rag(question, k=3):
    chunks = search_semantic(question, k)
    prompt = build_prompt(question, chunks)
    t0 = time.perf_counter()
    answer = generate(prompt)
    print(f"\nQ: {question}\nsumber: {chunks['source'].tolist()}")
    print(f"A ({time.perf_counter() - t0:.1f}s): {answer}" if answer else prompt[:1500])
    return answer

rag("Kenapa duration dibuang dari model bank marketing?")
rag("Apa beda precision dan recall?")
rag("Berapa harga GPU NVIDIA H100?")     # tidak ada di catatan -> seharusnya menolak menjawab
