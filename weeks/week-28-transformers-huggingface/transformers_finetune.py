# %% [markdown]
# # Week 28 (Track B) — Transformer dan Hugging Face
# 1. Attention dengan tangan  2. Tokenizer  3. Model pretrained siap pakai
# 4. Fine-tune DistilBERT (loop PyTorch sendiri, seperti Week 22)  5. Bandingkan dengan baseline Week 27
#
# CPU: fine-tune 4000 kalimat x 2 epoch +-10 menit. Di Colab GPU: <1 menit (bisa pakai seluruh data).

# %%
import json
import math
import time

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from mlbootcamp.paths import MODELS_DIR
from mlbootcamp.torch_utils import set_seed

SEED = 42
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, torch.get_num_threads()))

# %% [markdown]
# ## 1. Attention: setiap kata melihat kata lain dan memutuskan mana yang penting
# Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V

# %%
words = ["the", "movie", "was", "not", "good"]
torch.manual_seed(0)
d = 8
X = torch.randn(len(words), d)                     # anggap ini embedding setiap kata
W_q, W_k, W_v = (torch.randn(d, d) / math.sqrt(d) for _ in range(3))
Q, K, V = X @ W_q, X @ W_k, X @ W_v                 # query: "aku mencari apa", key: "aku berisi apa", value: isinya
scores = Q @ K.T / math.sqrt(d)                     # (5, 5): kecocokan setiap pasangan kata
weights = scores.softmax(dim=-1)                    # setiap baris jumlahnya 1
out = weights @ V                                   # representasi baru = rata-rata berbobot dari value
print("bobot attention (baris = kata yang 'melihat'):")
print("        " + " ".join(f"{w:>6}" for w in words))
for w, row in zip(words, weights):
    print(f"{w:>7} " + " ".join(f"{v:6.2f}" for v in row))
print("output shape:", out.shape, "-> setiap kata kini membawa informasi dari kata lain (konteks)")
# Bobot di sini acak (belum dilatih). Setelah training, "good" bisa belajar memperhatikan "not".

# %% [markdown]
# ## 2. Tokenizer: teks -> subword -> id

# %%
MODEL_NAME = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
enc = tok(["This movie isn't unwatchable!", "Great."], padding=True, truncation=True, max_length=16)
for ids in enc["input_ids"]:
    print(tok.convert_ids_to_tokens(ids))
print("attention_mask:", enc["attention_mask"])     # 0 = padding, diabaikan model
print("vocab size:", tok.vocab_size)
# "unwatchable" -> 'un', '##watch', '##able': kata langka dipecah jadi potongan yang dikenal -> tidak ada OOV.

# %% [markdown]
# ## 3. Model pretrained yang SUDAH di-fine-tune orang lain (Hugging Face Hub)

# %%
ds = load_dataset("stanfordnlp/sst2")
test = ds["validation"].to_pandas()                  # dipakai sebagai TEST (lihat Week 27)
clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
probe = ["this movie is not bad at all", "i expected it to be bad , but it was wonderful",
         "i expected it to be wonderful , but it was bad", "not the worst film ever", "a masterpiece of boredom"]
for s, r in zip(probe, clf(probe)):
    print(f"{r['label']:<8} {r['score']:.2f}  {s}")
t0 = time.perf_counter()
off_pred = [1 if r["label"] == "POSITIVE" else 0 for r in clf(test["sentence"].tolist(), batch_size=32)]
off_time = (time.perf_counter() - t0) / len(test) * 1e3
print(f"off-the-shelf SST-2 model: test acc={accuracy_score(test.label, off_pred):.4f} ({off_time:.1f} ms/kalimat)")

# %% [markdown]
# ## 4. Fine-tune sendiri: distilbert-base-uncased (belum pernah melihat sentimen) + kepala klasifikasi baru

# %%
N_TRAIN, EPOCHS, BATCH, MAX_LEN, LR = 4000, 2, 16, 64, 3e-5
# Supaya mirip test (kalimat utuh), ambil hanya kalimat >= 8 kata (train SST-2 banyak berisi potongan frasa)
full = ds["train"].to_pandas()
full = full[full["sentence"].str.split().str.len() >= 8]
train_df = full.sample(N_TRAIN, random_state=SEED)
val_df = full.drop(train_df.index).sample(800, random_state=SEED)

def encode(df):
    e = tok(df["sentence"].tolist(), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    return list(zip(e["input_ids"], e["attention_mask"], torch.tensor(df["label"].to_numpy())))

collate = lambda batch: tuple(torch.stack(t) for t in zip(*batch))
train_loader = DataLoader(encode(train_df), batch_size=BATCH, shuffle=True, collate_fn=collate,
                          generator=torch.Generator().manual_seed(SEED))
val_loader = DataLoader(encode(val_df), batch_size=64, collate_fn=collate)
test_loader = DataLoader(encode(test), batch_size=64, collate_fn=collate)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
print(f"parameter: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} juta")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = EPOCHS * len(train_loader)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: max(0.0, 1 - s / total_steps))  # linear decay


@torch.no_grad()
def predict(loader):
    model.eval()
    preds, labels = [], []
    for ids, mask, y in loader:
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        preds.append(logits.argmax(-1).cpu()); labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


for epoch in range(1, EPOCHS + 1):
    model.train()
    t0, running = time.perf_counter(), 0.0
    for step, (ids, mask, y) in enumerate(train_loader, 1):
        out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device))  # loss dihitung di dalam
        optimizer.zero_grad(); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # cegah gradien meledak
        optimizer.step(); scheduler.step()
        running += out.loss.item()
        if step % 50 == 0:
            print(f"  epoch {epoch} step {step}/{len(train_loader)} loss={running / step:.4f}")
    p, l = predict(val_loader)
    print(f"epoch {epoch}: train loss={running / len(train_loader):.4f} val acc={accuracy_score(l, p):.4f} "
          f"({time.perf_counter() - t0:.0f}s)")
train_minutes = (time.perf_counter() - t0) / 60 * EPOCHS

# %% 5. Test (sekali) dan perbandingan biaya
t0 = time.perf_counter()
p_test, l_test = predict(test_loader)
ft_ms = (time.perf_counter() - t0) / len(test) * 1e3
ft = {"test_acc": accuracy_score(l_test, p_test), "test_f1": f1_score(l_test, p_test), "infer_ms_per_text": ft_ms}
print(f"fine-tuned DistilBERT (4000 kalimat): test acc={ft['test_acc']:.4f} f1={ft['test_f1']:.4f} ({ft_ms:.1f} ms/kalimat)")

model.eval()
with torch.no_grad():
    e = tok(probe, padding=True, return_tensors="pt").to(device)
    probs = model(**e).logits.softmax(-1)[:, 1].cpu().numpy()
for s, pr in zip(probe, probs):
    print(f"P(pos)={pr:.2f}  {s}")

save_dir = MODELS_DIR / "week28_distilbert_sst2"
model.save_pretrained(save_dir); tok.save_pretrained(save_dir)
size_mb = sum(f.stat().st_size for f in save_dir.iterdir()) / 1e6
baseline = json.loads((MODELS_DIR / "week27_baseline_results.json").read_text()) if (MODELS_DIR / "week27_baseline_results.json").exists() else {}
summary = {
    "tfidf_bigram_logreg (week27)": baseline.get("TF-IDF uni+bigram + LogReg"),
    "distilbert_finetuned_4k": {**{k: round(v, 4) for k, v in ft.items()}, "size_mb": round(size_mb), "device": str(device)},
    "distilbert_sst2_offtheshelf": {"test_acc": round(accuracy_score(test.label, off_pred), 4), "infer_ms_per_text": round(off_time, 2)},
}
(MODELS_DIR / "week28_comparison.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
