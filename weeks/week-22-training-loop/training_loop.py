# %% [markdown]
# # Week 22 — Training loop lengkap: Dataset, DataLoader, train/val, simpan & muat
# Data: FashionMNIST (60.000 gambar 28x28 grayscale, 10 kelas pakaian). Unduh otomatis ±30 MB ke data/.

# %%
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from mlbootcamp.paths import DATA_DIR, MODELS_DIR

SEED = 42
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# %% [markdown]
# ## 1. Dataset: objek yang tahu (a) berapa banyak sampel, (b) cara mengambil sampel ke-i

# %%
class TabularDataset(Dataset):
    """Contoh Dataset buatan sendiri dari DataFrame — pola yang sama dipakai untuk data apa pun."""

    def __init__(self, df: pd.DataFrame, feature_cols, target_col):
        self.X = torch.tensor(df[feature_cols].to_numpy(), dtype=torch.float32)
        self.y = torch.tensor(df[target_col].to_numpy(), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


toy = TabularDataset(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "label": [0, 1, 0]}), ["a", "b"], "label")
print(len(toy), toy[1])

# %% FashionMNIST: transform mengubah gambar PIL -> tensor (1, 28, 28) bernilai 0-1, lalu dinormalisasi
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
full_train = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transform)
train_set, val_set = random_split(full_train, [50_000, 10_000], generator=torch.Generator().manual_seed(SEED))
img, label = train_set[0]
print("satu sampel:", img.shape, img.dtype, "label:", label, CLASSES[label])
print(f"train={len(train_set)} val={len(val_set)} test={len(test_set)}")

# %% ## 2. DataLoader: membagi data menjadi batch, mengacak urutan setiap epoch
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)    # acak -> tiap epoch beda urutan
val_loader = DataLoader(val_set, batch_size=512, shuffle=False)       # evaluasi tidak perlu diacak
test_loader = DataLoader(test_set, batch_size=512, shuffle=False)
xb, yb = next(iter(train_loader))
print("satu batch:", xb.shape, yb.shape, "| batch per epoch:", len(train_loader))

# %% ## 3. Model
class MLP(nn.Module):
    def __init__(self, hidden=256, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                         # (N, 1, 28, 28) -> (N, 784)
            nn.Linear(28 * 28, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


model = MLP().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# %% ## 4. Loop training dan validasi — DIPISAH
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()                                  # mengaktifkan dropout/batchnorm mode training
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)          # data ke device yang sama dengan model
        logits = model(X)                          # forward
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()                            # backward
        optimizer.step()                           # update
        total_loss += loss.item() * len(y)         # .item() -> angka Python (lepas dari graph)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()                                   # tanpa graph: hemat memori, lebih cepat
def evaluate(model, loader, loss_fn):
    model.eval()                                   # mematikan dropout, batchnorm memakai statistik tersimpan
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        total_loss += loss_fn(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


# %% ## 5. Training + catat metrik per epoch + simpan checkpoint terbaik
EPOCHS = 5
history = []
best_val_acc = 0.0
ckpt_path = MODELS_DIR / "week22_fashion_mlp.pt"
for epoch in range(1, EPOCHS + 1):
    t0 = time.perf_counter()
    tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
    va_loss, va_acc = evaluate(model, val_loader, loss_fn)
    history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
    print(f"epoch {epoch}: train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f} "
          f"({time.perf_counter() - t0:.1f}s)")
    if va_acc > best_val_acc:                      # simpan hanya kalau validasi membaik
        best_val_acc = va_acc
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "val_acc": va_acc}, ckpt_path)
pd.DataFrame(history).to_csv(MODELS_DIR / "week22_history.csv", index=False)

# %% ## 6. Muat ulang: buat arsitektur yang SAMA, lalu isi bobotnya
ckpt = torch.load(ckpt_path, map_location=device)
restored = MLP().to(device)
restored.load_state_dict(ckpt["model_state"])
print(f"checkpoint dari epoch {ckpt['epoch']} (val acc {ckpt['val_acc']:.4f})")
print("keys state_dict:", list(ckpt["model_state"].keys()))
test_loss, test_acc = evaluate(restored, test_loader, loss_fn)
print(f"TEST: loss={test_loss:.4f} acc={test_acc:.4f}")

# %% Akurasi per kelas
preds, labels = [], []
restored.eval()
with torch.no_grad():
    for X, y in test_loader:
        preds.append(restored(X.to(device)).argmax(1).cpu()); labels.append(y)
preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
per_class = {CLASSES[c]: round(float((preds[labels == c] == c).mean()), 3) for c in range(10)}
print(json.dumps(per_class, indent=1))
