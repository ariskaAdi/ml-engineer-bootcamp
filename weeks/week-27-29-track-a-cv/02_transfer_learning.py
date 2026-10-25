# %% [markdown]
# # Track A, Week 28 — TorchVision dan Transfer Learning
# Data: CIFAR-10 (60.000 gambar warna 32x32, 10 kelas; unduh +-170 MB). Kita pakai subset kecil
# (5000 train) supaya bisa berjalan di CPU: ini juga situasi nyata — data berlabel sering sedikit.
# Model: ResNet18 pretrained ImageNet (unduh +-45 MB).
#
# Bandingkan: (A) CNN kecil dari nol, (B) ResNet18 beku + kepala baru, (C) fine-tune blok terakhir.
# CPU: +-15 menit total. Di Colab GPU: +-2 menit.

# %%
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from mlbootcamp.paths import DATA_DIR, MODELS_DIR
from mlbootcamp.torch_utils import set_seed

SEED, IMG = 42, 96
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# %% Transform: TRAIN boleh augmentasi; VAL/TEST tidak boleh (harus deterministik)
train_tf = transforms.Compose([
    transforms.Resize(IMG),
    transforms.RandomCrop(IMG, padding=8),
    transforms.RandomHorizontalFlip(),            # kucing dicerminkan tetap kucing (tapi angka "2" tidak!)
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # pakai statistik data yang dipakai MELATIH model pretrained
])
eval_tf = transforms.Compose([transforms.Resize(IMG), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

train_full = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tf)
train_full_eval = datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=eval_tf)
test_full = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=eval_tf)

rng = np.random.default_rng(SEED)
idx = rng.permutation(len(train_full))
train_idx, val_idx = idx[:5000], idx[5000:6000]          # val diambil dari TRAIN resmi, test tetap terpisah
test_idx = rng.permutation(len(test_full))[:2000]
# Leakage check: indeks train dan val tidak beririsan; test berasal dari file berbeda.
assert not set(train_idx) & set(val_idx)
train_set, val_set = Subset(train_full, train_idx), Subset(train_full_eval, val_idx)
test_set = Subset(test_full, test_idx)
loader = lambda ds, shuffle: DataLoader(ds, batch_size=64, shuffle=shuffle, generator=torch.Generator().manual_seed(SEED))
print(f"train={len(train_set)} val={len(val_set)} test={len(test_set)}")


@torch.no_grad()
def accuracy(model, ds):
    model.eval()
    correct = n = 0
    for x, y in loader(ds, False):
        correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item(); n += len(y)
    return correct / n


def train(model, params, epochs, lr, name):
    model.to(device)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in loader(train_set, True):
            loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  {name} epoch {epoch}: val acc={accuracy(model, val_set):.3f} ({time.perf_counter() - t0:.0f}s)")
    return time.perf_counter() - t0

results = {}

# %% (A) CNN kecil dari nol
small_cnn = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
    nn.Flatten(), nn.Linear(128, 10))
secs = train(small_cnn, small_cnn.parameters(), epochs=3, lr=2e-3, name="dari nol")
results["A. CNN kecil dari nol"] = (accuracy(small_cnn, test_set), secs, sum(p.numel() for p in small_cnn.parameters()))

# %% (B) Pretrained ResNet18, SEMUA layer dibekukan, hanya kepala baru yang dilatih
def resnet():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 10)      # ganti kepala 1000 kelas ImageNet -> 10 kelas
    return m

frozen = resnet()
for name, p in frozen.named_parameters():
    p.requires_grad = name.startswith("fc.")     # hanya fc yang belajar
n_train = sum(p.numel() for p in frozen.parameters() if p.requires_grad)
secs = train(frozen, [p for p in frozen.parameters() if p.requires_grad], epochs=3, lr=1e-3, name="beku")
results["B. ResNet18 beku + kepala baru"] = (accuracy(frozen, test_set), secs, n_train)

# %% (C) Fine-tune: buka layer4 (blok terakhir) + kepala, learning rate kecil supaya fitur pretrained tidak rusak
ft = resnet()
for name, p in ft.named_parameters():
    p.requires_grad = name.startswith(("layer4.", "fc."))
n_train = sum(p.numel() for p in ft.parameters() if p.requires_grad)
secs = train(ft, [p for p in ft.parameters() if p.requires_grad], epochs=3, lr=3e-4, name="fine-tune")
results["C. ResNet18 fine-tune layer4"] = (accuracy(ft, test_set), secs, n_train)
torch.save(ft.state_dict(), MODELS_DIR / "trackA_resnet18_cifar.pt")

# %% Ringkasan (TEST, 2000 gambar yang tidak pernah dipakai untuk memilih apa pun)
print(f"\n{'model':<34} {'test acc':>8} {'detik':>7} {'param dilatih':>14}")
for k, (acc, s, n) in results.items():
    print(f"{k:<34} {acc:8.3f} {s:7.0f} {n:14,}")
