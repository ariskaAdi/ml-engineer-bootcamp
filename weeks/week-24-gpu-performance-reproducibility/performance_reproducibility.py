# %% [markdown]
# # Week 24 — Performa GPU dan Reproducibility
#
# Jalankan di laptop (CPU) DAN di Google Colab (T4 GPU), lalu bandingkan.
# Di Colab: upload folder src/mlbootcamp atau salin fungsi yang dipakai, lalu `pip install -e .`
# (atau cukup jalankan file ini setelah `!pip install torchvision`).

# %%
import json
import platform
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mlbootcamp.paths import DATA_DIR, MODELS_DIR
from mlbootcamp.torch_utils import FASHION_MEAN, FASHION_STD, load_fashion_tensors, make_loader, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "| threads CPU:", torch.get_num_threads())


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, repeats=5, warmup=1):
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); sync()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(np.std(times))

# %% [markdown]
# ## 1. Cara mengukur yang SALAH vs BENAR

# %%
A = torch.randn(2048, 2048, device=device); B = torch.randn(2048, 2048, device=device)
t0 = time.perf_counter(); A @ B; naive = time.perf_counter() - t0          # salah: tanpa warm-up, tanpa sync
med, sd = timeit(lambda: A @ B)
print(f"naif (sekali, tanpa warm-up/sync): {naive * 1e3:.2f} ms | benar: median {med * 1e3:.2f} ms +- {sd * 1e3:.2f}")
# Di GPU, versi naif bisa tampak 0.05 ms karena timer berhenti SEBELUM GPU selesai menghitung.

# %% ## 2. Satu langkah training CNN: CPU vs GPU, ukuran batch berbeda
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


results = []
for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
    for bs in [8, 128, 1024]:
        set_seed(0)
        model = SmallCNN().to(dev)
        opt = torch.optim.Adam(model.parameters())
        x = torch.randn(bs, 1, 28, 28, device=dev); y = torch.randint(0, 10, (bs,), device=dev)

        def step():
            opt.zero_grad(); nn.functional.cross_entropy(model(x), y).backward(); opt.step()

        med, _ = timeit(step, repeats=5)
        results.append({"device": dev, "batch": bs, "ms_per_step": med * 1e3, "images_per_s": bs / med})
        print(f"{dev:<5} batch={bs:>5}: {med * 1e3:8.2f} ms/step  -> {bs / med:10,.0f} gambar/detik")

# %% [markdown]
# ## 3. Bottleneck data loading
# Model yang cepat tidak berguna kalau data datang lambat.

# %%
from torchvision import datasets, transforms

pil_ds = datasets.FashionMNIST(DATA_DIR, train=True, download=True,
                               transform=transforms.Compose([transforms.ToTensor(),
                                                             transforms.Normalize((FASHION_MEAN,), (FASHION_STD,))]))
X_mem, y_mem = load_fashion_tensors(train=True)
N_BATCHES = 100


def time_loader(loader, warmup=5):
    it = iter(loader)
    for _ in range(warmup):                 # lewati batch awal (start worker, cache dingin)
        next(it)
    t0 = time.perf_counter()
    for _ in range(N_BATCHES):
        xb, yb = next(it)
        xb = xb.to(device, non_blocking=True)
    sync()
    return (time.perf_counter() - t0) / N_BATCHES * 1e3


print(f"PIL + transform per gambar, num_workers=0 : {time_loader(DataLoader(pil_ds, batch_size=128, shuffle=True)):6.2f} ms/batch")
# num_workers>0 menjalankan loading di proses lain. Di Windows proses anak dibuat dengan "spawn" dan
# meng-import ulang SELURUH script ini (semua benchmark ikut jalan lagi!). Karena itu demo ini hanya
# dijalankan di Linux/Colab. Di script training biasa: bungkus kode dengan `if __name__ == "__main__":`.
if platform.system() != "Windows":
    print(f"PIL + transform, num_workers=2           : "
          f"{time_loader(DataLoader(pil_ds, batch_size=128, shuffle=True, num_workers=2)):6.2f} ms/batch")
print(f"tensor di memori (TensorDataset)          : {time_loader(make_loader(X_mem, y_mem, 128, shuffle=True)):6.2f} ms/batch")
# Bandingkan dengan ms/step model di bagian 2: kalau loading lebih lambat dari step, GPU menganggur menunggu data.

# %% ## 4. Memori
params = sum(p.numel() for p in SmallCNN().parameters())
print(f"parameter SmallCNN: {params:,} -> {params * 4 / 1e6:.2f} MB (float32)")
print(f"satu batch 1024 gambar: {1024 * 28 * 28 * 4 / 1e6:.2f} MB")
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    m = SmallCNN().cuda(); o = torch.optim.Adam(m.parameters())
    xb = torch.randn(1024, 1, 28, 28, device="cuda")
    nn.functional.cross_entropy(m(xb), torch.randint(0, 10, (1024,), device="cuda")).backward(); o.step()
    print(f"puncak memori GPU training step: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    # Aktivasi (output tiap layer, disimpan untuk backward) biasanya JAUH lebih besar dari parameter.

# %% [markdown]
# ## 5. Reproducibility: seed sama -> hasil sama?

# %%
def tiny_train(seed):
    set_seed(seed)
    model = SmallCNN().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    loader = make_loader(X_mem[:2048], y_mem[:2048], 256, shuffle=True, seed=seed)
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); loss = nn.functional.cross_entropy(model(xb), yb); loss.backward(); opt.step()
    return loss.item()

print("seed 1, run A:", tiny_train(1))
print("seed 1, run B:", tiny_train(1))
print("seed 2       :", tiny_train(2))
# Di GPU, beberapa operasi (misalnya atomicAdd di cuDNN) bisa sedikit tidak deterministik.
# Untuk hasil identik: torch.use_deterministic_algorithms(True) + torch.backends.cudnn.benchmark = False (lebih lambat).

# %% ## 6. Catat lingkungan eksperimen
env = {
    "python": platform.python_version(), "torch": torch.__version__,
    "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
    "cpu_threads": torch.get_num_threads(), "seed": 1, "benchmark": results,
}
(MODELS_DIR / f"week24_env_{device.type}.json").write_text(json.dumps(env, indent=2))
print(json.dumps({k: v for k, v in env.items() if k != "benchmark"}, indent=1))
