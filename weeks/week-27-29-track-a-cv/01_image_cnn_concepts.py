# %% [markdown]
# # Track A, Week 27 — Data Gambar dan Konsep CNN
# Gambar = tensor angka. Konvolusi = filter kecil yang digeser ke seluruh gambar.

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

from mlbootcamp.paths import DATA_DIR, out

# %% 1. Gambar sebagai angka
ds = datasets.FashionMNIST(DATA_DIR, train=True, download=True)
img, label = ds[0]                       # PIL Image grayscale 28x28
arr = np.asarray(img)
print("shape:", arr.shape, "dtype:", arr.dtype, "min/max:", arr.min(), arr.max())
print(arr[10:16, 10:16])                 # potongan nilai piksel 0..255

rgb = np.zeros((64, 64, 3), dtype=np.uint8)   # gambar warna buatan: H x W x C
rgb[:, :32, 0] = 255                          # kiri merah
rgb[:32, :, 2] = 200                          # atas biru
print("RGB shape (H, W, C):", rgb.shape, "-> tensor PyTorch (C, H, W):", torch.from_numpy(rgb).permute(2, 0, 1).shape)

# %% 2. Resize dan normalisasi
x = torch.from_numpy(arr).float().div(255)[None, None]          # (1, 1, 28, 28), nilai 0..1
small = F.interpolate(x, size=(14, 14), mode="bilinear", align_corners=False)
big = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
norm = (x - 0.2860) / 0.3530                                      # mean/std dataset -> berpusat di 0
print(f"normalisasi: mean {x.mean():.3f}->{norm.mean():.3f}, std {x.std():.3f}->{norm.std():.3f}")

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for ax, (t, title) in zip(axes, [(x, "asli 28x28"), (small, "resize 14x14 (detail hilang)"),
                                  (big, "resize 112x112 (tidak menambah detail)"), (norm, "dinormalisasi")]):
    ax.imshow(t[0, 0], cmap="gray"); ax.set_title(title, fontsize=9); ax.axis("off")
plt.tight_layout(); plt.savefig(out("trackA_1_resize_normalize.png")); plt.show()

# %% 3. Konvolusi dengan filter buatan tangan
sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])      # mendeteksi tepi vertikal
sobel_y = sobel_x.T                                                       # tepi horizontal
blur = torch.ones(3, 3) / 9
kernels = torch.stack([sobel_x, sobel_y, blur])[:, None]                  # (out=3, in=1, 3, 3)
fmap = F.conv2d(x, kernels, padding=1)                                    # (1, 3, 28, 28)
print("input", x.shape, "-> feature maps", fmap.shape)

# Hitung SATU nilai output dengan tangan untuk memastikan paham
i, j = 14, 14
patch = F.pad(x, (1, 1, 1, 1))[0, 0, i:i + 3, j:j + 3]
print(f"manual (patch * sobel_x).sum() = {(patch * sobel_x).sum():.4f} | conv2d = {fmap[0, 0, i, j]:.4f}")

pooled = F.max_pool2d(fmap, 2)                                            # (1, 3, 14, 14): ambil maksimum tiap 2x2
print("setelah max pool 2x2:", pooled.shape)

fig, axes = plt.subplots(2, 3, figsize=(9, 6))
for k, name in enumerate(["Sobel X (tepi vertikal)", "Sobel Y (tepi horizontal)", "Blur"]):
    axes[0, k].imshow(fmap[0, k], cmap="RdBu_r"); axes[0, k].set_title(name, fontsize=9)
    axes[1, k].imshow(pooled[0, k], cmap="RdBu_r"); axes[1, k].set_title("setelah max-pool", fontsize=9)
for a in axes.ravel():
    a.axis("off")
plt.tight_layout(); plt.savefig(out("trackA_2_conv_filters.png")); plt.show()

# %% 4. Kenapa konvolusi, bukan Linear?
linear_params = (28 * 28) * 128
conv_params = 32 * (1 * 3 * 3) + 32
print(f"Linear 784->128: {linear_params:,} parameter | Conv 32 filter 3x3: {conv_params:,} parameter")
# Konvolusi: (1) bobot yang sama dipakai di seluruh gambar (weight sharing) -> jauh lebih sedikit parameter,
# (2) melihat pola LOKAL (tepi, tekstur), (3) objek yang bergeser tetap terdeteksi oleh filter yang sama.

# %% 5. Filter yang DIPELAJARI oleh CNN Portfolio 2 (kalau sudah dilatih)
from mlbootcamp.paths import ROOT

ckpt_path = ROOT / "projects/p2-fashion-cnn/artifacts/model.pt"
if ckpt_path.exists():
    w = torch.load(ckpt_path, map_location="cpu")["model_state"]["features.0.0.weight"]   # (32, 1, 3, 3)
    fig, axes = plt.subplots(4, 8, figsize=(10, 5))
    for k, ax in enumerate(axes.ravel()):
        ax.imshow(w[k, 0], cmap="RdBu_r"); ax.axis("off")
    plt.suptitle("32 filter 3x3 layer pertama yang dipelajari CNN P2"); plt.tight_layout()
    plt.savefig(out("trackA_3_learned_filters.png")); plt.show()
else:
    print("Latih P2 dulu untuk melihat filter yang dipelajari.")
