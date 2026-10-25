# %% [markdown]
# # Track A, Week 29 — OpenCV, Preprocessing Konsisten, dan Analisis Error
# Jalankan setelah 02_transfer_learning.py (memakai models/trackA_resnet18_cifar.pt).

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import datasets, models

from mlbootcamp.paths import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, out

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MEAN, STD = np.array([0.485, 0.456, 0.406], dtype=np.float32), np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG = 96

# %% 1. Simpan 1000 gambar test sebagai file PNG (seperti data yang datang ke sistem produksi)
test = datasets.CIFAR10(DATA_DIR, train=False, download=True)
img_dir = OUTPUTS_DIR / "trackA_images"; img_dir.mkdir(exist_ok=True)
rng = np.random.default_rng(7)
rows = []
for i in rng.permutation(len(test))[:1000]:
    pil, label = test[int(i)]
    path = img_dir / f"{i:05d}.png"
    pil.save(path)
    rows.append({"path": path, "label": label})
files = pd.DataFrame(rows)

# %% 2. OpenCV dasar: cv2 membaca sebagai BGR (bukan RGB!), uint8, shape (H, W, C)
bgr = cv2.imread(str(files.path[0]))
print("cv2.imread:", bgr.shape, bgr.dtype, "| piksel [0,0] BGR =", bgr[0, 0])
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
print("setelah cvtColor RGB =", rgb[0, 0])
resized = cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_LINEAR)
flipped = cv2.flip(rgb, 1)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
fig, axes = plt.subplots(1, 5, figsize=(13, 3))
for ax, (im, t) in zip(axes, [(bgr, "BGR ditampilkan sbg RGB (salah warna)"), (rgb, "RGB benar"), (resized, f"resize {IMG}"),
                              (flipped, "flip"), (edges, "Canny edges")]):
    ax.imshow(im, cmap="gray" if im.ndim == 2 else None); ax.set_title(t, fontsize=8); ax.axis("off")
plt.tight_layout(); plt.savefig(out("trackA_4_opencv.png")); plt.show()

# %% 3. SATU fungsi preprocessing untuk training DAN inference
def preprocess(bgr_image: np.ndarray) -> torch.Tensor:
    """BGR uint8 (H, W, 3) dari OpenCV -> tensor (3, IMG, IMG) persis seperti transform training."""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_LINEAR)
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x).permute(2, 0, 1)


def preprocess_buggy(bgr_image):
    """Kesalahan umum: lupa konversi BGR->RGB."""
    x = (cv2.resize(bgr_image, (IMG, IMG)).astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x).permute(2, 0, 1)

# %% 4. Muat model dan bandingkan preprocessing benar vs salah
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load(MODELS_DIR / "trackA_resnet18_cifar.pt", map_location="cpu"))
model.eval()


@torch.no_grad()
def predict_files(fn):
    probs = []
    for i in range(0, len(files), 100):
        batch = torch.stack([fn(cv2.imread(str(p))) for p in files.path[i:i + 100]])
        probs.append(model(batch).softmax(1))
    return torch.cat(probs).numpy()

probs = predict_files(preprocess)
probs_bug = predict_files(preprocess_buggy)
files["pred"], files["conf"] = probs.argmax(1), probs.max(1)
print(f"akurasi preprocessing konsisten : {(files.pred == files.label).mean():.3f}")
print(f"akurasi lupa BGR->RGB          : {(probs_bug.argmax(1) == files.label).mean():.3f}")
# Tidak ada error yang muncul — model tetap 'jalan', hanya diam-diam lebih buruk. Itu bahayanya.

# %% 5. Metrik per kelas + confusion
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(files.label, files.pred, target_names=CLASSES, digits=3))
cm = pd.DataFrame(confusion_matrix(files.label, files.pred), index=CLASSES, columns=CLASSES)
pairs = [(CLASSES[i], CLASSES[j], cm.iloc[i, j]) for i in range(10) for j in range(10) if i != j]
print("pasangan paling sering tertukar:", sorted(pairs, key=lambda t: -t[2])[:5])

# %% 6. Lihat contoh yang salah (paling yakin tapi salah = paling menarik)
wrong = files[files.pred != files.label].sort_values("conf", ascending=False).head(12)
fig, axes = plt.subplots(2, 6, figsize=(13, 5))
for ax, r in zip(axes.ravel(), wrong.itertuples()):
    ax.imshow(cv2.cvtColor(cv2.imread(str(r.path)), cv2.COLOR_BGR2RGB))
    ax.set_title(f"benar: {CLASSES[r.label]}\ntebak: {CLASSES[r.pred]} ({r.conf:.2f})", fontsize=8); ax.axis("off")
plt.tight_layout(); plt.savefig(out("trackA_5_misclassified.png")); plt.show()
