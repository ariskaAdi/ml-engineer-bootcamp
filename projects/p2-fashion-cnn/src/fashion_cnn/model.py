"""Arsitektur CNN + preprocessing yang dipakai SAMA saat training dan inference."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
MEAN, STD = 0.2860, 0.3530
IMG_SIZE = 28


class FashionCNN(nn.Module):
    """2 blok konvolusi -> classifier. ±0,4 juta parameter, cukup untuk dilatih di CPU."""

    def __init__(self, channels: int = 32, dropout: float = 0.3, n_classes: int = 10):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.MaxPool2d(2),                                   # 28 -> 14 -> 7
            )

        self.features = nn.Sequential(block(1, channels), block(channels, channels * 2))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(channels * 2 * 7 * 7, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (N, 1, 28, 28)
        return self.classifier(self.features(x))


def normalize(x: torch.Tensor) -> torch.Tensor:
    """x bernilai 0..1 -> dinormalisasi seperti data training."""
    return (x - MEAN) / STD


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Gambar apa pun -> tensor (1, 1, 28, 28) seperti FashionMNIST.

    FashionMNIST: grayscale, objek TERANG di latar GELAP, 28x28. Foto produk biasa justru
    latar putih, jadi dibalik (invert) kalau latarnya terang.
    """
    gray = ImageOps.grayscale(img)
    gray = ImageOps.pad(gray, (max(gray.size),) * 2, color=_border_color(gray))  # jadikan persegi tanpa distorsi
    gray = gray.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    if _border_mean(arr) > 0.5:          # latar terang -> balik
        arr = 1.0 - arr
    return normalize(torch.from_numpy(arr)).unsqueeze(0).unsqueeze(0)


def _border_mean(arr: np.ndarray) -> float:
    return float(np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]]).mean())


def _border_color(gray: Image.Image) -> int:
    return int(round(_border_mean(np.asarray(gray, dtype=np.float32) / 255.0) * 255))
