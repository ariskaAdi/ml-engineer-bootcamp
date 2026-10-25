"""Inference untuk file gambar.

    uv run python projects/p2-fashion-cnn/src/fashion_cnn/predict.py path/ke/gambar.png [gambar2.jpg ...]
    uv run python projects/p2-fashion-cnn/src/fashion_cnn/predict.py --export-samples 10   # buat contoh PNG dari test set
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fashion_cnn.model import CLASSES, FashionCNN, preprocess_image  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_DIR / "artifacts"


@lru_cache(maxsize=1)
def load_model(path: Path = ARTIFACTS / "model.pt") -> FashionCNN:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    model = FashionCNN(cfg["channels"], cfg["dropout"])
    model.load_state_dict(ckpt["model_state"])
    return model.eval()


@torch.no_grad()
def predict_image(img: Image.Image, top_k: int = 3) -> list[dict]:
    probs = load_model()(preprocess_image(img)).softmax(1)[0]
    top = probs.topk(top_k)
    return [{"label": CLASSES[i], "probability": round(p.item(), 4)} for p, i in zip(top.values, top.indices)]


def export_samples(n: int, dest: Path = PROJECT_DIR / "samples") -> list[Path]:
    """Simpan n gambar test FashionMNIST sebagai PNG (latar gelap, seperti data aslinya)."""
    from torchvision import datasets

    ds = datasets.FashionMNIST(PROJECT_DIR / "data", train=False, download=True)
    dest.mkdir(exist_ok=True)
    paths = []
    for i in range(n):
        img, label = ds[i]
        p = dest / f"{i:02d}_{CLASSES[label].replace('/', '-')}.png"
        img.resize((112, 112), Image.Resampling.NEAREST).save(p)
        paths.append(p)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--export-samples", type=int, default=0)
    args = ap.parse_args()
    files = [Path(p) for p in args.images]
    if args.export_samples:
        files += export_samples(args.export_samples)
    for f in files:
        print(f"{f.name:<28} -> {predict_image(Image.open(f))}")
