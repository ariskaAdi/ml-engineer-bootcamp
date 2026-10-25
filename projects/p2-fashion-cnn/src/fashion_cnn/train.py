"""Latih FashionCNN. Semua setting lewat argumen CLI dan tercatat di artifacts/.

Contoh (dari root repo):
    uv run python projects/p2-fashion-cnn/src/fashion_cnn/train.py --epochs 8
    uv run python projects/p2-fashion-cnn/src/fashion_cnn/train.py --epochs 2 --train-size 5000   # cepat
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fashion_cnn.model import CLASSES, FashionCNN, normalize  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    epochs: int = 8
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    channels: int = 32
    patience: int = 3
    train_size: int = 50_000          # sisa dari 60.000 -> validation (maks 10.000)
    augment: bool = True              # flip horizontal acak (pakaian simetris kiri-kanan)
    seed: int = 42
    data_dir: str = str(PROJECT_DIR / "data")
    out_dir: str = str(PROJECT_DIR / "artifacts")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_split(cfg: Config):
    from torchvision import datasets

    def to_tensors(ds):
        return ds.data.unsqueeze(1).float().div(255), ds.targets.clone()   # (N,1,28,28) 0..1

    X, y = to_tensors(datasets.FashionMNIST(cfg.data_dir, train=True, download=True))
    X_test, y_test = to_tensors(datasets.FashionMNIST(cfg.data_dir, train=False, download=True))
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(cfg.seed))
    val_idx = perm[:10_000]
    tr_idx = perm[10_000:10_000 + cfg.train_size]
    return (X[tr_idx], y[tr_idx]), (X[val_idx], y[val_idx]), (X_test, y_test)


def run_epoch(model, loader, device, loss_fn, optimizer=None, augment=False):
    """Satu epoch. optimizer=None -> mode evaluasi (tanpa gradien)."""
    training = optimizer is not None
    model.train(training)
    total, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            if augment:
                flip = torch.rand(len(X), device=device) < 0.5
                X = torch.where(flip.view(-1, 1, 1, 1), X.flip(-1), X)
            logits = model(normalize(X))
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total / n, correct / n


@torch.no_grad()
def predict_all(model, X, device, batch=1024):
    model.eval()
    return torch.cat([model(normalize(X[i:i + batch].to(device))).softmax(1).cpu() for i in range(0, len(X), batch)])


def main(cfg: Config) -> dict:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    (X_tr, y_tr), (X_val, y_val), (X_test, y_test) = load_split(cfg)
    print(f"device={device} train={len(y_tr)} val={len(y_val)} test={len(y_test)}")

    g = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=cfg.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=1024)

    model = FashionCNN(cfg.channels, cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, epochs=cfg.epochs,
                                                    steps_per_epoch=len(train_loader))
    loss_fn = nn.CrossEntropyLoss()

    # scheduler.step() dipanggil per batch -> bungkus optimizer.step
    original_step = optimizer.step

    def step_with_schedule(*a, **kw):
        result = original_step(*a, **kw)
        scheduler.step()
        return result

    optimizer.step = step_with_schedule

    history, best_loss, best_state, bad = [], float("inf"), None, 0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, loss_fn, optimizer, augment=cfg.augment)
        va_loss, va_acc = run_epoch(model, val_loader, device, loss_fn)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss,
                        "val_acc": va_acc, "lr": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - t0})
        print(f"epoch {epoch:>2}: train {tr_loss:.4f}/{tr_acc:.4f}  val {va_loss:.4f}/{va_acc:.4f}  "
              f"({history[-1]['seconds']:.0f}s)")
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)      # tersimpan setiap epoch
        if va_loss < best_loss:
            best_loss, best_state, bad = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"early stopping di epoch {epoch}")
                break

    model.load_state_dict(best_state)
    optimizer.step = original_step
    torch.save({"model_state": model.state_dict(), "config": asdict(cfg), "classes": CLASSES},
               out / "model.pt")

    probs = predict_all(model, X_test, device)
    pred = probs.argmax(1)
    cm = torch.zeros(10, 10, dtype=torch.long)
    for t, p in zip(y_test, pred):
        cm[t, p] += 1
    per_class = {c: round(cm[i, i].item() / cm[i].sum().item(), 4) for i, c in enumerate(CLASSES)}
    metrics = {
        "test_accuracy": round((pred == y_test).float().mean().item(), 4),
        "best_val_loss": round(best_loss, 4),
        "best_epoch": int(pd.DataFrame(history)["val_loss"].idxmin()) + 1,
        "per_class_accuracy": per_class,
        "confusion_matrix": cm.tolist(),
        "config": asdict(cfg),
        "env": {"python": platform.python_version(), "torch": torch.__version__, "device": str(device)},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.save(out / "test_probs.npy", probs.numpy())
    print(f"TEST accuracy = {metrics['test_accuracy']}  -> artefak di {out}")
    return metrics


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field, default in asdict(Config()).items():
        if isinstance(default, bool):
            p.add_argument(f"--{field.replace('_', '-')}", type=lambda s: s.lower() in {"1", "true", "yes"}, default=default)
        else:
            p.add_argument(f"--{field.replace('_', '-')}", type=type(default), default=default)
    return Config(**vars(p.parse_args(argv)))


if __name__ == "__main__":
    main(parse_args())
