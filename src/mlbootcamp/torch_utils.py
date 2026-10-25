"""Loop training PyTorch dari Week 22, dirapikan supaya bisa dipakai ulang (Week 23–25)."""

from __future__ import annotations

import copy
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .paths import DATA_DIR

FASHION_MEAN, FASHION_STD = 0.2860, 0.3530


def set_seed(seed: int = 42) -> None:
    """Seed semua sumber keacakan yang biasa dipakai."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_fashion_tensors(train: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """FashionMNIST langsung sebagai tensor di memori: (N, 1, 28, 28) float32 ternormalisasi + label.

    Jauh lebih cepat daripada transform PIL per sampel (lihat Week 24)."""
    from torchvision import datasets

    ds = datasets.FashionMNIST(DATA_DIR, train=train, download=True)
    X = ds.data.unsqueeze(1).float().div(255).sub(FASHION_MEAN).div(FASHION_STD)
    return X, ds.targets.clone()


def make_loader(X, y, batch_size=128, shuffle=False, seed=42) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle, generator=g)


def train_one_epoch(model, loader, loss_fn, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        total_loss += loss_fn(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def fit(model, optimizer, train_loader, val_loader, epochs: int, device=None,
        patience: int | None = None, verbose: bool = False) -> tuple[pd.DataFrame, nn.Module]:
    """Latih, catat metrik per epoch. Kalau `patience` diisi: early stopping berdasarkan val_loss,
    dan bobot TERBAIK dikembalikan ke model."""
    device = device or get_device()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    rows, best_loss, best_state, bad_epochs = [], float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, loss_fn, device)
        rows.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                     "val_loss": va_loss, "val_acc": va_acc, "seconds": time.perf_counter() - t0})
        if verbose:
            print(f"  epoch {epoch:>3}: train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")
        if va_loss < best_loss:
            best_loss, best_state, bad_epochs = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if patience is not None and bad_epochs >= patience:
                break
    if patience is not None and best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(rows), model
