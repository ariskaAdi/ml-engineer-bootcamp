# %% [markdown]
# # Week 23 — Optimisasi dan Regularisasi
# Setiap eksperimen mengubah SATU hal. Data FashionMNIST dimuat ke memori (cepat).

# %%
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from mlbootcamp.paths import out
from mlbootcamp.torch_utils import fit, get_device, load_fashion_tensors, make_loader, set_seed

device = get_device()
X_all, y_all = load_fashion_tensors(train=True)
perm = torch.randperm(len(y_all), generator=torch.Generator().manual_seed(0))
val_idx, train_idx = perm[:10_000], perm[10_000:]
X_val, y_val = X_all[val_idx], y_all[val_idx]
val_loader = make_loader(X_val, y_val, batch_size=1024)


def mlp(hidden=512, dropout=0.0):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, 10),
    )


def run(name, n_train=10_000, epochs=10, lr=1e-3, batch_size=128, opt="adam", dropout=0.0,
        weight_decay=0.0, patience=None):
    set_seed(42)
    idx = train_idx[:n_train]
    loader = make_loader(X_all[idx], y_all[idx], batch_size=batch_size, shuffle=True)
    model = mlp(dropout=dropout)
    if opt == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9 if opt == "sgd_momentum" else 0.0,
                                    weight_decay=weight_decay)
    hist, model = fit(model, optimizer, loader, val_loader, epochs, device, patience=patience)
    best = hist.loc[hist["val_loss"].idxmin()]
    print(f"{name:<32} epochs={len(hist):>2}  best val_loss={best.val_loss:.3f} (epoch {int(best.epoch)})  "
          f"val_acc={best.val_acc:.3f}  final train_acc={hist.train_acc.iloc[-1]:.3f}  "
          f"{hist.seconds.mean():.2f}s/epoch")
    return hist


def plot(histories: dict, title: str, fname: str, metric="val_loss"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    for name, h in histories.items():
        line, = axes[0].plot(h.epoch, h.train_loss, "--", alpha=0.6)
        axes[0].plot(h.epoch, h[metric], color=line.get_color(), label=name)
        axes[1].plot(h.epoch, h.val_acc, color=line.get_color(), label=name)
    axes[0].set(title=f"{title}: loss (putus-putus = train)", xlabel="epoch", yscale="log")
    axes[1].set(title="val accuracy", xlabel="epoch"); axes[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out(fname)); plt.show()

# %% 1. Learning rate (Adam)
print("== Learning rate ==")
lr_runs = {f"lr={lr}": run(f"Adam lr={lr}", lr=lr) for lr in [1e-4, 1e-3, 1e-2, 1e-1]}
plot(lr_runs, "Learning rate", "week23_1_lr.png")

# %% 2. Batch size (epoch sama, lr sama)
print("\n== Batch size ==")
bs_runs = {f"bs={bs}": run(f"batch={bs}", batch_size=bs) for bs in [16, 128, 1024]}
plot(bs_runs, "Batch size", "week23_2_batch.png")

# %% 3. SGD vs Adam
print("\n== Optimizer ==")
opt_runs = {
    "SGD lr=0.01": run("SGD lr=0.01", opt="sgd", lr=0.01),
    "SGD+momentum lr=0.01": run("SGD+momentum lr=0.01", opt="sgd_momentum", lr=0.01),
    "Adam lr=1e-3": run("Adam lr=1e-3", opt="adam", lr=1e-3),
}
plot(opt_runs, "Optimizer", "week23_3_optimizer.png")

# %% [markdown]
# ## 4. Regularisasi: buat overfit dulu (data train kecil: 2000 gambar, model besar, 60 epoch)

# %%
print("\n== Regularisasi (2000 gambar train) ==")
reg = dict(n_train=2000, epochs=60)
reg_runs = {
    "tanpa regularisasi": run("tanpa regularisasi", **reg),
    "dropout 0.5": run("dropout 0.5", dropout=0.5, **reg),
    "weight decay 0.1": run("weight decay 0.1 (AdamW)", weight_decay=0.1, **reg),
    "early stopping (patience 5)": run("early stopping patience=5", patience=5, **reg),
    "lebih banyak data (20k)": run("tanpa regularisasi, 20k data", n_train=20_000, epochs=60),
}
plot(reg_runs, "Regularisasi", "week23_4_regularization.png")

summary = pd.DataFrame({name: {"best_val_loss": h.val_loss.min(), "final_val_loss": h.val_loss.iloc[-1],
                               "final_train_acc": h.train_acc.iloc[-1], "final_val_acc": h.val_acc.iloc[-1],
                               "gap_acc": h.train_acc.iloc[-1] - h.val_acc.iloc[-1]}
                        for name, h in reg_runs.items()}).T.round(3)
print(summary.to_string())
