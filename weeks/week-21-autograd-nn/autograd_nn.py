# %% [markdown]
# # Week 21 — Autograd dan Neural Network

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch import nn

from mlbootcamp.paths import out

torch.manual_seed(0)

# %% [markdown]
# ## 1. Autograd = chain rule otomatis
# Ulangi contoh Week 5: L = (y - (w*x + b))^2, x=2, y=10, w=1.5, b=0.5
# Hasil tangan Week 5: dL/dw = -26, dL/db = -13

# %%
x, y = torch.tensor(2.0), torch.tensor(10.0)
w = torch.tensor(1.5, requires_grad=True)   # "lacak operasi pada tensor ini"
b = torch.tensor(0.5, requires_grad=True)
y_hat = w * x + b                            # PyTorch membangun computational graph
loss = (y - y_hat) ** 2
print("grad_fn:", loss.grad_fn, "<-", loss.grad_fn.next_functions[0][0])
loss.backward()                              # chain rule dari loss ke setiap daun
print(f"dL/dw = {w.grad.item()}  dL/db = {b.grad.item()}")

# %% Gradien DIJUMLAHKAN kalau backward dipanggil lagi -> harus di-nol-kan setiap langkah
loss2 = (y - (w * x + b)) ** 2
loss2.backward()
print("setelah backward kedua (terakumulasi):", w.grad.item())
w.grad = None; b.grad = None

# %% Satu langkah gradient descent "manual" dengan autograd
lr = 0.05
for step in range(5):
    loss = (y - (w * x + b)) ** 2
    loss.backward()
    with torch.no_grad():                   # update parameter TIDAK boleh ikut direkam di graph
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_(); b.grad.zero_()
    print(f"step {step}: loss={loss.item():.4f} w={w.item():.3f} b={b.item():.3f}")

# %% [markdown]
# ## 2. Data: dua bulan sabit (tidak bisa dipisah dengan garis lurus)

# %%
X, Y = make_moons(n_samples=1000, noise=0.25, random_state=0)
X_tr, X_va, Y_tr, Y_va = train_test_split(X, Y, test_size=0.3, random_state=0)
X_tr, X_va = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(X_va, dtype=torch.float32)
Y_tr, Y_va = torch.tensor(Y_tr, dtype=torch.long), torch.tensor(Y_va, dtype=torch.long)

# %% [markdown]
# ## 3. nn.Module: cara standar mendefinisikan model

# %%
class MLP(nn.Module):
    def __init__(self, n_in=2, hidden=32, n_out=2, activation=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),   # bobot (hidden, n_in) + bias -> parameter yang dilatih
            activation(),              # non-linearitas: tanpa ini, tumpukan Linear = satu Linear saja
            nn.Linear(hidden, hidden),
            activation(),
            nn.Linear(hidden, n_out),  # output = LOGIT (skor mentah), bukan peluang
        )

    def forward(self, x):
        return self.net(x)


model = MLP()
print(model)
print("jumlah parameter:", sum(p.numel() for p in model.parameters()))

# %% Fungsi aktivasi
z = torch.linspace(-4, 4, 200)
fig, ax = plt.subplots(figsize=(7, 3))
for name, fn in [("ReLU", torch.relu), ("Sigmoid", torch.sigmoid), ("Tanh", torch.tanh)]:
    ax.plot(z, fn(z), label=name)
ax.legend(); ax.set_title("Fungsi aktivasi"); plt.tight_layout(); plt.savefig(out("week21_1_activations.png")); plt.show()

# %% Loss: CrossEntropyLoss = softmax + negative log likelihood, menerima LOGIT dan label int64
logits = torch.tensor([[2.0, 0.5], [0.1, 3.0]])
target = torch.tensor([0, 1])
ce = nn.CrossEntropyLoss()(logits, target)
manual = -torch.log_softmax(logits, dim=1)[torch.arange(2), target].mean()
print(f"CE PyTorch={ce.item():.4f}  manual={manual.item():.4f}")
print("peluang:", torch.softmax(logits, dim=1))

# %% [markdown]
# ## 4. Training: prediksi -> loss -> gradien -> update

# %%
def train(model, optimizer, epochs=300):
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        logits = model(X_tr)                 # 1. prediksi (forward)
        loss = loss_fn(logits, Y_tr)         # 2. seberapa salah
        optimizer.zero_grad()                # 3a. hapus gradien lama
        loss.backward()                      # 3b. gradien loss terhadap SEMUA parameter
        optimizer.step()                     # 4. update: param -= lr * grad (untuk SGD)
        model.eval()
        with torch.no_grad():
            val_acc = (model(X_va).argmax(1) == Y_va).float().mean().item()
        history.append((loss.item(), val_acc))
    return np.array(history)


results = {}
for name, make_opt in [("SGD lr=0.1", lambda p: torch.optim.SGD(p, lr=0.1)),
                       ("SGD+momentum", lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9)),
                       ("Adam lr=0.01", lambda p: torch.optim.Adam(p, lr=0.01))]:
    torch.manual_seed(0)
    m = MLP()
    results[name] = (m, train(m, make_opt(m.parameters())))
    print(f"{name:<14} loss akhir={results[name][1][-1, 0]:.3f}  val acc={results[name][1][-1, 1]:.3f}")

torch.manual_seed(0)
linear_only = nn.Linear(2, 2)                # pembanding: tanpa hidden layer = logistic regression
h_lin = train(linear_only, torch.optim.Adam(linear_only.parameters(), lr=0.01))
print(f"{'Linear saja':<14} loss akhir={h_lin[-1, 0]:.3f}  val acc={h_lin[-1, 1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
for name, (_, h) in results.items():
    axes[0].plot(h[:, 0], label=name); axes[1].plot(h[:, 1], label=name)
axes[0].set(title="training loss", xlabel="epoch"); axes[1].set(title="validation accuracy", xlabel="epoch")
axes[0].legend(); plt.tight_layout(); plt.savefig(out("week21_2_optimizers.png")); plt.show()

# %% Decision boundary: MLP bisa melengkung, Linear tidak
xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (title, m) in zip(axes, [("Linear saja", linear_only), ("MLP (Adam)", results["Adam lr=0.01"][0])]):
    with torch.no_grad():
        zz = torch.softmax(m(grid), 1)[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.6)
    ax.scatter(X_va[:, 0], X_va[:, 1], c=Y_va, cmap="RdBu_r", s=8, edgecolor="k", linewidth=0.2)
    ax.set_title(title)
plt.tight_layout(); plt.savefig(out("week21_3_boundary.png")); plt.show()
