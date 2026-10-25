# %% [markdown]
# # Week 5 — Aljabar Linear dan Gradient Descent dari nol (NumPy)

# %%
import matplotlib.pyplot as plt
import numpy as np

from mlbootcamp.datasets import make_house_prices
from mlbootcamp.paths import out

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Vektor, dot product, matriks
# Prediksi model linear = dot product antara fitur dan bobot:  ŷ = w1*x1 + w2*x2 + b

# %%
x = np.array([120.0, 3.0])          # satu rumah: luas 120 m2, 3 kamar
w = np.array([10.0, 50.0])          # bobot
b = 100.0
print("dot manual :", x[0] * w[0] + x[1] * w[1] + b)
print("dot numpy  :", x @ w + b)

X = np.array([[120.0, 3.0], [80.0, 2.0], [200.0, 4.0]])   # 3 rumah -> matriks (3, 2)
print("X shape", X.shape, "w shape", w.shape, "-> X @ w shape", (X @ w).shape)
print("prediksi 3 rumah sekaligus:", X @ w + b)
print("X.T shape:", X.T.shape)  # transpose: (2, 3)
# Aturan perkalian matriks: (n, k) @ (k, m) -> (n, m). Angka "dalam" harus sama.

# %% [markdown]
# ## 2. Turunan = kemiringan; gradien = vektor turunan
# f(w) = (w - 3)^2  -> f'(w) = 2(w - 3). Minimum di w=3, karena di sana kemiringan 0.

# %%
f = lambda w: (w - 3) ** 2
df = lambda w: 2 * (w - 3)
h = 1e-5
for w0 in [0.0, 3.0, 5.0]:
    numeric = (f(w0 + h) - f(w0 - h)) / (2 * h)       # turunan numerik (central difference)
    print(f"w={w0}: turunan rumus={df(w0):+.4f}  numerik={numeric:+.4f}")
# Tanda turunan memberi arah: negatif -> naikkan w; positif -> turunkan w.

# %% [markdown]
# ## 3. Chain rule (dasar backpropagation di Week 21)
# L = (y - ŷ)^2  dengan  ŷ = w*x + b
# dL/dw = dL/dŷ * dŷ/dw = -2(y - ŷ) * x

# %%
x0, y0, w0, b0 = 2.0, 10.0, 1.5, 0.5
L = lambda w, b: (y0 - (w * x0 + b)) ** 2
y_hat = w0 * x0 + b0
dL_dyhat = -2 * (y0 - y_hat)
grad_w_chain = dL_dyhat * x0       # dŷ/dw = x
grad_b_chain = dL_dyhat * 1        # dŷ/db = 1
grad_w_num = (L(w0 + h, b0) - L(w0 - h, b0)) / (2 * h)
grad_b_num = (L(w0, b0 + h) - L(w0, b0 - h)) / (2 * h)
print(f"dL/dw chain={grad_w_chain:.4f} numerik={grad_w_num:.4f}")
print(f"dL/db chain={grad_b_chain:.4f} numerik={grad_b_num:.4f}")

# %% [markdown]
# ## 4. Regresi linear dengan gradient descent
# Data: harga rumah vs luas tanah. Fitur di-standardisasi (mean 0, std 1) supaya
# gradient descent stabil — fitur berskala ratusan membuat gradien sangat besar.

# %%
house = make_house_prices(with_missing=False)
house = house[house["city"] == "Yogyakarta"]
x_raw = house["land_area"].to_numpy()
y = house["price"].to_numpy()
x_mean, x_std = x_raw.mean(), x_raw.std()
x_s = (x_raw - x_mean) / x_std


def mse(w, b):
    return np.mean((y - (w * x_s + b)) ** 2)


def gradient_descent(lr, steps=200):
    w, b = 0.0, 0.0
    history = []
    for _ in range(steps):
        y_hat = w * x_s + b
        error = y_hat - y
        grad_w = 2 * np.mean(error * x_s)   # dari chain rule di atas, dirata-ratakan
        grad_b = 2 * np.mean(error)
        w -= lr * grad_w                     # langkah BERLAWANAN arah gradien
        b -= lr * grad_b
        history.append((w, b, mse(w, b)))
    return np.array(history)


hist = gradient_descent(lr=0.1)
w_gd, b_gd, loss_gd = hist[-1]
print(f"GD: w={w_gd:.2f}, b={b_gd:.2f}, MSE={loss_gd:,.0f}")

# Solusi tertutup (normal equation / least squares) sebagai pembanding
A = np.column_stack([x_s, np.ones_like(x_s)])
(w_ls, b_ls), *_ = np.linalg.lstsq(A, y, rcond=None)
print(f"LS: w={w_ls:.2f}, b={b_ls:.2f}, MSE={mse(w_ls, b_ls):,.0f}")
# Kembalikan ke skala asli: harga = w/std * luas + (b - w*mean/std)
print(f"Artinya: setiap +1 m2 tanah -> +{w_gd / x_std:.2f} juta (di Yogyakarta)")

# %% Pengaruh learning rate
fig, ax = plt.subplots(figsize=(8, 4))
for lr in [0.01, 0.1, 0.5, 1.01]:
    h_ = gradient_descent(lr, steps=60)
    ax.plot(h_[:, 2], label=f"lr={lr}")
ax.set(yscale="log", xlabel="langkah", ylabel="MSE", title="Learning rate: terlalu kecil lambat, terlalu besar meledak")
ax.legend()
plt.tight_layout(); plt.savefig(out("week05_1_learning_rates.png")); plt.show()

# %% Loss sebagai fungsi parameter + jalur gradient descent
ws = np.linspace(w_ls - 1500, w_ls + 1500, 120)
bs = np.linspace(b_ls - 1500, b_ls + 1500, 120)
W, B = np.meshgrid(ws, bs)
Z = np.mean((y[None, None, :] - (W[..., None] * x_s + B[..., None])) ** 2, axis=-1)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
cs = axes[0].contour(W, B, Z, levels=30)
axes[0].plot(hist[:30, 0], hist[:30, 1], "r.-", label="jalur GD (lr=0.1)")
axes[0].plot(w_ls, b_ls, "k*", ms=12, label="minimum")
axes[0].set(xlabel="w", ylabel="b", title="Kontur loss MSE(w, b)"); axes[0].legend()
axes[1].plot(ws, [mse(wi, b_ls) for wi in ws])
axes[1].set(xlabel="w (b tetap optimal)", ylabel="MSE", title="Potongan loss: berbentuk mangkuk (convex)")
plt.tight_layout(); plt.savefig(out("week05_2_loss_surface.png")); plt.show()

# %% Garis hasil
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(x_raw, y, s=8, alpha=0.4)
grid = np.linspace(x_raw.min(), x_raw.max(), 100)
ax.plot(grid, w_gd * (grid - x_mean) / x_std + b_gd, "r", label="model GD")
ax.set(xlabel="luas tanah (m2)", ylabel="harga (juta)"); ax.legend()
plt.tight_layout(); plt.savefig(out("week05_3_fit.png")); plt.show()

# %% [markdown]
# ## 5. Bonus: banyak fitur sekaligus (bentuk matriks)
# ŷ = X @ w + b ; gradien = (2/n) X.T @ (ŷ - y)

# %%
feats = ["land_area", "building_area", "bathrooms", "distance_km"]
Xm = house[feats].to_numpy(dtype=float)
Xm = (Xm - Xm.mean(0)) / Xm.std(0)
wv, bv = np.zeros(len(feats)), 0.0
for _ in range(500):
    err = Xm @ wv + bv - y
    wv -= 0.1 * 2 / len(y) * (Xm.T @ err)
    bv -= 0.1 * 2 * err.mean()
print("bobot per fitur (skala standar):", {f: round(float(v), 1) for f, v in zip(feats, wv)})
print("MSE multi-fitur:", f"{np.mean((Xm @ wv + bv - y) ** 2):,.0f}", "vs 1 fitur:", f"{loss_gd:,.0f}")
