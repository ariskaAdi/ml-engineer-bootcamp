# %% [markdown]
# # Week 20 — Tensor dan Dasar GPU
# Laptop tanpa GPU tetap bisa menjalankan semuanya; bagian GPU otomatis aktif di Google Colab
# (Runtime -> Change runtime type -> T4 GPU).

# %%
import time

import numpy as np
import torch

print("torch", torch.__version__, "| CUDA tersedia:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# %% 1. Membuat tensor: shape dan dtype
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(a.shape, a.dtype)                   # torch.Size([2, 3]) torch.int64
b = torch.tensor([[1.0, 2.0, 3.0]])
print(b.shape, b.dtype)                   # float32 = default untuk deep learning
print(torch.zeros(2, 3), torch.ones(2, 3).dtype, torch.arange(6).reshape(2, 3))
x = torch.randn(4, 3, 28, 28)              # batch 4 gambar, 3 channel, 28x28
print("batch gambar:", x.shape, "| jumlah elemen:", x.numel())

# %% Mencampur dtype: integer / integer -> float, int + float -> float
print((a / 2).dtype, (a + b).dtype)
# Banyak loss function MENUNTUT dtype tertentu: CrossEntropyLoss butuh label int64 (long) dan logit float32.
labels = torch.tensor([0, 2, 1])
print(labels.dtype, labels.float().dtype)

# %% 2. Indexing — sama seperti NumPy
m = torch.arange(12).reshape(3, 4)
print(m[0], m[:, 1], m[1:, 2:], m[m > 6])

# %% 3. Broadcasting — aturan sama seperti NumPy (Week 1)
row = torch.tensor([10, 20, 30, 40])        # (4,)
col = torch.tensor([[1], [2], [3]])          # (3,1)
print((m + row).shape, (m * col).shape)
# Kasus nyata: normalisasi per channel. Gambar (N, C, H, W), mean per channel (C,) -> reshape ke (1, C, 1, 1)
mean = torch.tensor([0.5, 0.4, 0.3]).view(1, 3, 1, 1)
print("normalisasi per channel:", ((x - mean)).shape)

# %% 4. reshape/view, squeeze/unsqueeze, permute — operasi shape yang paling sering
img = torch.rand(28, 28)
print(img.unsqueeze(0).shape)                 # (1, 28, 28) tambah dimensi channel
print(img.unsqueeze(0).unsqueeze(0).shape)    # (1, 1, 28, 28) tambah dimensi batch
print(torch.rand(1, 1, 28, 28).squeeze().shape)
hwc = torch.rand(32, 32, 3)                   # format gambar biasa (OpenCV/PIL): H, W, C
print(hwc.permute(2, 0, 1).shape)             # PyTorch butuh C, H, W
print(x.flatten(start_dim=1).shape)           # (4, 3*28*28) untuk masuk ke layer Linear

# %% 5. NumPy <-> torch
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)        # BERBAGI memori dengan arr (dtype float64!)
arr[0] = 100
print(t)                          # ikut berubah
t2 = torch.tensor(arr)           # menyalin
print(t2.float().dtype, t.numpy())

# %% [markdown]
# ## 6. Device: CPU vs GPU
# Aturan emas: semua tensor dalam satu operasi (termasuk parameter model) harus di device yang SAMA.

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = torch.nn.Linear(3, 1).to(device)       # .to() memindahkan parameter model
inp = torch.randn(5, 3).to(device)             # data juga harus dipindah
print(model(inp).device)

# %% Mendiagnosis device mismatch.
# Tanpa GPU kita pakai device "meta" (tensor tanpa data) untuk memancing error yang sama.
other = torch.device("cuda") if torch.cuda.is_available() else torch.device("meta")
try:
    model(torch.randn(5, 3, device=other) if other.type == "meta" else torch.randn(5, 3))  # sengaja salah
except (RuntimeError, NotImplementedError) as e:
    print("ERROR:", str(e).splitlines()[0][:150])
# Cara diagnosis: print(tensor.device) dan next(model.parameters()).device, lalu .to(device) yang lupa.
print("param di:", next(model.parameters()).device, "| input di:", inp.device)

# %% 7. Kapan GPU membantu? Ukur dengan benar.
def bench(fn, repeats=5):
    fn()                                         # warm-up (alokasi memori, kompilasi kernel)
    if torch.cuda.is_available():
        torch.cuda.synchronize()                 # GPU bekerja asinkron -> tunggu selesai sebelum stop timer
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.median(times)

for n in [16, 256, 2048]:
    A_cpu, B_cpu = torch.randn(n, n), torch.randn(n, n)
    t_cpu = bench(lambda: A_cpu @ B_cpu)
    msg = f"matmul {n:>4}x{n:<4} CPU {t_cpu * 1e3:8.3f} ms"
    if torch.cuda.is_available():
        A_gpu, B_gpu = A_cpu.to("cuda"), B_cpu.to("cuda")
        t_gpu = bench(lambda: A_gpu @ B_gpu)
        t_copy = bench(lambda: (A_cpu.to("cuda") @ B_cpu.to("cuda")).cpu())
        msg += f" | GPU {t_gpu * 1e3:8.3f} ms | GPU+copy {t_copy * 1e3:8.3f} ms"
    print(msg)
# Workload kecil: overhead (memanggil kernel, menyalin data CPU<->GPU) > waktu hitung -> GPU tidak lebih cepat.
