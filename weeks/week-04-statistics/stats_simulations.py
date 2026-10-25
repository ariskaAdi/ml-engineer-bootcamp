# %% [markdown]
# # Week 4 — Statistik dan Peluang lewat Simulasi
#
# Ide utama: daripada menghafal rumus, kita SIMULASIKAN ribuan kali dengan NumPy,
# lalu cek apakah rumusnya cocok dengan hasil simulasi.

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mlbootcamp.paths import out

rng = np.random.default_rng(42)

# %% [markdown]
# ## 1. Satu angka ringkasan tidak cukup
# Tiga "toko" dengan rata-rata transaksi SAMA (100 ribu), tapi bentuk distribusinya sangat berbeda.

# %%
n = 5000
toko_a = rng.normal(100, 10, n)                     # stabil
toko_b = rng.exponential(100, n)                    # banyak kecil, sedikit sangat besar
toko_c = np.concatenate([rng.normal(50, 5, n // 2), rng.normal(150, 5, n // 2)])  # dua kelompok
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharex=True)
for ax, (name, x) in zip(axes, {"A normal": toko_a, "B miring": toko_b, "C dua puncak": toko_c}.items()):
    ax.hist(x, bins=60)
    ax.set_title(f"{name}\nmean={x.mean():.0f} median={np.median(x):.0f} std={x.std():.0f}")
plt.tight_layout(); plt.savefig(out("week04_1_same_mean.png")); plt.show()
# Di toko C, TIDAK ADA transaksi yang nilainya dekat rata-rata (100)!

# %% [markdown]
# ## 2. Kenapa sampel bervariasi — dan Central Limit Theorem
# Populasi = 1 juta transaksi (miring). Kita ambil sampel n=30, hitung mean, ulangi 5000 kali.

# %%
population = rng.exponential(100, 1_000_000)
mu, sigma = population.mean(), population.std()

def sample_means(n, reps=5000):
    idx = rng.integers(0, len(population), size=(reps, n))
    return population[idx].mean(axis=1)

fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
axes[0].hist(population[:20000], bins=60); axes[0].set_title("Populasi (miring)")
for ax, n in zip(axes[1:], [5, 30, 200]):
    m = sample_means(n)
    ax.hist(m, bins=60)
    ax.set_title(f"Mean sampel n={n}\nstd={m.std():.1f} | rumus σ/√n={sigma / np.sqrt(n):.1f}")
plt.tight_layout(); plt.savefig(out("week04_2_clt.png")); plt.show()
# CLT: distribusi MEAN sampel mendekati normal, walaupun populasinya miring.
# Standard error (std dari mean sampel) = σ/√n -> sampel 4x lebih besar = error 2x lebih kecil.

# %% [markdown]
# ## 3. Sampling bias: sampel besar tidak menolong kalau caranya salah
# Survei kepuasan hanya lewat aplikasi -> pelanggan muda lebih banyak menjawab.

# %%
N = 100_000
age = rng.integers(18, 70, N)
satisfaction = 8 - 0.05 * (age - 18) + rng.normal(0, 1, N)       # yang tua kurang puas
p_answer = np.where(age < 35, 0.30, 0.05)                         # yang muda lebih rajin menjawab
answered = rng.random(N) < p_answer
print(f"Kepuasan sebenarnya (semua)  : {satisfaction.mean():.2f}")
print(f"Kepuasan dari survei aplikasi: {satisfaction[answered].mean():.2f}  (n={answered.sum()})")
random_sample = rng.choice(N, 500, replace=False)
print(f"Sampel acak n=500            : {satisfaction[random_sample].mean():.2f}")
# Sampel acak kecil (500) lebih akurat daripada sampel bias besar (±13.000).

# %% [markdown]
# ## 4. Peluang bersyarat dan Teorema Bayes
# Alarm fraud: 1% transaksi fraud. Model menangkap 95% fraud (sensitivity),
# tapi juga salah alarm pada 5% transaksi normal.
# Pertanyaan: kalau alarm berbunyi, berapa peluang transaksi itu benar-benar fraud?

# %%
p_fraud, p_alarm_given_fraud, p_alarm_given_ok = 0.01, 0.95, 0.05
p_alarm = p_alarm_given_fraud * p_fraud + p_alarm_given_ok * (1 - p_fraud)
bayes = p_alarm_given_fraud * p_fraud / p_alarm
print(f"Rumus Bayes   P(fraud | alarm) = {bayes:.3f}")

N = 1_000_000
fraud = rng.random(N) < p_fraud
alarm = np.where(fraud, rng.random(N) < p_alarm_given_fraud, rng.random(N) < p_alarm_given_ok)
print(f"Simulasi      P(fraud | alarm) = {fraud[alarm].mean():.3f}")
# Hanya ±16%! Karena fraud langka, sebagian besar alarm berasal dari 99% transaksi normal.
# Ini alasan "precision" rendah pada data tidak seimbang (Week 8 & 15).

# %% [markdown]
# ## 5. Confidence interval: apa arti "95%"?
# Ambil 100 sampel, buat CI 95% dari masing-masing, lalu hitung berapa yang mengandung μ.

# %%
n, reps = 40, 100
fig, ax = plt.subplots(figsize=(10, 4))
covered = 0
for i in range(reps):
    s = population[rng.integers(0, len(population), n)]
    se = s.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, df=n - 1)
    lo, hi = s.mean() - t * se, s.mean() + t * se
    ok = lo <= mu <= hi
    covered += ok
    ax.plot([i, i], [lo, hi], color="tab:blue" if ok else "red")
ax.axhline(mu, color="black", lw=1)
ax.set_title(f"{covered}/{reps} interval mengandung μ (merah = tidak)")
plt.tight_layout(); plt.savefig(out("week04_3_ci.png")); plt.show()
# "95%" adalah sifat METODE-nya: dari banyak sampel, ±95% interval menangkap μ.
# Untuk satu interval tertentu, μ ada di dalamnya atau tidak — bukan "95% peluang".

# %% Bootstrap CI: tanpa rumus, cukup resample sampel itu sendiri
s = population[rng.integers(0, len(population), 200)]
boot = np.array([rng.choice(s, len(s), replace=True).mean() for _ in range(5000)])
print(f"Bootstrap 95% CI mean : [{np.percentile(boot, 2.5):.1f}, {np.percentile(boot, 97.5):.1f}]  (mu={mu:.1f})")
boot_med = np.array([np.median(rng.choice(s, len(s), replace=True)) for _ in range(5000)])
print(f"Bootstrap 95% CI median: [{np.percentile(boot_med, 2.5):.1f}, {np.percentile(boot_med, 97.5):.1f}]")

# %% [markdown]
# ## 6. Uji hipotesis dan A/B test
# Tombol lama (A) konversi 10%. Tombol baru (B) benar-benar 11%. 5000 user per grup.
# H0: tidak ada beda. p-value = peluang melihat selisih SEBESAR INI (atau lebih) KALAU H0 benar.

# %%
n_a = n_b = 5000
a = rng.random(n_a) < 0.10
b = rng.random(n_b) < 0.11
diff = b.mean() - a.mean()
print(f"Konversi A={a.mean():.3%}  B={b.mean():.3%}  selisih={diff:.3%}")

# (a) Uji permutasi: kalau label A/B tidak berarti apa-apa (H0), acak labelnya berkali-kali
pooled = np.concatenate([a, b])
perm_diffs = np.empty(10_000)
for i in range(len(perm_diffs)):
    rng.shuffle(pooled)
    perm_diffs[i] = pooled[:n_b].mean() - pooled[n_b:].mean()
p_perm = (np.abs(perm_diffs) >= abs(diff)).mean()

# (b) Uji z dua proporsi (rumus)
p_pool = pooled.mean()
z = diff / np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
p_z = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"p-value permutasi={p_perm:.4f} | p-value uji-z={p_z:.4f}")

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(perm_diffs, bins=60, color="gray")
ax.axvline(diff, color="red", label="selisih yang diamati")
ax.axvline(-diff, color="red", ls="--")
ax.set_title("Distribusi selisih KALAU H0 benar (permutasi)"); ax.legend()
plt.tight_layout(); plt.savefig(out("week04_4_permutation.png")); plt.show()

# %% Kalau H0 benar, p-value tersebar merata -> 5% eksperimen "signifikan" padahal tidak ada efek
p_values = []
for _ in range(2000):
    a0 = rng.random(2000) < 0.10
    b0 = rng.random(2000) < 0.10     # SAMA persis, tidak ada efek
    pp = np.concatenate([a0, b0]).mean()
    zz = (b0.mean() - a0.mean()) / np.sqrt(pp * (1 - pp) * (2 / 2000))
    p_values.append(2 * (1 - stats.norm.cdf(abs(zz))))
p_values = np.array(p_values)
print(f"Tanpa efek nyata, {np.mean(p_values < 0.05):.1%} uji tetap memberi p<0.05 (false positive)")

# %% Power: berapa user yang dibutuhkan untuk mendeteksi 10% -> 11%?
def power(n, p1=0.10, p2=0.11, reps=1000):
    hits = 0
    for _ in range(reps):
        x1, x2 = rng.binomial(n, p1), rng.binomial(n, p2)
        pp = (x1 + x2) / (2 * n)
        zz = (x2 - x1) / n / np.sqrt(pp * (1 - pp) * 2 / n)
        hits += 2 * (1 - stats.norm.cdf(abs(zz))) < 0.05
    return hits / reps

for n in [1000, 5000, 15000]:
    print(f"n={n:>6} per grup -> power ~ {power(n):.0%}")
# Dengan 5000/grup, efek nyata 1 poin persen sering TIDAK terdeteksi. Hitung ukuran sampel SEBELUM eksperimen.
