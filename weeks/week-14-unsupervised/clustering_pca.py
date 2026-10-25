# %% [markdown]
# # Week 14 — Unsupervised Learning: K-Means dan PCA
#
# Tidak ada target y. Kita mencari struktur di data pelanggan toko.
# Dataset punya kolom `true_segment` yang DISEMBUNYIKAN dari model — hanya untuk mengecek.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from mlbootcamp.datasets import make_customers
from mlbootcamp.paths import out

df = make_customers()
features = ["annual_spend", "visits_per_month", "avg_basket", "days_since_last", "pct_online"]
X = df[features]
print(X.describe().round(2))

# %% 1. Kenapa scaling penting untuk algoritma berbasis jarak
# avg_basket (puluhan-ratusan) vs pct_online (0-1): tanpa scaling, jarak hampir hanya ditentukan avg_basket.
X_scaled = StandardScaler().fit_transform(X)
for name, data in [("TANPA scaling", X.to_numpy()), ("DENGAN scaling", X_scaled)]:
    labels = KMeans(4, n_init=10, random_state=0).fit_predict(data)
    print(f"{name:<15} ARI vs segmen asli = {adjusted_rand_score(df['true_segment'], labels):.3f}")
# ARI (Adjusted Rand Index): 1 = sama persis dengan segmen asli, 0 = seperti acak.
# Di dunia nyata kita TIDAK punya segmen asli; ini hanya mungkin karena datanya sintetis.

# %% 2. Memilih k: elbow (inertia) dan silhouette
ks = range(2, 10)
inertias, sils = [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=0).fit(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled, km.labels_))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(ks, inertias, "o-"); axes[0].set(xlabel="k", ylabel="inertia", title="Elbow: inertia selalu turun")
axes[1].plot(ks, sils, "o-"); axes[1].set(xlabel="k", ylabel="silhouette", title="Silhouette: makin tinggi makin terpisah")
plt.tight_layout(); plt.savefig(out("week14_1_choose_k.png")); plt.show()
print({k: round(s, 3) for k, s in zip(ks, sils)})

# %% 3. Profil cluster: ARTI setiap cluster harus dibaca manusia
k = 4
km = KMeans(k, n_init=10, random_state=0).fit(X_scaled)
df["cluster"] = km.labels_
profile = df.groupby("cluster")[features].median().round(2)
profile["n"] = df["cluster"].value_counts().sort_index()
print(profile.to_string())
print(pd.crosstab(df["cluster"], df["true_segment"]))

# %% 4. K-Means tidak stabil: hasil bisa bergantung pada inisialisasi
for seed in range(3):
    lab = KMeans(4, n_init=1, init="random", random_state=seed).fit_predict(X_scaled)
    print(f"seed={seed}: ARI vs run pertama = {adjusted_rand_score(km.labels_, lab):.3f}")
# n_init=10 menjalankan 10 inisialisasi dan memilih yang inertianya terkecil.

# %% 5. PCA: 5 dimensi -> 2 dimensi untuk visualisasi
pca = PCA().fit(X_scaled)
print("explained variance ratio:", pca.explained_variance_ratio_.round(3))
print("kumulatif:", pca.explained_variance_ratio_.cumsum().round(3))
loadings = pd.DataFrame(pca.components_[:2].T, index=features, columns=["PC1", "PC2"]).round(2)
print("loadings (kontribusi fitur ke setiap komponen):\n", loadings)

Z = pca.transform(X_scaled)[:, :2]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, col, title in [(axes[0], "cluster", "Cluster K-Means"), (axes[1], "true_segment", "Segmen asli (tersembunyi)")]:
    for val in sorted(df[col].unique()):
        m = df[col] == val
        ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.6, label=str(val))
    ax.set(xlabel="PC1", ylabel="PC2", title=title); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(out("week14_2_pca.png")); plt.show()

# %% 6. Bahaya: K-Means SELALU menghasilkan k cluster, bahkan pada data tanpa struktur
noise = np.random.default_rng(0).uniform(size=(800, 5))
lab_noise = KMeans(4, n_init=10, random_state=0).fit_predict(noise)
print(f"Data acak seragam: K-Means tetap memberi 4 cluster, ukuran {np.bincount(lab_noise)}, "
      f"silhouette={silhouette_score(noise, lab_noise):.3f} (rendah)")
