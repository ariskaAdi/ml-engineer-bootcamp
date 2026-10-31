# Week 14 — Unsupervised Learning

## File dan cara menjalankan
`uv run python weeks/week-14-unsupervised/clustering_pca.py` → `outputs/week14_*.png`.
Data: `make_customers()` — 800 pelanggan, 5 fitur perilaku. Ada kolom `true_segment` yang **tidak** diberikan ke model. Kolom ini hanya ada karena datanya sintetis, supaya kita bisa menilai hasil clustering. Di dunia nyata kolom seperti ini tidak ada.

## K-Means
1. Pilih k titik pusat (centroid) awal.
2. Setiap titik masuk ke centroid terdekat.
3. Centroid dipindah ke rata-rata anggotanya.
4. Ulangi 2–3 sampai stabil.

Yang diminimalkan: **inertia** = jumlah kuadrat jarak setiap titik ke centroid-nya.
Asumsi tersembunyi: cluster berbentuk bulat, ukurannya mirip, dan semua fitur sama penting (karena memakai jarak Euclid).

## Kenapa scaling penting
| | ARI vs segmen asli |
|---|---|
| Tanpa scaling | 0,395 |
| Dengan StandardScaler | **0,990** |
`avg_basket` bernilai puluhan–ratusan, sedangkan `pct_online` 0–1. Tanpa scaling, jarak antar pelanggan hampir hanya ditentukan oleh `avg_basket`. Segmen "online_only" (yang dibedakan oleh `pct_online`) jadi tidak terlihat. Berlaku untuk semua algoritma berbasis jarak: K-Means, KNN, SVM, PCA.

## Memilih k
- **Elbow**: inertia selalu turun kalau k ditambah (k = n → inertia 0), jadi cari "siku" tempat penurunan melambat. Sering ambigu.
- **Silhouette** (−1 sampai 1): seberapa dekat titik ke cluster-nya sendiri dibandingkan ke cluster tetangga. Hasil: k=4 tertinggi (0,584).
- Keputusan akhir juga harus mempertimbangkan **bisnis**: apakah tim marketing bisa menangani 4 kampanye berbeda?

## Profil cluster (k=4) — inilah yang dibaca manusia
| Cluster | Spend (jt/th) | Kunjungan/bln | Keranjang (rb) | Hari sejak terakhir | % online | n | Nama yang kita berikan |
|---|---|---|---|---|---|---|---|
| 0 | 7,8 | 9,8 | 67 | 4 | 10% | 213 | Pemburu diskon, sering datang |
| 1 | 4,1 | 1,5 | 218 | 45 | 20% | 259 | Jarang belanja, berisiko hilang |
| 2 | 25,0 | 8,1 | 261 | 4 | 30% | 174 | Pelanggan setia bernilai besar |
| 3 | 9,2 | 3,2 | 232 | 13 | 94% | 154 | Online-only |

Nama cluster dibuat **oleh kita** dari tabel profil, bukan oleh algoritma.

## Kenapa cluster ≠ segmen bisnis yang bermakna (exit criteria)
1. **K-Means selalu menghasilkan k cluster, bahkan pada data acak.** Pada data seragam tanpa struktur, kita tetap mendapat 4 cluster "rapi" (195/206/185/214 anggota), hanya silhouette-nya rendah (0,156).
2. **Hasil bergantung pada pilihan kita**: fitur yang dimasukkan, scaling, k, dan algoritma. Pilihan berbeda → "segmen" berbeda.
3. **Tidak stabil**: dengan satu inisialisasi acak, seed=2 memberi pembagian yang berbeda (ARI 0,685). Pakai `n_init` ≥ 10.
4. **Tidak ada "jawaban benar"** untuk dicek. Metrik seperti silhouette hanya mengukur geometri, bukan kegunaan.
5. **Segmen yang berguna harus bisa ditindaklanjuti**: apakah perlakuan berbeda per cluster benar-benar mengubah hasil? Itu hanya bisa dibuktikan dengan eksperimen (A/B test, Week 4).

## PCA
PCA mencari arah baru (komponen) yang menangkap variasi data sebanyak mungkin. Komponen saling tegak lurus dan diurutkan dari yang paling penting.
- Explained variance: PC1 42%, PC2 31,4% → 2 komponen menangkap **73,4%** informasi dari 5 fitur. 3 komponen: 90%.
- Loadings PC1: `visits_per_month` (+0,64), `days_since_last` (−0,53), `avg_basket` (−0,40) → PC1 ≈ "seberapa sering dan baru-baru ini belanja".
- PC2: `annual_spend` (+0,59), `avg_basket` (+0,53), `pct_online` (+0,52) → PC2 ≈ "nilai belanja".
- Kegunaan: visualisasi 2D (`week14_2_pca.png`), mengurangi fitur yang saling berkorelasi, mempercepat model.
- Keterbatasan: komponen sulit dijelaskan ke orang bisnis; PCA linear; **wajib scaling dulu** (kalau tidak, fitur dengan angka terbesar mendominasi PC1).

## Analogi untuk orang awam
- **Clustering** → menyortir tumpukan kancing tanpa label: kamu mengelompokkan berdasarkan warna dan ukuran, tapi nama kelompoknya kamu sendiri yang memberi.
- **K-Means** → menaruh 4 tiang bendera di lapangan, setiap orang berjalan ke tiang terdekat, lalu tiang dipindah ke tengah kerumunannya. Diulang sampai tidak ada yang pindah.
- **Scaling** → membandingkan jarak antar kota dengan satuan campur (meter dan kilometer): angka yang besar mendominasi walaupun tidak lebih penting.
- **Cluster ≠ segmen bisnis** → membagi siswa ke 4 kelompok berdasarkan tinggi badan selalu berhasil, tapi itu tidak berarti ada "4 jenis siswa" yang perlu diajar dengan cara berbeda.
- **PCA** → memotret patung 3D dari sudut terbaik: dari 3 dimensi menjadi foto 2 dimensi, tapi sebagian besar bentuknya tetap terlihat.

## Exit criteria — tes diri
- [ ] Jelaskan kenapa "K-Means menemukan 4 cluster" tidak berarti pelanggan benar-benar terbagi menjadi 4 jenis.
- [ ] Jelaskan apa yang terjadi kalau scaling dilupakan.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
