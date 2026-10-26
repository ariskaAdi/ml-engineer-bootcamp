# Track A (Computer Vision) — Week 27–29, materi tambahan

> Track utama bootcamp ini adalah NLP (Track B). Folder ini supaya kamu juga paham dasar computer vision. Portfolio gambar sudah ada di P2 (FashionCNN).

## File dan cara menjalankan
```powershell
uv run python weeks/week-27-29-track-a-cv/01_image_cnn_concepts.py      # <1 menit
uv run python weeks/week-27-29-track-a-cv/02_transfer_learning.py       # unduh CIFAR-10 170 MB + ResNet18 45 MB; ±15 menit CPU
uv run python weeks/week-27-29-track-a-cv/03_opencv_error_analysis.py   # butuh model dari 02
```
Plot: `outputs/trackA_*.png`. Di Google Colab (GPU) script 02 selesai ±2 menit.

---

## Week 27 — Gambar sebagai angka dan konsep CNN (`01`)
**Dari gambar mentah ke input model (exit criteria):**
1. Gambar = array piksel. Grayscale `(H, W)` nilai 0–255; warna `(H, W, 3)` dengan channel R, G, B.
2. PyTorch butuh `(C, H, W)` float → `permute(2, 0, 1)`, lalu bagi 255 → 0–1.
3. **Resize** ke ukuran tetap (model butuh ukuran input konsisten). Memperkecil membuang detail; memperbesar tidak menambah detail.
4. **Normalisasi** `(x − mean) / std` per channel, dengan mean/std **dataset training** (atau ImageNet untuk model pretrained). Untuk satu gambar hasilnya tidak tepat 0/1 (mis. mean 0,27); yang mendekati 0/1 adalah rata-rata seluruh dataset.
5. Gabungkan menjadi batch `(N, C, H, W)`.

**Konvolusi:** filter kecil (3×3) digeser ke seluruh gambar; setiap posisi = jumlah perkalian elemen. Script menghitung satu nilai dengan tangan dan hasilnya sama dengan `F.conv2d` (0,0078).
- Sobel X mendeteksi tepi vertikal, Sobel Y tepi horizontal, filter rata-rata mengaburkan (`trackA_2_conv_filters.png`).
- **Max pooling 2×2**: ambil nilai maksimum → ukuran setengah, lebih tahan pergeseran kecil.
- CNN tidak memakai filter buatan tangan; ia **mempelajari** filternya (`trackA_3_learned_filters.png` = 32 filter layer pertama P2).
- Hemat parameter: Linear 784→128 = 100.352 parameter; 32 filter konvolusi 3×3 = 320 parameter (bobot yang sama dipakai di seluruh gambar).
- Layer awal mempelajari tepi/warna; layer dalam menggabungkannya menjadi tekstur, bagian objek, lalu objek.

---

## Week 28 — TorchVision dan transfer learning (`02`)
Data: subset CIFAR-10 (5.000 train / 1.000 val dari train resmi / 2.000 test dari file test resmi), di-resize ke 96 px. 3 epoch setiap model.

| Model | Test acc | Waktu CPU | Parameter yang dilatih |
|---|---|---|---|
| A. CNN kecil dari nol | 0,309 | 324 s | 94.986 |
| B. ResNet18 pretrained **beku** + kepala baru | 0,644 | 258 s | **5.130** |
| C. ResNet18 **fine-tune** layer4 + kepala | **0,805** | 255 s | 8,4 juta |

- **Transfer learning**: ResNet18 sudah belajar fitur visual umum dari 1,2 juta gambar ImageNet. Kita ganti kepala `fc` (1000 kelas → 10 kelas) dan melatih sedikit bagian saja.
- **Beku (feature extraction)**: hanya kepala yang belajar (5 ribu parameter). Murah, tahan overfit, cocok untuk data sangat sedikit. Dengan hanya 5.000 gambar, hasilnya sudah 2× lebih baik daripada CNN dari nol.
- **Fine-tune**: blok terakhir ikut belajar dengan learning rate kecil (3e-4) → fitur menyesuaikan dengan gambar CIFAR kecil. Lompatan terbesar (+16 poin).
- Dengan data sedikit, melatih dari nol hampir selalu kalah. Dengan data sangat banyak dan domain yang sangat berbeda (misalnya citra medis), selisihnya mengecil.
- **Normalisasi ImageNet** (`mean=[0.485, 0.456, 0.406]`) wajib untuk model pretrained ImageNet, karena itu distribusi yang dilihatnya saat pretraining.

**Augmentasi** (`RandomCrop`, `RandomHorizontalFlip`) hanya untuk **train**. Val/test memakai transform deterministik. Augmentasi harus masuk akal untuk domainnya: flip horizontal aman untuk kucing/mobil, tapi tidak untuk angka "2"/"5" atau teks.

**Langkah anti-leakage (exit criteria):**
- Indeks train dan val dicek tidak beririsan (`assert`); test berasal dari file test resmi yang terpisah.
- Hyperparameter dan pemilihan model dilihat dari val; test hanya untuk tabel akhir.
- Augmentasi tidak dipakai di val/test.
- Untuk data sungguhan, waspadai gambar yang hampir identik (foto beruntun, objek yang sama) di train dan test. Pisahkan per objek/sesi (`GroupKFold`).

---

## Week 29 — OpenCV, preprocessing konsisten, analisis error (`03`)
1.000 gambar test disimpan sebagai PNG, lalu dibaca dengan OpenCV, seperti data yang datang ke sistem produksi.

**OpenCV membaca warna sebagai BGR, bukan RGB.**
| Preprocessing saat inference | Accuracy |
|---|---|
| Satu fungsi `preprocess()` (BGR→RGB, resize, normalisasi ImageNet) | **0,786** |
| Lupa `cvtColor(BGR→RGB)` | 0,673 |
Tidak ada error, model tetap "jalan", tapi **diam-diam 11 poin lebih buruk**. Inilah alasan preprocessing training dan inference harus berasal dari satu fungsi/kode yang sama (exit criteria), dan perlu tes yang membandingkan hasilnya.
Catatan: 0,786 sedikit di bawah 0,805 dari script 02. Selain subset gambar yang berbeda, training memakai resize PIL (dengan antialiasing) sedangkan inference memakai `cv2.resize`. Ini contoh *training-serving skew* yang halus. Solusi terbaik: pakai library resize yang sama di keduanya.

**Analisis error per kelas** (1.000 gambar):
| Kelas | Recall | Catatan |
|---|---|---|
| ship | 0,886 | Latar laut membantu (dan bisa menjadi "jalan pintas") |
| airplane | 0,920 | Sering tertukar dengan burung/kapal karena latar langit/laut |
| **cat** | **0,404** | Kelas terlemah |
Pasangan paling sering tertukar: cat → dog (21), cat → frog (15), automobile ↔ truck (13), cat → bird (12), deer → bird (10).
Gambar 32×32 yang diperbesar ke 96 px sangat buram; kucing dan anjing hanya dibedakan detail kecil. Lihat `trackA_5_misclassified.png` (12 kesalahan dengan keyakinan tertinggi). Biasanya ditemukan: gambar terpotong, objek kecil, atau label yang memang meragukan.

Ide perbaikan yang bisa diuji: resolusi input lebih besar, augmentasi lebih kuat, lebih banyak data kucing/anjing, fine-tune lebih banyak layer, atau model yang lebih besar (ResNet50/EfficientNet) di GPU.

## Opsional setelah track ini
Object detection (YOLO), segmentasi. Tidak wajib untuk semua posisi ML engineer.

## Exit criteria — tes diri
- [ ] Jelaskan langkah dari file JPG sampai tensor `(1, 3, 96, 96)` yang siap masuk model.
- [ ] Jelaskan beda model beku dan fine-tune, dan kapan memilih masing-masing.
- [ ] Jelaskan kenapa bug BGR/RGB berbahaya walaupun tidak menimbulkan error.
