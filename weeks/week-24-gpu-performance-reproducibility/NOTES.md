# Week 24 — Performa GPU dan Reproducibility

## File dan cara menjalankan
`uv run python weeks/week-24-gpu-performance-reproducibility/performance_reproducibility.py` (±1 menit).
Lalu jalankan **juga di Google Colab dengan T4 GPU** dan isi tabel "GPU" di bawah. Script otomatis mendeteksi CUDA dan menyimpan `models/week24_env_<device>.json`.

## 1. Mengukur waktu dengan benar
| Cara | Hasil (CPU laptop, matmul 2048²) |
|---|---|
| Naif: sekali, tanpa warm-up | 109 ms |
| Benar: warm-up + 5× ulang + median | 97,6 ± 8,1 ms |
Di GPU perbedaannya jauh lebih ekstrem: GPU **asinkron**. `A @ B` hanya *mengantrikan* pekerjaan lalu langsung kembali. Tanpa `torch.cuda.synchronize()`, timer mengukur waktu mengantri (±0,05 ms), bukan waktu menghitung.
Aturan: warm-up → synchronize → ulangi beberapa kali → laporkan median ± std.

## 2. Satu langkah training CNN kecil
| Device | Batch 8 | Batch 128 | Batch 1024 |
|---|---|---|---|
| CPU laptop (4 thread) | 10,4 ms (766 img/s) | 81,9 ms (1.563 img/s) | 951 ms (1.077 img/s) |
| GPU T4 (isi dari Colab) | | | |

Yang diharapkan di GPU: batch 8 hampir tidak lebih cepat daripada CPU (overhead peluncuran kernel mendominasi), sedangkan batch besar bisa puluhan kali lebih cepat, karena ribuan core GPU baru terpakai penuh kalau pekerjaannya cukup besar. Di CPU, throughput justru turun di batch 1024 (data tidak muat di cache).

## 3. Bottleneck data loading
| Cara memuat data | ms per batch 128 |
|---|---|
| PIL + `transform` per gambar, `num_workers=0` | **58** |
| Tensor sudah di memori (`TensorDataset`) | **2,2** |
Satu langkah training CNN di batch 128 butuh ±82 ms di CPU. Di GPU mungkin hanya ±2 ms. Artinya, di GPU, loading PIL 58 ms membuat GPU **menganggur ±95% waktu** menunggu data. Gejalanya: utilisasi GPU rendah di `nvidia-smi` walaupun training "lambat".

Cara mengatasi:
- `num_workers > 0` di DataLoader: loading berjalan paralel di proses lain sementara GPU menghitung.
- `pin_memory=True` + `.to(device, non_blocking=True)`: penyalinan ke GPU lebih cepat.
- Pra-proses sekali dan simpan sebagai tensor/array (seperti `load_fashion_tensors`).
- Augmentasi di GPU.

**Jebakan Windows:** `num_workers > 0` memakai metode *spawn*, sehingga setiap proses worker **meng-import ulang seluruh script**. Di percobaan pertama, semua benchmark ikut berjalan ulang di 4 worker (hasilnya 639 ms/batch dan output tercetak dua kali). Solusinya: semua kode training dibungkus di `if __name__ == "__main__":`. Demo num_workers di script hanya berjalan di Linux/Colab.

## 4. Memori
- Parameter SmallCNN: 421.642 × 4 byte = **1,7 MB**.
- Batch input 1024 gambar: 3,2 MB.
- Saat training, GPU juga menyimpan **aktivasi** setiap layer (diperlukan untuk backward), gradien (sama besar dengan parameter), dan state Adam (2× parameter). Aktivasi biasanya yang paling besar dan bertambah seiring ukuran batch. Kalau `CUDA out of memory` → kecilkan batch, pakai mixed precision (`torch.autocast`), atau *gradient accumulation*.
- Di Colab: `torch.cuda.max_memory_allocated()` (dicetak oleh script) dan `!nvidia-smi`.

## 5. Reproducibility
`set_seed(1)` dua kali → loss **identik** (2,0958707...). Seed 2 → berbeda (2,2100).
Yang perlu di-seed: `random`, `numpy`, `torch` (CPU & CUDA), urutan `DataLoader` (`generator`), dan split data.
Di GPU, beberapa kernel cuDNN tidak deterministik (urutan penjumlahan floating point berbeda → selisih ±1e-6 yang bisa membesar). Untuk hasil identik: `torch.use_deterministic_algorithms(True)` dan `torch.backends.cudnn.benchmark = False` (lebih lambat).
Selalu catat: versi Python, torch, CUDA, cuDNN, nama GPU/CPU, seed, hyperparameter (`week24_env_*.json`).

## 6. Kapan perlu menulis kode CUDA sendiri?
**Hampir tidak pernah** untuk ML engineer. PyTorch sudah memanggil kernel CUDA/cuDNN/cuBLAS yang sangat dioptimalkan. Urutan yang benar kalau training lambat:
1. Ukur dulu: apakah bottleneck di data loading, transfer CPU↔GPU, atau perhitungan?
2. Perbaiki data pipeline, naikkan batch, pakai mixed precision (`torch.autocast`), `torch.compile`.
3. Kernel kustom (CUDA C++/Triton) hanya untuk operasi baru yang belum ada di library, atau untuk riset/infra performa tinggi. Itu pekerjaan spesialis.

## Exit criteria — tes diri
- [ ] Jelaskan perbedaan kecepatan CPU vs GPU yang kamu ukur di Colab, dan 2 keterbatasan pengukuranmu (misalnya GPU Colab berbagi dengan pengguna lain, ukuran model sangat kecil, waktu loading tidak termasuk).
- [ ] Jelaskan kenapa GPU bisa menganggur walaupun model berjalan di GPU.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
