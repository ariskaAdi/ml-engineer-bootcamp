# Week 20 — Tensor dan Dasar GPU

## File dan cara menjalankan
`uv run python weeks/week-20-tensors-gpu/tensors_gpu.py`
Laptop ini memakai PyTorch CPU (`torch 2.x+cpu`, tanpa GPU NVIDIA). Untuk bagian GPU: upload file ke **Google Colab**, pilih *Runtime → Change runtime type → T4 GPU*, lalu jalankan. Kode otomatis mendeteksi CUDA.

## Tensor = array NumPy + 2 kemampuan tambahan
1. Bisa berjalan di **GPU**.
2. Bisa menghitung **gradien otomatis** (autograd, Week 21).

| Konsep | Catatan |
|---|---|
| `shape` | Gambar: `(N, C, H, W)` = batch, channel, tinggi, lebar. Tabular: `(N, fitur)`. Teks: `(N, panjang_token)` |
| `dtype` | `float32` default untuk bobot & input. Label klasifikasi `int64` (`long`). `float64` dari NumPy sering membuat error "expected Float but found Double" → `.float()` |
| Indexing & broadcasting | Aturannya sama dengan NumPy (Week 1) |
| `view`/`reshape`, `unsqueeze`/`squeeze`, `permute`, `flatten` | Operasi shape harian. `permute(2,0,1)`: HWC (OpenCV/PIL) → CHW (PyTorch) |
| `torch.from_numpy` | **Berbagi memori** dengan array NumPy; `torch.tensor(arr)` menyalin |

Normalisasi per channel: mean `(C,)` di-reshape menjadi `(1, C, 1, 1)` supaya bisa di-broadcast ke `(N, C, H, W)`.

## CPU vs GPU
- CPU: sedikit core yang sangat pintar → cocok untuk logika bercabang dan tugas berurutan.
- GPU: ribuan core sederhana → cocok untuk operasi yang sama pada data sangat banyak secara paralel (perkalian matriks besar, konvolusi).
- Data harus disalin dari RAM ke memori GPU (VRAM) lewat PCIe, dan penyalinan ini lambat dibandingkan perhitungannya.

## Device
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)          # parameter pindah
x, y = x.to(device), y.to(device) # data pindah — setiap batch!
```
**Device mismatch** (exit criteria): `RuntimeError: Expected all tensors to be on the same device...` atau (di demo CPU) `Tensor on device cpu is not on the expected device meta!`.
Diagnosis: cetak `x.device` dan `next(model.parameters()).device`, cari `.to(device)` yang terlupa (biasanya pada data di dalam loop, atau tensor baru yang dibuat dengan `torch.zeros(...)` tanpa `device=`).

## Kenapa workload kecil tidak lebih cepat di GPU (exit criteria)
Hasil CPU laptop ini: matmul 16×16 = 0,013 ms; 256×256 = 0,18 ms; 2048×2048 = 80 ms.
Di GPU, setiap operasi punya **overhead tetap** (meluncurkan kernel ±5–20 µs, menyalin data CPU↔GPU). Untuk matriks 16×16, overhead jauh lebih besar daripada waktu hitungnya, sehingga GPU bisa lebih lambat daripada CPU. Untuk 2048×2048, perhitungan mendominasi, dan GPU bisa puluhan kali lebih cepat. Kalau data disalin setiap kali (kolom "GPU+copy"), keuntungannya berkurang banyak.

Cara mengukur dengan benar:
1. **Warm-up** dulu (panggilan pertama termasuk alokasi memori & inisialisasi).
2. **`torch.cuda.synchronize()`** sebelum menghentikan timer. GPU bekerja asinkron: tanpa sinkronisasi, timer berhenti sebelum GPU selesai dan hasilnya tampak "sangat cepat" padahal salah.
3. Ulangi beberapa kali dan ambil median.

Isi setelah menjalankan di Colab:
| Ukuran | CPU (ms) | GPU (ms) | GPU + copy (ms) |
|---|---|---|---|
| 16 | 0,013 | | |
| 256 | 0,182 | | |
| 2048 | 80,1 | | |

## Analogi untuk orang awam
- **Tensor** → spreadsheet yang bisa bertumpuk ke banyak dimensi (tabel → buku → rak buku), plus kemampuan dihitung di GPU.
- **CPU vs GPU** → CPU seperti beberapa profesor jenius; GPU seperti ribuan siswa SD. Untuk 10.000 soal penjumlahan sederhana, ribuan siswa jauh lebih cepat. Untuk satu soal rumit, profesor lebih cocok.
- **Device mismatch** → satu chef di dapur A, bahan masakannya di dapur B. Masakan tidak bisa dibuat sampai bahan dan chef berada di dapur yang sama.
- **Workload kecil tidak lebih cepat di GPU** → memanggil truk pengiriman untuk mengirim satu amplop: waktu persiapan truknya lebih lama daripada mengantar sendiri dengan motor.

## Exit criteria — tes diri
- [ ] Sengaja buat device mismatch, baca pesan error-nya, dan perbaiki.
- [ ] Jelaskan kenapa matmul 16×16 tidak lebih cepat di GPU.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
