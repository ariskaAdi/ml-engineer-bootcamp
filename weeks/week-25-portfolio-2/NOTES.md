# Week 25 — Portfolio Project 2: Deep Learning

Kode: [projects/p2-fashion-cnn/](../../projects/p2-fashion-cnn/) — baca [README-nya](../../projects/p2-fashion-cnn/README.md).

## Hasil singkat
CNN dari nol, 8 epoch di CPU (±35 menit): **test accuracy 0,929** vs MLP Week 22 0,876. Kelas tersulit: Shirt (75,6%), sering tertukar dengan T-shirt.

## Kenapa CNN lebih baik daripada MLP untuk gambar
MLP meratakan gambar menjadi 784 angka. Informasi "piksel ini bertetangga dengan piksel itu" hilang, dan pola yang sama di posisi berbeda harus dipelajari ulang. Konvolusi memakai filter kecil yang **sama** di seluruh gambar → belajar pola lokal (tepi, tekstur, kerah) dengan parameter jauh lebih sedikit, dan tetap mengenali pola walaupun posisinya bergeser.

## Komponen baru dibanding Week 22–23
| Komponen | Fungsi |
|---|---|
| `nn.Conv2d(c_in, c_out, 3, padding=1)` | 3×3 filter; `padding=1` menjaga ukuran 28×28 |
| `nn.BatchNorm2d` | Menormalisasi aktivasi per batch → training lebih stabil, lr bisa lebih besar. Perilakunya berbeda di `train()` vs `eval()` (memakai statistik tersimpan) |
| `nn.MaxPool2d(2)` | Ambil maksimum tiap 2×2 → ukuran setengah, lebih tahan pergeseran kecil |
| **OneCycleLR** | lr naik lalu turun selama training. Biasanya konvergen lebih cepat daripada lr tetap. Di script, `scheduler.step()` dipanggil **setiap batch** |
| **Augmentasi** flip horizontal | Data train "baru" setiap epoch tanpa mengumpulkan data → mengurangi overfit. Hanya untuk train, tidak untuk val/test |
| `argparse` + `@dataclass Config` | Semua setting bisa diubah dari CLI dan tersimpan otomatis di `metrics.json` |
| `history.csv` ditulis setiap epoch | Kalau training terputus, metrik sampai epoch terakhir tetap ada |

## Pemisahan training dan inference
- `model.py`: arsitektur + `preprocess_image()`. Dipakai bersama oleh training dan inference.
- `train.py`: hanya training.
- `predict.py`: hanya inference; memuat `model.pt` (bobot + config) dan membangun ulang arsitektur dari config.
Pemisahan ini menyiapkan Phase 6: service inference cukup mengimpor `model.py` + `predict.py`, tanpa kode training.

## Analogi untuk orang awam
- **CNN vs MLP** → MLP seperti membaca foto yang sudah dirobek menjadi 784 potongan acak; CNN melihat foto utuh dengan kaca pembesar kecil yang digeser ke seluruh bagian.
- **Augmentasi (flip)** → melatih kenalan mengenali wajahmu dari foto biasa dan foto cermin. Data "baru" tanpa memotret ulang.
- **BatchNorm** → menyamakan volume semua mikrofon sebelum rekaman, supaya tidak ada yang terlalu keras atau terlalu pelan.
- **Pemisahan train.py / predict.py** → dapur produksi dan meja kasir: pelanggan hanya butuh kasir, bukan seluruh dapur.

## Exit criteria — tes diri
- [ ] Jalankan ulang training dari README (atau versi cepat `--epochs 1 --train-size 5000`) dan inference pada gambar sampel.
- [ ] Jelaskan setiap baris loop `run_epoch` di `train.py`, termasuk kenapa `torch.set_grad_enabled(training)`.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
