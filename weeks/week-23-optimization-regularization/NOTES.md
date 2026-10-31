# Week 23 — Optimisasi dan Regularisasi

## File dan cara menjalankan
`uv run python weeks/week-23-optimization-regularization/optim_regularization.py` (±7 menit di CPU) → `outputs/week23_*.png`.
Loop training dipindah ke [src/mlbootcamp/torch_utils.py](../../src/mlbootcamp/torch_utils.py) (`fit()` dengan early stopping). FashionMNIST dimuat langsung sebagai tensor di memori, sehingga ±20× lebih cepat daripada transform per gambar.
Setiap eksperimen mengubah **satu** faktor: MLP 784→512→512→10, seed 42, 10.000 gambar train, validasi 10.000.

## 1. Learning rate (Adam, 10 epoch)
| lr | Best val loss | Val acc | Catatan |
|---|---|---|---|
| 1e-4 | 0,426 (epoch 9) | 0,848 | Masih turun di akhir → terlalu lambat |
| **1e-3** | **0,409** (epoch 5) | 0,852 | Cepat; setelah epoch 5 val loss naik (mulai overfit) |
| 1e-2 | 0,478 | 0,838 | Bergejolak |
| 1e-1 | 14,4 (epoch 1) | 0,318 | **Divergen**: train acc 0,14 (≈ tebak acak) |
Learning rate adalah hyperparameter terpenting. Mulai dari default (Adam 1e-3), lalu coba ×10 / ÷10.

## 2. Batch size (lr sama, 10 epoch)
| Batch | Best val loss | Val acc | Detik/epoch |
|---|---|---|---|
| 16 | 0,427 | 0,859 | 5,5 |
| 128 | 0,409 | 0,852 | 1,6 |
| 1024 | 0,423 | 0,851 | 1,1 |
- Batch kecil = banyak update per epoch tapi gradiennya *noisy*, dan lambat (overhead per langkah).
- Batch besar = gradien halus, cepat per epoch (perangkat terpakai penuh), tapi update per epoch lebih sedikit. Biasanya perlu learning rate lebih besar untuk mengimbangi (train acc batch 1024 hanya 0,885 vs 0,920).
- 32–256 adalah titik awal yang aman.

## 3. SGD vs Adam (10 epoch)
| Optimizer | Best val loss | Val acc |
|---|---|---|
| SGD lr=0,01 | 0,601 | 0,779 |
| SGD + momentum 0,9 | 0,415 | 0,854 |
| Adam lr=1e-3 | **0,409** | 0,852 |
SGD polos lambat. Momentum hampir menyamai Adam. Adam = default yang praktis; SGD+momentum dengan tuning yang baik kadang menggeneralisasi sedikit lebih baik di model besar (misalnya CNN untuk gambar).

## 4. Regularisasi — membuat overfit dulu, lalu memperbaikinya
2.000 gambar train, model 0,67 juta parameter, 60 epoch:
| Eksperimen | Best val loss | **Val loss akhir** | Train acc akhir | Val acc akhir | Gap acc |
|---|---|---|---|---|---|
| Tanpa regularisasi | 0,528 (epoch 6) | **1,092** | 1,000 | 0,835 | 0,165 |
| Dropout 0,5 | **0,497** (epoch 12) | 0,715 | 0,951 | 0,832 | 0,119 |
| Weight decay 0,1 (AdamW) | 0,526 | 0,957 | 1,000 | 0,837 | 0,163 |
| Early stopping (patience 5) | 0,528 | **0,530** | 0,940 | 0,830 | 0,110 |
| Lebih banyak data (20.000) | **0,370** | 0,992 | 0,993 | **0,875** | 0,118 |

### Diagnosis overfitting dengan bukti (exit criteria)
Tanpa regularisasi:
1. **Train acc = 100%**: model menghafal 2.000 gambar.
2. **Val loss turun sampai epoch 6 (0,528), lalu naik terus sampai 1,092**, sementara train loss terus turun. Kurva train dan val **berpisah** (`week23_4_regularization.png`).
3. Gap akurasi train–val 16,5 poin.
Val **accuracy** hampir tidak turun (0,835), tapi val **loss** naik dua kali lipat: model menjadi *terlalu yakin* pada prediksi yang salah. Karena itu pantau loss, bukan hanya accuracy.

### Cara kerja setiap teknik
- **Dropout**: saat training, setiap neuron dimatikan acak dengan peluang p. Model tidak bisa bergantung pada satu jalur saja → terpaksa mempelajari fitur yang lebih umum. Otomatis mati saat `model.eval()`. Efek di sini paling besar: val loss terbaik 0,497 dan kenaikannya jauh lebih kecil.
- **Weight decay**: menghukum bobot besar (L2, sama dengan Ridge di Week 10). Di **AdamW**, peluruhannya dikali learning rate (0,1 × 1e-3 per langkah), sehingga efeknya kecil di sini. Perlu nilai lebih besar atau lebih banyak epoch.
- **Early stopping**: hentikan kalau val loss tidak membaik selama `patience` epoch, lalu kembalikan bobot terbaik. Tidak membuat model lebih baik, tapi mencegahnya menjadi lebih buruk (val loss akhir 0,530 vs 1,092), dan menghemat waktu (11 vs 60 epoch).
- **Lebih banyak data**: obat paling ampuh (val loss 0,370, val acc 0,875). Dengan 20.000 gambar, model tetap overfit setelah epoch 4, jadi teknik lain tetap diperlukan.

Kombinasi yang umum: dropout + weight decay + early stopping + augmentasi data.

## Analogi untuk orang awam
- **Learning rate** → menyetir sambil mencari parkir: terlalu pelan tidak sampai-sampai, terlalu cepat bablas melewati tempat parkirnya.
- **Batch size** → mengambil keputusan berdasarkan pendapat 16 orang (cepat berubah, ramai) atau 1.024 orang (stabil, tapi jarang berganti arah).
- **Overfitting** → aktor yang menghafal naskah persis untuk satu penonton, lalu kebingungan saat penontonnya bertanya hal lain.
- **Dropout** → latihan tim sepak bola dengan pemain yang diacak absen: setiap pemain terpaksa bisa bermain di berbagai posisi, tidak bergantung pada satu bintang.
- **Weight decay** → pajak untuk bobot yang terlalu besar: model didorong memakai solusi yang sederhana.
- **Early stopping** → berhenti memanggang kue ketika sudah matang, sebelum gosong; dan simpan kue di momen terbaiknya.

## Exit criteria — tes diri
- [ ] Dari plot `week23_4_regularization.png`, tunjukkan titik mulai overfit dan jelaskan buktinya (train vs val loss, gap akurasi).
- [ ] Jelaskan kenapa val loss bisa naik walaupun val accuracy hampir tetap.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
