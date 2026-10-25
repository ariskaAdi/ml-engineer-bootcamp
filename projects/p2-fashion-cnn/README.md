# FashionCNN — Klasifikasi Gambar Pakaian

Portfolio Project 2 (Week 25). CNN PyTorch yang mengenali 10 jenis pakaian dari gambar grayscale 28×28, dilatih dari nol di CPU laptop dalam ±35 menit.

## Data
[FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) (Zalando Research, **lisensi MIT**): 60.000 gambar train + 10.000 test, 10 kelas seimbang. Diunduh otomatis lewat `torchvision`.
Split: 50.000 train / 10.000 validation (dari train resmi, seed 42) / 10.000 test resmi (dipakai sekali di akhir).

## Model
```
[Conv3x3(32) → BN → ReLU] ×2 → MaxPool   28→14
[Conv3x3(64) → BN → ReLU] ×2 → MaxPool   14→7
Flatten → Dropout 0.3 → Linear(3136→128) → ReLU → Dropout 0.3 → Linear(128→10)
```
±0,47 juta parameter. AdamW (lr 2e-3, weight decay 1e-4), OneCycle LR schedule, batch 128, 8 epoch, early stopping (patience 3) berdasarkan val loss, augmentasi flip horizontal.

## Hasil
| Model | Test accuracy |
|---|---|
| MLP Week 22 (baseline, 5 epoch) | 0,876 |
| **FashionCNN (8 epoch)** | **0,929** |

Kurva per epoch (`artifacts/history.csv`): val acc 0,838 → 0,936; train acc 0,942 vs val 0,936 di epoch terakhir → gap kecil, tidak overfit (dropout + augmentasi + BN). Val loss masih turun di epoch 8, jadi lebih banyak epoch kemungkinan masih membantu.

Akurasi per kelas:
| Kelas | Acc | | Kelas | Acc |
|---|---|---|---|---|
| Trouser | 0,989 | | Coat | 0,914 |
| Bag | 0,987 | | Pullover | 0,905 |
| Sandal | 0,979 | | T-shirt/top | 0,888 |
| Sneaker | 0,979 | | **Shirt** | **0,756** |
| Ankle boot | 0,957 | | Dress | 0,938 |

Kesalahan terbanyak: Shirt → T-shirt (111), T-shirt → Shirt (83), Shirt → Coat (59), Pullover ↔ Shirt (43/42). Pada resolusi 28×28 grayscale, kemeja, kaos, pullover, dan coat hanya dibedakan oleh detail kecil (kerah, kancing, panjang lengan), dan beberapa label di dataset aslinya sendiri ambigu.

## Inference
```powershell
uv run python projects/p2-fashion-cnn/src/fashion_cnn/predict.py --export-samples 10      # 10 contoh test -> samples/*.png
uv run python projects/p2-fashion-cnn/src/fashion_cnn/predict.py path/ke/foto.jpg
```
Output: top-3 label + peluang. `preprocess_image()` di `model.py` adalah **satu-satunya** fungsi preprocessing untuk inference: grayscale → pad menjadi persegi → resize 28×28 → **invert kalau latarnya terang** (FashionMNIST = objek terang di latar gelap; foto produk biasanya sebaliknya) → normalisasi dengan mean/std training.

## Reproduksi
```powershell
uv sync
uv run python projects/p2-fashion-cnn/src/fashion_cnn/train.py --epochs 8      # ±35 menit CPU, ±3 menit GPU Colab
uv run python projects/p2-fashion-cnn/src/fashion_cnn/train.py --epochs 1 --train-size 5000   # uji cepat
uv run pytest projects/p2-fashion-cnn                                          # 5 tes
```
Semua hyperparameter adalah argumen CLI (`--help`) dan tersimpan di `artifacts/metrics.json` bersama versi torch & device. Checkpoint `artifacts/model.pt` berisi bobot + config, sehingga arsitektur bisa dibangun ulang saat inference.

## Keterbatasan
- FashionMNIST adalah gambar studio yang sangat bersih dan terpusat. Foto nyata (latar ramai, model manusia, pencahayaan, sudut) akan jauh lebih buruk tanpa data tambahan. Untuk produksi: transfer learning dari model pretrained pada gambar warna resolusi lebih tinggi (Track A Week 28).
- Model selalu memilih salah satu dari 10 kelas, termasuk untuk gambar yang bukan pakaian (tidak ada kelas "lainnya"/deteksi out-of-distribution).
- Kelas Shirt hanya 75,6%. Kalau kelas ini penting secara bisnis, perlu resolusi lebih tinggi atau data tambahan.
- Dilatih di CPU dengan seed tetap; di GPU hasil bisa berbeda di desimal kecil (kernel non-deterministik).
