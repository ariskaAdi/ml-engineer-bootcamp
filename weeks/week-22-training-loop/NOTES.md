# Week 22 — Training Loop

## File dan cara menjalankan
`uv run python weeks/week-22-training-loop/training_loop.py` (±3 menit di CPU; unduh FashionMNIST ±30 MB ke `data/` saat pertama kali).
Output: `models/week22_fashion_mlp.pt` (checkpoint) dan `models/week22_history.csv` (metrik per epoch).

## Dataset dan DataLoader
- **`Dataset`** cukup punya dua method: `__len__` (berapa sampel) dan `__getitem__(i)` (ambil sampel ke-i sebagai `(x, y)`). `TabularDataset` di script adalah pola untuk data apa pun (CSV, gambar dari folder, teks).
- **`transform`** dijalankan setiap kali sampel diambil: `ToTensor()` mengubah gambar PIL (0–255, HWC) menjadi tensor float (0–1, CHW); `Normalize(mean, std)` membuat nilainya berpusat di 0 (mean/std FashionMNIST = 0,286 / 0,353).
- **`DataLoader`** menggabungkan sampel menjadi batch `(128, 1, 28, 28)`. `shuffle=True` untuk train (urutan berbeda setiap epoch, supaya gradien tidak bias oleh urutan), `False` untuk val/test.
- **`random_split`** dengan `generator` ber-seed → 50.000 train / 10.000 val, bisa diulang. Test set (10.000) adalah file terpisah dari dataset aslinya.

Istilah: **batch** = sekumpulan sampel untuk satu langkah update; **epoch** = satu putaran melewati seluruh data train (391 batch × 128).

## Loop training vs validasi
| | `train_one_epoch` | `evaluate` |
|---|---|---|
| Mode | `model.train()` | `model.eval()` |
| Gradien | Ya: `zero_grad → backward → step` | Tidak: `@torch.no_grad()` |
| Mengubah bobot | Ya | **Tidak pernah** |

`model.train()` / `model.eval()` tidak menjalankan training. Keduanya hanya mengubah perilaku layer seperti Dropout dan BatchNorm (Week 23). Lupa `eval()` saat validasi = hasil acak/salah.
`loss.item()` mengambil angka Python. Kalau `loss` (tensor) yang dijumlahkan, seluruh graph ikut tersimpan dan memori bocor.
Loss dijumlahkan sebagai `loss × len(y)` lalu dibagi total sampel, supaya batch terakhir yang lebih kecil tidak terhitung terlalu berat.

## Hasil (MLP 784→256→128→10, Adam lr 1e-3, 5 epoch)
| Epoch | Train loss | Train acc | Val loss | Val acc |
|---|---|---|---|---|
| 1 | 0,499 | 0,819 | 0,399 | 0,855 |
| 3 | 0,321 | 0,881 | 0,330 | 0,881 |
| 5 | 0,275 | 0,898 | 0,317 | **0,884** |
**Test: 0,876.**

Train acc mulai melewati val acc di epoch 4–5 → awal overfitting (dibahas Week 23).
Per kelas: **Shirt 66%**, Coat 76%, Pullover 77%: tiga kelas atasan yang mirip. Trouser, Bag, Sandal, Ankle boot > 95%. Model tanpa konvolusi kesulitan membedakan detail bentuk; CNN di Portfolio 2 memperbaikinya.

## Simpan dan muat
```python
torch.save({"epoch": e, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, path)
model = MLP(); model.load_state_dict(torch.load(path)["model_state"])
```
- `state_dict` = dictionary nama layer → tensor bobot (`net.1.weight`, ...). Yang disimpan hanya angka, **bukan** kode arsitektur. Untuk memuat, kelas `MLP` yang sama harus tersedia.
- Menyimpan `optimizer_state` + `epoch` memungkinkan melanjutkan training yang terputus (state Adam ikut tersimpan).
- Checkpoint disimpan hanya kalau val acc membaik → yang tersimpan adalah epoch terbaik, bukan epoch terakhir.
- `map_location=device` supaya checkpoint dari GPU bisa dimuat di CPU.

## Kenapa 1 epoch ±35 detik di CPU?
Sebagian besar waktunya bukan untuk menghitung model, tapi untuk `__getitem__` → konversi PIL → tensor satu per satu di Python. Ini **bottleneck data loading** (Week 24).

## Exit criteria — tes diri
- [ ] Tulis `train_one_epoch` dan `evaluate` dari nol tanpa melihat.
- [ ] Catat loss & accuracy train/val per epoch, lalu jelaskan kapan model mulai overfit.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
