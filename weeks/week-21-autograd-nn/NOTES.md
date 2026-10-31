# Week 21 — Autograd dan Neural Network

## File dan cara menjalankan
`uv run python weeks/week-21-autograd-nn/autograd_nn.py` (±10 detik) → `outputs/week21_*.png`.
Data: `make_moons` (dua bulan sabit, 1000 titik) — tidak bisa dipisahkan dengan garis lurus.

## Autograd = chain rule Week 5, otomatis
`requires_grad=True` → PyTorch mencatat setiap operasi menjadi **computational graph** (`PowBackward0 ← SubBackward0 ← AddBackward0 ← MulBackward0`). `loss.backward()` menelusuri graph itu dari belakang dan menerapkan chain rule.
Hasilnya **sama persis** dengan perhitungan tangan Week 5: `dL/dw = −26`, `dL/db = −13`.

Tiga jebakan:
1. **Gradien terakumulasi.** `backward()` kedua kali → −52 (dijumlahkan). Karena itu setiap langkah training diawali `optimizer.zero_grad()`.
2. **Update parameter harus di `torch.no_grad()`.** Kalau tidak, update itu sendiri ikut direkam di graph.
3. **Evaluasi di `torch.no_grad()`** → tidak membangun graph, lebih hemat memori dan lebih cepat.

## Komponen model
- **`nn.Module`**: definisikan layer di `__init__`, alurnya di `forward`. Parameter otomatis terdaftar (`model.parameters()` → 1218 angka untuk MLP 2→32→32→2).
- **`nn.Linear(in, out)`**: `y = x·Wᵀ + b`.
- **Fungsi aktivasi** (`week21_1_activations.png`): tanpa aktivasi, tumpukan Linear tetap = satu Linear. **ReLU** `max(0, x)` adalah default (sederhana, gradien tidak mengecil untuk x > 0). Sigmoid/Tanh jenuh di ujung → gradien mendekati 0 (*vanishing gradient*).
- **Loss**:
  - Klasifikasi multi-kelas → `CrossEntropyLoss` (input: **logit** mentah + label `int64`). Di dalamnya sudah ada softmax. Jangan pakai softmax dua kali!
  - Biner dengan 1 output → `BCEWithLogitsLoss`.
  - Regresi → `MSELoss` / `L1Loss`.
  - Script membuktikan CE PyTorch (0,1275) = `−mean(log_softmax[label])` manual.
- **Optimizer**: aturan update. SGD: `w −= lr·grad`. Momentum: rata-rata bergerak dari gradien (seperti bola yang menggelinding). Adam: learning rate menyesuaikan per parameter. Biasanya pilihan default yang aman (lr 1e-3 – 1e-2).

## Hubungan prediksi → loss → gradien → update (exit criteria)
```python
logits = model(X)              # 1. PREDIKSI dengan parameter saat ini
loss = loss_fn(logits, y)      # 2. LOSS: satu angka, seberapa salah prediksi itu
optimizer.zero_grad()
loss.backward()                # 3. GRADIEN: untuk setiap parameter, arah yang MENAIKKAN loss
optimizer.step()               # 4. UPDATE: geser setiap parameter sedikit ke arah sebaliknya
```
Loss adalah fungsi dari parameter (data tetap). Gradien memberi tahu bagaimana loss berubah kalau setiap parameter digeser sedikit. Update menggeser parameter ke arah yang menurunkan loss. Diulang ratusan kali → prediksi makin baik.

## Hasil (300 epoch, full batch)
| Model / optimizer | Loss akhir | Val accuracy |
|---|---|---|
| Linear saja (= logistic regression) | 0,299 | 0,823 |
| MLP + SGD lr=0,1 | 0,236 | 0,893 |
| MLP + SGD + momentum 0,9 | 0,147 | **0,933** |
| MLP + Adam lr=0,01 | **0,142** | 0,930 |

Model linear hanya bisa membuat batas lurus. MLP dengan hidden layer + ReLU bisa membuat batas melengkung (`week21_3_boundary.png`). SGD polos paling lambat turun; momentum dan Adam jauh lebih cepat (`week21_2_optimizers.png`).

## Analogi untuk orang awam
- **Neural network** → rangkaian penyaring kopi bertingkat: setiap lapisan mengubah dan menyaring informasi sebelum diteruskan.
- **Fungsi aktivasi** → keran yang hanya mengalirkan air kalau tekanannya cukup. Tanpa keran ini, setumpuk lapisan sama saja dengan satu lapisan.
- **Loss** → skor "seberapa jauh tembakanmu dari target".
- **Autograd / backpropagation** → setelah tembakan meleset, pelatih menelusuri mundur: "bahumu terlalu tinggi 2 cm, siku terlalu ke kiri 1 cm". Setiap bagian tahu harus dikoreksi seberapa banyak.
- **Optimizer** → gaya belajar: SGD seperti melangkah sesuai kemiringan saat ini; momentum seperti bola menggelinding yang tidak mudah berhenti di lubang kecil; Adam menyesuaikan panjang langkah untuk setiap otot secara terpisah.
- **`zero_grad()`** → menghapus papan tulis sebelum menulis koreksi baru; kalau lupa, koreksi lama ikut terhitung.

## Exit criteria — tes diri
- [ ] Tanpa melihat, tulis 5 baris inti loop training dan jelaskan setiap baris.
- [ ] Jelaskan kenapa `zero_grad()` perlu dan apa yang terjadi kalau lupa.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
