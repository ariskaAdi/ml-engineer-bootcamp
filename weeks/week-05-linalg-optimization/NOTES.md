# Week 05 — Aljabar Linear dan Intuisi Optimisasi

## File dan cara menjalankan
`uv run python weeks/week-05-linalg-optimization/gradient_descent.py` → 3 plot `outputs/week05_*.png`.

## 1. Vektor dan matriks dalam ML
- Satu baris data (satu rumah) = **vektor fitur** `x`.
- Model linear: `ŷ = x · w + b` (dot product = kalikan pasangan lalu jumlahkan).
- Banyak rumah sekaligus: matriks `X` berukuran `(n_rumah, n_fitur)` → `ŷ = X @ w + b`, hasilnya `(n_rumah,)`.
- Aturan shape: `(n, k) @ (k, m) → (n, m)`. Hampir semua error di deep learning nanti adalah shape yang tidak cocok, jadi biasakan menulis shape di komentar.
- **Transpose** `X.T` menukar baris dan kolom. Ia muncul di rumus gradien: `X.T @ error` menjumlahkan kontribusi setiap rumah ke setiap bobot.

## 2. Turunan, gradien, learning rate
- **Turunan** f'(w) = kemiringan fungsi di titik w. Negatif → fungsi turun kalau w dinaikkan.
- **Gradien** = kumpulan turunan terhadap setiap parameter, menunjuk ke arah loss *naik* paling cepat.
- **Gradient descent**: `w ← w − lr · gradien`. Melangkah berlawanan arah gradien = turun.
- **Learning rate** (lihat `week05_1_learning_rates.png`):
  - 0,01: turun, tapi lambat.
  - 0,1–0,5: cepat konvergen.
  - 1,01: setiap langkah melewati minimum terlalu jauh, sehingga loss **membesar** (divergen).
- **Turunan numerik** `(f(w+h) − f(w−h)) / 2h` dipakai untuk mengecek turunan rumus (*gradient check*). Script menunjukkan keduanya sama.

## 3. Chain rule
Kalau `L` bergantung pada `ŷ`, dan `ŷ` bergantung pada `w`:
`dL/dw = dL/dŷ × dŷ/dw`.
Untuk `L = (y − ŷ)²` dan `ŷ = wx + b`: `dL/dw = −2(y − ŷ)·x`, `dL/db = −2(y − ŷ)`.
Neural network = rantai fungsi yang panjang. **Backpropagation** = chain rule yang dihitung dari belakang (loss) ke depan (input). PyTorch melakukannya otomatis di Week 21; minggu ini kita melakukannya dengan tangan.

## 4. Hasil regresi linear (rumah Yogyakarta, harga vs luas tanah)
| Metode | w | b | MSE |
|---|---|---|---|
| Gradient descent (lr=0,1, 200 langkah) | 620,16 | 1290,86 | 106.102 |
| Least squares (`np.linalg.lstsq`) | 620,16 | 1290,86 | 106.102 |

Keduanya sama, jadi implementasi GD benar. Interpretasi: +1 m² tanah ≈ +10,1 juta.
4 fitur sekaligus menurunkan MSE menjadi ±54.900.

Catatan: bobot `bathrooms` negatif (−14,4), padahal di data asli kamar mandi *menambah* harga. Penyebabnya: kamar mandi sangat berkorelasi dengan luas bangunan (*multikolinearitas*), sehingga model "membagi" efeknya secara tidak stabil. Jangan membaca bobot model linear sebagai sebab-akibat (kaitkan dengan Week 3).

## 5. Kenapa standardisasi fitur?
Luas tanah bernilai ratusan, sedangkan bias berskala 1. Tanpa standardisasi, gradien untuk `w` jauh lebih besar daripada untuk `b`, sehingga loss berbentuk lembah panjang dan sempit. Learning rate yang aman untuk `w` terlalu lambat untuk `b`. Setelah `(x − mean) / std`, kontur loss lebih bulat dan GD langsung menuju minimum (lihat plot kontur).

## 6. Loss surface
Untuk regresi linear + MSE, loss berbentuk **mangkuk (convex)**: hanya ada satu minimum, jadi GD pasti sampai kalau learning rate cukup kecil. Neural network tidak convex, tapi ide langkahnya sama.

## Analogi untuk orang awam
- **Vektor dan dot product** → resep: jumlah setiap bahan (fitur) dikali seberapa penting bahan itu bagi rasa (bobot), lalu dijumlahkan menjadi skor akhir.
- **Gradient descent** → turun gunung saat kabut tebal: kamu tidak melihat lembah, tapi bisa merasakan kemiringan di bawah kaki, lalu melangkah ke arah yang paling menurun.
- **Learning rate** → panjang langkah. Terlalu kecil = sampai bawah besok pagi; terlalu besar = melompati lembah dan malah naik ke bukit seberang (divergen).
- **Chain rule** → efek domino atau rantai pabrik: kalau mesin A melambat 2× dan mesin B bergantung pada A dengan faktor 3×, efek totalnya 6×. Backpropagation menghitung efek berantai ini dari ujung ke awal.
- **Standardisasi fitur** → menyamakan satuan sebelum membandingkan (cm vs km). Tanpa itu, "gunung" loss jadi lembah sempit memanjang yang sulit dituruni.

## Exit criteria — tes diri
- [ ] Jelaskan dengan kata-kata: bagaimana gradient descent mengubah parameter supaya loss turun? Kenapa tandanya minus?
- [ ] Tulis ulang loop GD dari nol tanpa melihat script.
- [ ] Jelaskan apa yang terjadi kalau learning rate terlalu besar.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
