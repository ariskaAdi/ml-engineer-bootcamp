# Week 10 — Generalisasi dan Cross-Validation

## File dan cara menjalankan
`uv run python weeks/week-10-generalization-cv/generalization_cv.py` → 3 plot `outputs/week10_*.png`.
Pipeline rumah dari Week 9 sekarang ada di [src/mlbootcamp/features.py](../../src/mlbootcamp/features.py) supaya bisa dipakai ulang (`house_pipeline(model)`).

## Underfitting, overfitting, generalisasi
**Generalisasi** = performa di data yang belum pernah dilihat. Itu satu-satunya yang penting.

Polinomial di 20 titik data (`week10_1_under_overfit.png`):
| Degree | Train MSE | Val MSE | Diagnosis |
|---|---|---|---|
| 1 | 0,218 | 0,285 | **Underfit**: train & val sama-sama buruk. Model terlalu sederhana |
| 4 | 0,018 | 0,096 | Pas |
| 15 | **0,010** | **1,462** | **Overfit**: train terbaik, val terburuk. Model menghafal noise |

Cara membaca selisih train vs validation:
- Keduanya buruk → underfit → model lebih kompleks / fitur lebih baik.
- Train bagus, val jauh lebih buruk → overfit → regularisasi, model lebih sederhana, lebih banyak data.
- Keduanya bagus dan dekat → bagus.

## Regularisasi
Tambahkan hukuman untuk bobot besar ke loss:
- **Ridge (L2)**: `loss + α·Σw²` → bobot mengecil, tapi jarang tepat 0.
- **Lasso (L1)**: `loss + α·Σ|w|` → banyak bobot menjadi **tepat 0** (seleksi fitur otomatis).

Polinomial degree 15: tanpa regularisasi, bobot terbesar 3696 dan val MSE 1,46. Ridge (α≈0,026): bobot terbesar 3,6 dan val MSE **0,090**, setara degree 4. Lasso membuat 11 dari 15 bobot = 0.
α adalah **hyperparameter**: terlalu kecil → overfit, terlalu besar → underfit (`week10_2_ridge_alpha.png`). Dipilih dengan validasi/CV (Week 12).

## K-fold cross-validation
Satu split validasi hanya memberi satu angka. Tiga split acak berbeda memberi MAE 178,0 / 179,9 / 184,8. Mana yang "benar"?
K-fold: data dibagi 5 bagian. Setiap bagian bergiliran menjadi validasi, sisanya untuk train. Hasil: **MAE 181,7 ± 8,2**.
- Mean = perkiraan performa yang lebih stabil.
- Std = seberapa yakin. Kalau dua model selisihnya lebih kecil dari std, perbedaannya mungkin kebetulan.
- Train MAE 178,7 ≈ val MAE 181,7 → model ini tidak overfit. Ini masuk akal untuk model linear dengan regularisasi.

Test set **tetap terpisah** dan tidak ikut CV.

**Learning curve** (`week10_3_learning_curve.png`): kalau kurva validasi masih turun di ujung kanan, menambah data akan membantu. Kalau sudah datar dan menempel ke kurva train, model sudah mencapai batasnya, jadi yang perlu dicari adalah fitur/model yang lebih baik.

## Strategi split (exit criteria)
| Situasi | Split | Kenapa |
|---|---|---|
| Baris saling independen | `KFold(shuffle=True)` | Standar |
| Klasifikasi, kelas tidak seimbang | `StratifiedKFold` | Proporsi kelas sama di setiap fold |
| Satu entitas punya banyak baris (pasien, pelanggan) | `GroupKFold` | Orang yang sama tidak boleh ada di train dan val sekaligus |
| Data berurutan waktu, tugasnya memprediksi masa depan | `TimeSeriesSplit` / split berdasarkan tanggal | Train harus selalu **sebelum** val |

**Demo penjualan harian** (random walk + akhir pekan):
- KFold acak: MAE **5,7**
- TimeSeriesSplit: MAE **24,9**

Dengan split acak, untuk menebak hari ke-100 model sudah melihat hari ke-99 dan ke-101, lalu cukup "menyambung titik". Saat produksi, besok belum terjadi. Skor 5,7 adalah ilusi; 24,9 adalah kenyataan.
`TimeSeriesSplit` selalu melatih di masa lalu dan memvalidasi di periode berikutnya (lihat cetakan fold di script).

Data rumah: CV acak MAE 181,7 vs latih 2022–2024 / uji 2025 MAE 192,6. Selisihnya kecil karena fitur `years_since_2022` menangkap tren, tapi split waktu tetap lebih jujur kalau tugasnya memprediksi harga rumah yang **akan** dijual.

## Daftar cek leakage
- Preprocessing di-fit di luar fold (Week 9).
- Fitur yang baru diketahui **setelah** kejadian (misalnya `total_charges` akhir untuk memprediksi churn bulan ini, atau durasi telepon di Bank Marketing).
- Duplikat / entitas yang sama di train dan test.
- Split acak pada data waktu.
- Memilih model berdasarkan test set.

## Exit criteria — tes diri
- [ ] Diberi skor train 0,98 dan val 0,70: apa diagnosisnya dan apa tiga hal yang bisa dicoba?
- [ ] Untuk prediksi permintaan minggu depan, data pasien dengan banyak kunjungan, dan deteksi spam: split apa yang dipakai?

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
