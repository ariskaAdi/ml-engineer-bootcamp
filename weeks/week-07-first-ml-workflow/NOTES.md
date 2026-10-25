# Week 07 — Workflow ML Pertama

## File dan cara menjalankan
`uv run python weeks/week-07-first-ml-workflow/first_ml_workflow.py`
Bagian A memakai data rumah sintetis (sama dengan Week 3). Bagian B mengulang alur yang sama dengan **California Housing** (data asli, terunduh otomatis).

## Istilah
- **Fitur `X`**: kolom input (luas, kamar, jarak, ...). Bentuknya matriks `(n_sampel, n_fitur)`.
- **Target `y`**: yang mau ditebak (harga).
- **Train set**: untuk melatih (model melihat jawabannya).
- **Validation set**: untuk memilih model/setting. Model tidak dilatih dengannya, tapi **kita** memakainya untuk membuat keputusan.
- **Test set**: dipakai **sekali**, di akhir, untuk memperkirakan performa di data baru. Kalau test dipakai untuk memilih model, angkanya menjadi terlalu optimis.

Split 60/20/20 dengan `random_state=42` supaya bisa diulang.

## Kenapa baseline?
`DummyRegressor(strategy="mean")` selalu menebak rata-rata harga train, tanpa melihat fitur. Kalau model "pintar" tidak jauh lebih baik daripada ini, berarti model belum belajar apa pun. R² baseline ≈ 0, menurut definisi.

## Metrik regresi
| Metrik | Rumus | Arti | Kapan dipakai |
|---|---|---|---|
| MAE | mean(\|y − ŷ\|) | Rata-rata meleset sekian juta | Mudah dijelaskan ke orang bisnis; tahan outlier |
| RMSE | √mean((y − ŷ)²) | Seperti MAE, tapi error besar dihukum lebih berat | Kalau salah besar sangat mahal |
| R² | 1 − SSE/SST | Proporsi variasi `y` yang dijelaskan model (1 = sempurna, 0 = setara baseline rata-rata, < 0 = lebih buruk dari baseline) | Perbandingan cepat; tidak punya satuan |

RMSE ≥ MAE selalu. Kalau RMSE jauh lebih besar daripada MAE, berarti ada beberapa error yang sangat besar.

## Hasil (validation set)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Dummy (mean) | 805,9 | 1163,1 | −0,003 |
| Dummy (median) | 789,0 | 1206,2 | −0,079 |
| LinearRegression (7 fitur angka) | 484,5 | 670,1 | 0,667 |
| + kota (one-hot) | 300,4 | **450,5** | **0,850** |
| + kota, prediksi log(harga) | **290,2** | 519,3 | 0,800 |

**Terpilih (RMSE): Linear + kota → TEST: MAE 277,7 · RMSE 388,0 · R² 0,875.**

Interpretasi:
- Model rata-rata meleset ±278 juta, sedangkan baseline ±806 juta: kesalahannya turun ±65%.
- Menambah kota memberi lompatan terbesar. Ini sesuai temuan EDA Week 3.
- Model log menang di MAE tapi kalah di RMSE. **Metrik menentukan pemenang**, jadi pilih metrik sebelum membandingkan model.
- Median error ±13–17% per kota. Error terbesar ada di rumah mahal di Jakarta: model linear menjumlahkan efek, padahal harga di data ini terbentuk dari perkalian (kota × luas × tren waktu).
- Koefisien `bedrooms` positif besar (+49), padahal di generator kamar tidur **tidak berpengaruh**. Ini efek korelasi dengan luas (Week 3 & 5). Koefisien ≠ sebab.

California Housing: Dummy R² 0,00 → Linear R² 0,61. Target dipotong di 5,0 (4,8% data), sehingga model linear kesulitan memprediksi rumah termahal.

## Kenapa melatih ulang di train+val sebelum test?
Setelah model dan setting dipilih, validation tidak lagi diperlukan untuk memilih. Menggabungkannya memberi model lebih banyak data. Test tetap tidak disentuh sampai langkah terakhir.

## Jebakan
- Melihat test set "sedikit saja" untuk memilih model → angka test jadi tidak jujur.
- Melatih dengan `log(y)` tapi lupa `exp()` sebelum menghitung metrik → MAE dalam satuan log, tidak bermakna.
- `pd.get_dummies` pada seluruh data sebelum split aman untuk kategori tetap, tapi Week 9 akan memakai `OneHotEncoder` di dalam pipeline (lebih aman untuk kategori baru di produksi).

## Exit criteria — tes diri
- [ ] Jelaskan kenapa model terpilih lebih baik daripada baseline, dengan angka MAE/RMSE/R².
- [ ] Jelaskan beda MAE dan RMSE dan kapan memilih masing-masing.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
