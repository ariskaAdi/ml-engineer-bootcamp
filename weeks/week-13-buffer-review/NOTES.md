# Week 13 — Buffer dan Review Phase 2

Tidak ada materi baru. Ulangi exit criteria yang belum lulus, lalu latih **menjelaskan dengan suara tanpa catatan**.

## Checklist exit criteria Phase 2
| Week | Exit criteria | Bukti |
|---|---|---|
| 7 | Bandingkan model dengan baseline dan jelaskan metrik | Tabel MAE/RMSE/R² Week 7 |
| 8 | Pilih metrik berdasarkan biaya FP/FN | Threshold 0,09 vs 0,5 (Week 8) |
| 9 | Hindari leakage dengan pipeline | Demo SelectKBest 0,76 vs 0,56 |
| 10 | Bedakan skor train vs val; pilih strategi split | KFold vs TimeSeriesSplit 5,7 vs 24,9 |
| 11 | Pilih model berdasarkan validasi, bukan kerumitan | Ridge/LogReg mengalahkan boosting |
| 12 | Notebook reproducible + alasan pemilihan model | `REPORT.md` |

## Latihan menjelaskan dengan suara (±2 menit per topik)

<details><summary>1. Overfitting</summary>

Model belajar pola + noise khusus data train, sehingga skor train sangat bagus tapi skor data baru buruk. Tandanya: selisih besar antara train dan validation (pohon tanpa batas: train AUC 1,00, val 0,64). Penanganan: model lebih sederhana, regularisasi (Ridge/Lasso, max_depth, min_samples_leaf), lebih banyak data, lebih sedikit fitur noise.
</details>

<details><summary>2. Data leakage</summary>

Informasi yang tidak tersedia saat prediksi sungguhan masuk ke training/validasi, sehingga skor validasi terlalu bagus dan model gagal di produksi. Contoh: scaler/seleksi fitur di-fit di semua data, split acak pada data waktu, fitur dari masa depan (durasi telepon, total tagihan akhir), entitas yang sama di train dan test. Pencegahan: pipeline, split yang tepat, tanyakan "apakah fitur ini sudah ada di saat prediksi?"
</details>

<details><summary>3. Cross-validation</summary>

Data dev dibagi K bagian; setiap bagian bergiliran menjadi validasi. Menghasilkan K skor → mean (perkiraan performa) dan std (ketidakpastian). Lebih stabil daripada satu split. Pilih jenisnya sesuai data: Stratified (kelas tidak seimbang), Group (entitas berulang), TimeSeries (waktu). Test set tetap di luar CV.
</details>

<details><summary>4. Pemilihan model</summary>

Bandingkan kandidat dengan CV yang sama dan metrik yang sama, termasuk baseline Dummy dan linear. Pilih skor CV terbaik; kalau selisihnya di bawah ±1 std, pilih yang lebih sederhana. Test set hanya untuk melaporkan hasil akhir sekali.
</details>

<details><summary>5. Trade-off metrik</summary>

Precision = dari yang ditandai positif, berapa yang benar. Recall = dari yang benar positif, berapa yang tertangkap. Menurunkan threshold menaikkan recall dan menurunkan precision. Pilih berdasarkan biaya: FN mahal → recall; FP mahal → precision; biaya diketahui → minimalkan biaya total. Accuracy menyesatkan untuk kelas tidak seimbang. Regresi: MAE mudah dijelaskan, RMSE menghukum error besar.
</details>

<details><summary>6. Kenapa pipeline penting untuk produksi?</summary>

Satu objek menyimpan semua langkah preprocessing + model. Data mentah → prediksi, dengan langkah yang identik dengan saat training. Mencegah training-serving skew dan leakage di CV.
</details>

## Latihan "kode dari nol" (tanpa melihat)
- [ ] Pipeline + ColumnTransformer untuk data churn.
- [ ] `cross_validate` dengan `return_train_score=True` dan baca hasilnya.
- [ ] `GridSearchCV` dengan nama parameter `model__C`.
- [ ] Hitung precision/recall dari confusion matrix.

## Catatan pribadi (isi sendiri)
- Topik yang masih belum lancar dijelaskan:
- Rencana:
