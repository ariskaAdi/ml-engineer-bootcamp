# Week 15 — Data Tidak Seimbang dan Analisis Error

## File dan cara menjalankan
`uv run python weeks/week-15-imbalanced-error-analysis/imbalanced_errors.py` → `outputs/week15_1_pr.png`.
Data: `make_fraud()` — 20.000 transaksi, **2% fraud**. Target operasional: precision ≥ 30% (tim investigasi tidak sanggup memeriksa terlalu banyak alarm palsu), dengan recall setinggi mungkin.
Metrik utama: **recall @ precision ≥ 30%** dan **PR-AUC**.

## Accuracy tidak berguna di sini
"Semua transaksi normal" → accuracy 98%, fraud tertangkap 0.

## Cara menangani ketidakseimbangan — hasil (validation)
| Cara | PR-AUC | Recall @ P≥30% | Pada threshold 0,5: TP / FP / FN |
|---|---|---|---|
| LogReg biasa | 0,786 | 0,848 | 61 / 4 / 38 |
| `class_weight="balanced"` | 0,785 | 0,869 | 87 / **308** / 12 |
| Oversampling 50:50 | 0,784 | 0,869 | 87 / 316 / 12 |
| Undersampling 1:5 | 0,786 | 0,869 | 75 / 50 / 24 |
| HistGradientBoosting | **0,813** | 0,869 | 65 / 9 / 34 |

**Pelajaran penting:** class weight dan resampling **hampir tidak mengubah PR-AUC**. Ranking transaksi (mana yang paling mencurigakan) praktis sama. Yang berubah adalah **skala peluang**: model "balanced" memberi peluang lebih tinggi untuk semua, sehingga pada threshold 0,5 ia menangkap lebih banyak fraud tapi juga 308 alarm palsu.
Efek yang sama bisa didapat dengan menurunkan threshold model biasa (0,5 → 0,032). Karena itu, langkah pertama untuk data tidak seimbang biasanya: **metrik yang tepat + memilih threshold**. Resampling/weight berguna terutama untuk model yang kesulitan belajar dari sedikit contoh positif, atau kalau kamu butuh peluang di sekitar 0,5.

Cara kerja:
- `class_weight="balanced"`: error pada kelas langka dihukum lebih berat (bobot ∝ 1/frekuensi).
- **Oversampling**: duplikasi contoh positif (SMOTE = membuat contoh sintetis di antara positif; ada di library `imbalanced-learn`).
- **Undersampling**: buang sebagian negatif. Training lebih cepat, tapi informasi terbuang.

## Resampling sebelum split = leakage
Oversampling seluruh data dev lalu split → PR-AUC "validasi" **0,999**. Salinan fraud yang sama ada di train dan validasi, jadi model cukup menghafal. **Resampling hanya boleh di data train** (atau di dalam pipeline `imblearn` saat CV).

## Analisis error
Pada threshold operasional (precision 30%), median fitur per kelompok:
| | amount | hour | km_from_home | new_merchant |
|---|---|---|---|---|
| TP (fraud tertangkap) | 418 | 4 | 89 | 1 |
| FN (fraud lolos) | 159 | 19 | 12 | 0 |
| FP (alarm palsu) | 187 | 11 | 25 | 1 |
| TN (normal) | 146 | 14 | 9 | 0 |

Fraud yang lolos terlihat sangat mirip transaksi normal (nominal kecil, dekat rumah). Ini fraud "tersamar" yang ada di generator data. Sebagian mungkin tidak akan pernah bisa ditangkap dengan fitur ini.
Tapi ada pola yang **bisa** diperbaiki: **33% FN terjadi malam hari**, padahal hanya 4% transaksi normal yang malam.

**Hipotesis**: LogReg memperlakukan `hour` sebagai angka lurus (0 → 23). Jam 23 dan jam 2 dianggap berjauhan, padahal keduanya "malam". Model linear tidak bisa mengungkapkan "jam 23–04 mencurigakan".

## Eksperimen terkontrol (satu faktor diubah)
Yang diubah: tambah fitur `is_night`. Yang sama: model, data, split, seed.
| | PR-AUC | Recall @ P≥30% |
|---|---|---|
| Sebelum | 0,786 | 0,848 |
| + `is_night` | **0,852** | **0,909** |
Diulang dengan 5-fold CV: PR-AUC naik **di 5/5 fold** (+0,046 sampai +0,056). Perbaikan ini konsisten, bukan kebetulan satu split.
**Test (sekali):** PR-AUC 0,839, precision 0,303, recall 0,890 pada threshold yang dipilih di validasi (0,024).

Alur yang dipakai (ulangi untuk setiap kelemahan): **lihat error → temukan pola → hipotesis yang bisa diuji → ubah satu hal → ukur dengan CV → simpulkan.**

Alternatif lain untuk menguji: encoding siklis `sin/cos(2π·hour/24)`, atau model pohon (HGB bisa memotong `hour ≤ 4 atau ≥ 23` sendiri, sehingga PR-AUC-nya sudah lebih tinggi).

## Analogi untuk orang awam
- **Data tidak seimbang** → mencari 2 jarum di antara 98 jerami. Bilang "semuanya jerami" sudah benar 98%, tapi tidak berguna.
- **Class weight / oversampling** → memberi tahu pelatih "setiap jarum yang terlewat dihitung 50 kali lebih parah". Model jadi lebih "berani" menebak jarum.
- **Threshold** → mengatur kepekaan detektor logam. Sering kali cukup memutar kenop ini tanpa membeli detektor baru.
- **Resampling sebelum split** → memfotokopi soal ujian lalu menyelipkan salinannya ke buku latihan: nilai ujian sempurna, tapi palsu.
- **Analisis error** → montir yang membuka mesin yang rusak dan menemukan pola ("selalu rusak saat malam dingin"), lalu menguji satu perbaikan pada satu waktu.

## Exit criteria — tes diri
- [ ] Sebutkan 2 kelompok kegagalan model dan usulkan perbaikan yang bisa diuji untuk masing-masing.
- [ ] Jelaskan kenapa `class_weight` tidak mengubah PR-AUC tapi mengubah jumlah TP/FP di threshold 0,5.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
