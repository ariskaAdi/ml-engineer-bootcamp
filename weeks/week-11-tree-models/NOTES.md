# Week 11 — Model Berbasis Pohon

## File dan cara menjalankan
`uv run python weeks/week-11-tree-models/tree_models.py` (±1 menit) → `outputs/week11_1_importance.png`.
Semua angka di bawah adalah **5-fold CV pada data dev** (test set belum disentuh).

## Cara kerja singkat
- **Decision Tree**: berulang kali memilih pertanyaan `fitur ≤ nilai?` yang paling memisahkan target (regresi: MSE turun paling banyak; klasifikasi: Gini/entropy turun paling banyak). Mudah dibaca, tapi pohon dalam sangat mudah overfit.
- **Random Forest**: ratusan pohon dalam, masing-masing dilatih pada sampel bootstrap dan subset fitur acak. Prediksi = rata-rata. Rata-rata dari banyak model yang overfit dengan cara *berbeda-beda* → variansnya turun (*bagging*).
- **Gradient Boosting**: pohon kecil dibangun **berurutan**; setiap pohon baru memperbaiki error (residual) dari gabungan pohon sebelumnya, dikali `learning_rate`. Biasanya model terkuat untuk data tabular. `HistGradientBoosting` = versi cepat di sklearn yang mengelompokkan nilai ke 255 bin dan mendukung nilai kosong secara langsung. Library populer lain: XGBoost, LightGBM, CatBoost.
- Pohon **tidak perlu scaling** (hanya membandingkan `≤`), tapi tetap butuh angka (one-hot untuk kategori).

## Hasil — Harga rumah (MAE, juta Rp; lebih kecil lebih baik)
| Model | Train | Val (CV) | ± |
|---|---|---|---|
| **Ridge + log (baseline)** | 178,7 | **181,7** | 8,2 |
| DecisionTree depth=3 | 416,2 | 442,8 | 18,9 |
| DecisionTree depth=10 | 105,4 | 330,3 | 31,0 |
| DecisionTree depth=None | **0,0** | 348,5 | 29,0 |
| RandomForest 300 | 102,8 | 233,1 | 16,8 |
| HistGradientBoosting + log | 104,2 | 210,6 | 15,9 |

## Hasil — Churn (ROC-AUC; lebih besar lebih baik)
| Model | Train | Val (CV) | ± |
|---|---|---|---|
| **LogisticRegression (baseline)** | 0,839 | **0,836** | 0,008 |
| DecisionTree depth=None | **1,000** | 0,635 | 0,012 |
| RandomForest 300 | 0,937 | 0,823 | 0,009 |
| HistGB default | 0,970 | 0,808 | 0,015 |
| HistGB lr=0,05, depth=3 | 0,859 | 0,829 | 0,011 |

## Kesimpulan (exit criteria)
**Model sederhana menang di kedua dataset.** Kedua dataset sintetis ini memang dibangun dari rumus (log-)linear: harga = perkalian faktor, churn = sigmoid dari jumlah faktor. Model linear dengan fitur yang tepat (interaksi Jakarta, log target dari Week 9) sudah menangkap struktur aslinya. Pohon harus mendekati garis lurus/kurva halus dengan "anak tangga", sehingga butuh banyak data.

Di data asli (Bank Marketing di Portfolio 1, misalnya), boosting biasanya menang, karena polanya tidak rapi. Pelajarannya bukan "linear selalu menang", tapi:
1. **Selalu bandingkan dengan baseline sederhana** memakai CV yang sama.
2. Pilih berdasarkan skor validasi + std, bukan berdasarkan nama model atau kerumitan.
3. Kalau selisih lebih kecil dari ±std, pilih model yang lebih sederhana, lebih cepat, dan lebih mudah dijelaskan.

Tanda overfit terlihat jelas: pohon tanpa batas kedalaman mendapat train MAE 0 / AUC 1,000 (menghafal), tapi validasinya paling buruk. Boosting default (train 0,970 vs val 0,808) juga overfit. Versi yang lebih "pelan" (lr 0,05, depth 3) lebih baik di validasi, dan ini akan dicari otomatis di Week 12.

## Feature importance — hati-hati
Dua kolom acak (`random_id` dengan ±10.000 nilai unik, `random_coin` 0/1) ditambahkan ke Random Forest:
| Fitur | Impurity (bawaan RF) | Permutation (di validation) |
|---|---|---|
| land_area | 0,528 | 0,474 |
| city_jakarta | 0,200 | 0,296 |
| random_id (**acak!**) | 0,013 | −0,001 |
| age_years | 0,013 | 0,002 |
| has_garage | 0,003 | 0,002 |

- **Impurity importance** dihitung dari data **train**, dan bias ke fitur dengan banyak nilai unik: `random_id` terlihat sama penting dengan `age_years`, dan lebih penting dari `has_garage`, karena pohon bisa memakai nilai unik untuk menghafal.
- **Permutation importance** = acak satu kolom di data **validasi**, lalu ukur seberapa besar skor turun. Kolom acak → ±0. Lebih jujur.
- Fitur yang saling berkorelasi (land_area, building_area, bedrooms) berbagi importance. Mengacak satu tidak terlalu merusak karena yang lain menggantikan, sehingga importance masing-masing terlihat lebih kecil.
- Importance = "seberapa banyak **model** memakai fitur ini", **bukan** "seberapa besar fitur ini **menyebabkan** target" (Week 16).

## Analogi untuk orang awam
- **Decision tree** → diagram alur "ya/tidak" di poster: mudah dibaca, tapi pohon yang terlalu dalam menghafal setiap kasus khusus.
- **Random forest** → bertanya pada 300 orang yang masing-masing hanya membaca sebagian informasi, lalu mengambil suara terbanyak. Kesalahan pribadi saling meniadakan.
- **Gradient boosting** → tim yang bekerja berurutan: orang kedua memperbaiki kesalahan orang pertama, orang ketiga memperbaiki sisa kesalahan, dan seterusnya.
- **Model sederhana bisa menang** → untuk mengukur meja, penggaris lebih tepat daripada mesin laser yang rumit. Pilih alat sesuai bentuk masalahnya.
- **Impurity importance yang bias** → menilai karyawan dari "seberapa sering namanya disebut di rapat": orang yang namanya unik dan sering dipanggil tampak penting, padahal kontribusinya kecil. Permutation importance menguji langsung: apa yang rusak kalau orang ini tidak ada?

## Exit criteria — tes diri
- [ ] Jelaskan kenapa Ridge dipilih walaupun Gradient Boosting "lebih canggih".
- [ ] Jelaskan beda bagging (RF) dan boosting dalam 2 kalimat.
- [ ] Jelaskan kenapa `random_id` muncul di impurity importance.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
