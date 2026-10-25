# Week 09 — Preprocessing, Pipeline, dan Feature Engineering

## File dan cara menjalankan
`uv run python weeks/week-09-pipelines-features/pipelines_features.py`
Menyimpan `models/week09_house_pipeline.joblib` (preprocessing + model dalam satu file).

## Komponen
| Komponen | Fungsi | Belajar (`fit`) apa dari train |
|---|---|---|
| `SimpleImputer(strategy="median")` | Isi nilai kosong | Median setiap kolom |
| `add_indicator=True` | Tambah kolom 0/1 "tadinya kosong?" | Kolom mana yang punya nilai kosong |
| `StandardScaler` | `(x − mean) / std` | Mean dan std |
| `OneHotEncoder(handle_unknown="ignore")` | Kategori → kolom 0/1 | Daftar kategori. Kategori baru di produksi (misalnya "Bali") → semua 0, tidak error |
| `ColumnTransformer` | Terapkan langkah berbeda ke kolom angka dan kategori | — |
| `Pipeline` | Rangkai langkah: preprocessing → model | — |
| `FunctionTransformer(add_features)` | Bungkus fungsi pandas biasa menjadi langkah pipeline | Tidak ada (stateless) |
| `TransformedTargetRegressor` | `log` target saat fit, `exp` saat predict, otomatis | — |

## Kenapa pipeline mencegah leakage (exit criteria)
**Leakage** = informasi yang tidak akan tersedia saat prediksi sungguhan "bocor" ke training, sehingga skor validasi terlalu optimis.

Kalau `StandardScaler().fit(semua_data)` dijalankan sebelum split, mean/std ikut dihitung dari data validasi/test. Model jadi sedikit "mengintip". Dengan pipeline:
- `pipe.fit(train)` → setiap langkah belajar **hanya** dari train.
- `pipe.predict(val)` → val hanya di-*transform* dengan statistik train.
- Dalam `cross_val_score(pipe, ...)`, seluruh pipeline di-fit ulang di setiap fold, jadi tidak ada yang bocor antar fold.

**Demo leakage di script:** 2000 fitur noise, target acak (jawaban yang benar: akurasi ±50%).
- Seleksi 20 fitur "terbaik" dengan **semua data**, lalu CV → akurasi **0,76** (palsu!).
- Seleksi fitur **di dalam pipeline** → **0,56** (≈ tebak koin, jujur).
Dengan ribuan fitur, beberapa pasti kebetulan berkorelasi dengan target. Kalau fitur dipilih dengan melihat data validasi, kebetulan itu ikut terbawa ke skor validasi.

Pipeline juga menyelesaikan masalah produksi: satu file `joblib` berisi semua langkah. Data mentah masuk → prediksi keluar. Tidak ada risiko preprocessing di server berbeda dengan di notebook (*training-serving skew*).

## Feature engineering: buat, lalu uji
Hasil validation (MAE, juta Rp). Baseline = pipeline Week 7 (sekarang menangani nilai kosong tanpa membuang baris):
| Eksperimen | MAE | RMSE | Kesimpulan |
|---|---|---|---|
| Baseline pipeline | 266,9 | 416,2 | — |
| + `years_since_2022` (bagian tanggal) | 258,9 | 407,8 | Membantu: menangkap tren harga naik |
| + `building_to_land` (rasio) | 266,2 | 415,5 | Hampir tidak berpengaruh |
| + `log_land` | 266,9 | 415,5 | Tidak berpengaruh |
| + `land_x_jakarta` (interaksi) | 232,2 | 370,5 | **Membantu paling banyak**: harga/m² di Jakarta jauh lebih tinggi; model linear tidak bisa menangkap ini tanpa interaksi |
| + `age_group` (binning) | 266,9 | 416,6 | Tidak membantu: umur sudah linear |
| Semua fitur baru | 219,7 | 360,6 | Gabungan lebih baik |
| Semua fitur + `log(target)` | **177,1** | **284,3** | Terbaik: efek perkalian menjadi penjumlahan |

Model akhir (dilatih di 2022–2024) di **TEST 2025**: MAE 192,7, RMSE 274,7. Bandingkan dengan Week 7 (MAE 277,7, split acak).

Pelajaran: **tidak semua fitur baru membantu.** Uji satu per satu (*ablation*), simpan yang terbukti membantu, dan catat hasilnya. Fitur yang paling berguna biasanya berasal dari pemahaman domain (harga Jakarta per m² berbeda), bukan dari transformasi acak.

## Jenis feature engineering
- **Bagian tanggal**: tahun, bulan, hari dalam minggu, "berapa hari sejak X".
- **Rasio**: building/land, harga/m², utang/pendapatan.
- **Binning**: angka → kelompok (umur → baru/muda/tua). Berguna kalau efeknya tidak linear.
- **Interaksi**: perkalian dua fitur, kalau efek fitur A bergantung pada B.
- **Transformasi**: log untuk data miring.

## Jebakan
- `pd.get_dummies` di data produksi bisa menghasilkan kolom yang berbeda dari training. `OneHotEncoder` di pipeline tidak.
- Lupa `handle_unknown="ignore"` → error saat ada kategori baru.
- Fitur yang memakai target (misalnya rata-rata harga per kota) **harus** dihitung di dalam fold/pipeline.

## Exit criteria — tes diri
- [ ] Jelaskan kenapa `scaler.fit(X_semua)` sebelum split adalah leakage, dan bagaimana pipeline mencegahnya.
- [ ] Bangun ulang pipeline dari nol: imputer + scaler untuk angka, imputer + one-hot untuk kategori, lalu LinearRegression.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
