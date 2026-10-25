# Week 03 — Exploratory Data Analysis (EDA)

## File dan cara menjalankan
`uv run python weeks/week-03-eda/eda_house_prices.py` → 6 plot di `outputs/week03_*.png`.
Lebih enak: buka file di VS Code dan jalankan per cell (Run Cell), supaya plot tampil di samping kode.

Dataset: `make_house_prices()` dari [src/mlbootcamp/datasets.py](../../src/mlbootcamp/datasets.py) — 3000 rumah, 5 kota, 2022–2025, harga dalam juta Rupiah. Dataset ini juga dipakai Week 7–12, jadi EDA ini sekaligus persiapan modeling.

## Kenapa menulis pertanyaan dulu?
Tanpa pertanyaan, EDA berubah menjadi "membuat semua plot yang mungkin" lalu bingung sendiri. Setiap plot di script menjawab satu pertanyaan (Q1–Q6 di bagian atas script).

## Temuan dan interpretasi
| Plot | Temuan | Artinya untuk modeling |
|---|---|---|
| 1. Distribusi harga | Miring ke kanan (skewness 1,95); mean 1772 > median 1502. Setelah `log`, skewness ≈ 0 | Error model akan didominasi rumah mahal. Pertimbangkan memprediksi `log(price)` |
| 2. Boxplot per kota | 133 rumah (4,4%) di atas batas IQR, kebanyakan rumah besar di Jakarta | Outlier di sini **wajar** (rumah besar memang mahal), bukan data salah → jangan dibuang otomatis |
| 3. Korelasi | land_area 0,78; building_area 0,77; bedrooms 0,68; distance −0,23; age ≈ 0 | Luas adalah sinyal terkuat. Banyak fitur saling berkorelasi (luas ↔ kamar ↔ kamar mandi) |
| 4. Scatter luas vs harga per kota | Pada skala log, hubungannya hampir lurus; tiap kota punya "garis" sendiri | Kota harus jadi fitur (one-hot) |
| 5. Tren per kuartal | Harga/m² naik ±6–7% per tahun | Data berurutan waktu → Week 10: split berdasarkan waktu, bukan acak |
| 6. Kamar tidur dalam kelompok luas | Korelasi turun dari 0,68 menjadi ±0,05–0,16 di dalam kelompok luas yang sama | Lihat bagian korelasi vs kausalitas |

**Nilai kosong.** `building_area` 5%, `age_years` 4%, `bathrooms` 3%. Median harga baris kosong mirip dengan yang tidak kosong → kemungkinan kosongnya acak (MCAR). Ini membuat imputasi sederhana (median) masuk akal di Week 9.

**Harga per m².** Untuk membandingkan kota, bandingkan harga/m² (Jakarta 17,9 jt vs Medan 9,3 jt), bukan harga total. Harga total tercampur dengan ukuran rumah.

## Korelasi ≠ kausalitas (exit criteria)
Kamar tidur berkorelasi 0,68 dengan harga. Tapi di generator data, **jumlah kamar tidur sama sekali tidak dipakai** untuk menghitung harga. Kamar tidur hanya ikut naik karena rumah yang luas punya lebih banyak kamar. Luas bangunan adalah **confounder**: ia memengaruhi kamar tidur dan harga sekaligus.

Bukti di plot 6: kalau luas bangunan dibuat hampir sama (dikelompokkan), korelasi kamar tidur dengan harga hampir hilang. Kelompok terakhir masih 0,44 karena rentangnya lebar (199–765 m²), jadi di dalamnya masih ada efek luas.

Pelajaran: "rumah dengan kamar lebih banyak lebih mahal" adalah fakta (asosiasi). "Tambah kamar → harga naik" adalah klaim sebab-akibat yang tidak bisa dibuktikan dari data observasi saja.

Tiga alasan korelasi muncul tanpa sebab-akibat: (1) confounder, (2) arah terbalik (Y menyebabkan X), (3) kebetulan / ukuran sampel kecil.

## Pilihan teknis kecil
- **Spearman vs Pearson.** Pearson mengukur hubungan *lurus*, dan sensitif terhadap outlier. Spearman memakai ranking, jadi menangkap hubungan monoton apa pun dan lebih tahan outlier. Untuk harga yang miring, Spearman lebih aman.
- **Skala log** di scatter membuat pola perkalian (harga ≈ luas × harga/m²) menjadi garis lurus.
- **IQR rule:** outlier = di luar `[Q1 − 1,5·IQR, Q3 + 1,5·IQR]`. Ini definisi statistik, bukan definisi "data salah".

## Checklist plot yang baik
Judul yang menyebut temuan, label sumbu dan satuan, skala log kalau datanya miring, `alpha` kecil kalau titiknya ribuan.

## Exit criteria — tes diri
- [ ] Jelaskan 3 temuan dari plot dengan kata-kata sendiri.
- [ ] Jelaskan kenapa "kamar tidur menyebabkan harga naik" tidak bisa disimpulkan dari plot 3.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
