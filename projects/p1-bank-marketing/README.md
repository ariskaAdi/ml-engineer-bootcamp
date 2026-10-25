# Bank Marketing — Siapa yang perlu ditelepon?

Portfolio Project 1 (Week 17–18). Model klasifikasi untuk memilih nasabah yang paling mungkin membuka **deposito berjangka** setelah ditelepon, sehingga call center dengan kapasitas terbatas bisa memprioritaskan daftar teleponnya.

## Masalah
- **Data:** [UCI Bank Marketing — bank-additional-full](https://archive.ics.uci.edu/dataset/222/bank+marketing) (Moro et al., 2014), kampanye telemarketing bank Portugal Mei 2008 – Nov 2010. **Lisensi CC BY 4.0.** 41.188 baris (41.176 setelah membuang 12 duplikat), 20 fitur.
- **Target:** `y` = nasabah membuka deposito (yes/no). Positif hanya **11,3%**.
- **Skenario bisnis:** call center hanya sanggup menelepon **20%** nasabah. Model meranking nasabah; yang kita ukur: berapa persen nasabah "yes" ikut tertangkap di 20% teratas.
- **Metrik utama:** PR-AUC (cocok untuk positif yang langka). Pendukung: ROC-AUC, *capture @ top 20%*.

## Hasil
5-fold Stratified CV di 80% data (dev):
| Model | PR-AUC | ± std | ROC-AUC | Waktu fit |
|---|---|---|---|---|
| Dummy (tebak base rate) | 0,113 | 0,000 | 0,500 | — |
| **Logistic Regression** | 0,450 | 0,014 | 0,790 | 0,9 s |
| Random Forest | **0,467** | 0,018 | 0,798 | 4,3 s |
| HistGradientBoosting | 0,465 | 0,017 | 0,799 | 2,4 s |

**Model terpilih: Logistic Regression.** Random Forest tertinggi, tapi Logistic Regression masih berada dalam 1 standard deviation (0,450 ≥ 0,467 − 0,018). Berdasarkan aturan *one-standard-error*, model yang lebih sederhana dipilih: ±4× lebih cepat dilatih, artefak lebih kecil, inference lebih ringan untuk API di Phase 6, dan koefisiennya bisa diperiksa.

**Test set (20%, dipakai sekali):**
| PR-AUC | ROC-AUC | Capture @ top 20% | Precision @ threshold | Recall @ threshold |
|---|---|---|---|---|
| 0,454 | 0,801 | **64,4%** | 35,7% | 64,4% |

Artinya: menelepon 20% nasabah teratas menangkap ±64% dari semua nasabah yang akan membuka deposito, dengan tingkat keberhasilan 35,7% per telepon. Menelepon secara acak hanya memberi 11,3% (lift ±3,2×).

## Keputusan penting
- **`duration` dibuang** (leakage). Lama telepon baru diketahui *setelah* telepon, dan sangat menentukan target (durasi 0 → selalu "no"). Dengan `duration`, skor akan tampak jauh lebih tinggi, tapi model tidak bisa dipakai sebelum menelepon.
- `pdays = 999` berarti "belum pernah dihubungi", bukan 999 hari. Diubah menjadi fitur `was_contacted_before` + `pdays = −1`.
- `"unknown"` dibiarkan sebagai kategori sendiri: `default = unknown` punya tingkat "yes" 5,2% vs 12,9%, jadi informatif.
- Threshold (0,117) dipilih dari prediksi *out-of-fold* data dev (batas 20% teratas), bukan dari test.

## Analisis error (`notebooks/02_error_analysis.py`)
- **Kontak via telephone** jauh lebih sulit: PR-AUC 0,25 vs 0,49 (cellular), dan hanya 5,8% yang ditelepon model.
- **Bulan Mei**: 42% dari FN berasal dari Mei (bulan dengan volume terbesar dan tingkat "yes" terendah, 6,6%). Model hampir tidak memilih nasabah di bulan ini.
- Nasabah dengan `poutcome = success` hampir selalu tertangkap (recall 100%); kegagalan terbanyak ada di nasabah baru (`nonexistent`, 92% dari FN) yang informasinya paling sedikit.
- FN cenderung terjadi saat suku bunga (euribor) tinggi (median 4,86 vs 0,88 di TP).

## Keterbatasan
- **Pergeseran waktu.** Data terurut waktu, dan tingkat "yes" naik dari 6,4% (80% data awal) ke 30,8% (20% data akhir) seiring krisis ekonomi. Dilatih di periode awal dan diuji di periode akhir, ROC-AUC turun 0,80 → **0,75** dan model meremehkan peluang (prediksi rata-rata 0,25 vs kenyataan 0,31). Untuk produksi: validasi berbasis waktu, monitoring drift, dan retraining berkala (Week 36).
- Split acak dipakai untuk evaluasi utama, sehingga angka di atas kemungkinan **lebih optimis** daripada performa di masa depan.
- Data dari satu bank di satu negara, 2008–2010. Tidak otomatis berlaku untuk konteks lain.
- Prediksi bersifat asosiasi; model tidak mengatakan bahwa menelepon *menyebabkan* nasabah membuka deposito (perlu eksperimen/uplift modeling).
- Fitur sensitif (umur, status pernikahan, pekerjaan) dipakai. Sebelum dipakai sungguhan, cek keadilan antar kelompok.

## Struktur
```
p1-bank-marketing/
├── src/bank_marketing/
│   ├── config.py       # path, kolom, seed
│   ├── data.py         # download, validasi, pembersihan
│   ├── features.py     # feature engineering + pipeline preprocessing
│   ├── train.py        # CV kandidat, pemilihan, evaluasi test, simpan artefak
│   └── predict.py      # inference
├── notebooks/          # 01_eda.py, 02_error_analysis.py (format # %%)
├── tests/              # pytest
└── artifacts/          # model.joblib, metadata.json (tidak di-commit)
```

## Reproduksi
Dari root repo bootcamp (environment `uv`):
```powershell
uv sync
uv run python projects/p1-bank-marketing/src/bank_marketing/train.py      # unduh data (±1 MB) + latih, ±1 menit
uv run python projects/p1-bank-marketing/src/bank_marketing/predict.py    # contoh inference
uv run pytest projects/p1-bank-marketing                                  # 11 tes
uv run python projects/p1-bank-marketing/notebooks/02_error_analysis.py
```
`artifacts/metadata.json` menyimpan seed, hash SHA-256 data, versi library, skor CV semua kandidat, threshold, dan skor test. Kalau hash datanya sama, hasilnya identik.

## Sitasi
Moro, S., Cortez, P., & Rita, P. (2014). *A data-driven approach to predict the success of bank telemarketing.* Decision Support Systems, 62, 22–31.
