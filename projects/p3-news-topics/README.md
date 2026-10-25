# News Topics — TF-IDF vs Embeddings vs DistilBERT

Portfolio Project 3 (Week 30, Track NLP). Mengklasifikasikan berita (judul + ringkasan) ke 4 topik: **World, Sports, Business, Sci/Tech**, dan membandingkan tiga pendekatan dari yang paling sederhana sampai yang paling berat.

## Data
[AG News](https://huggingface.co/datasets/fancyzhx/ag_news) (Zhang, Zhao & LeCun, 2015) via Hugging Face: 120.000 train + 7.600 test, 4 kelas seimbang.
**Lisensi:** disediakan untuk riset/pendidikan non-komersial. Proyek ini untuk belajar; jangan dipakai komersial tanpa izin.
Split: 5.000 validasi diambil dari train (seed 42); test resmi 7.600 dipakai sekali. Teks dibersihkan dari artefak HTML (`#39;`, `&lt;b&gt;`, `\`).

## Hasil (test, 7.600 berita)
| Pendekatan | Data train | Accuracy | Macro-F1 | Waktu fit (CPU) | Inference (CPU) | Ukuran |
|---|---|---|---|---|---|---|
| **TF-IDF (1–2 gram) + LogReg** | 115.000 | **0,923** | **0,922** | 4 menit | **0,3 ms** | ±30 MB |
| TF-IDF + LogReg | 8.000 | 0,883 | 0,883 | 9 detik | 0,3 ms | kecil |
| e5-small embedding + LogReg | 8.000 | 0,898 | 0,898 | 9 menit | 51 ms | 470 MB |
| DistilBERT fine-tune (1 epoch) | 8.000 | 0,907 | 0,907 | 51 menit | 87 ms | 270 MB |
(Waktu diukur di laptop CPU 4 thread yang sedang menjalankan job lain; di GPU DistilBERT ±1–2 menit.)

### Interpretasi
1. **Dengan data yang sama (8k), model pretrained menang**: DistilBERT 0,907 > e5 0,898 > TF-IDF 0,883. Pretraining membawa pengetahuan bahasa yang tidak bisa dipelajari dari 8k contoh saja.
2. **Tapi TF-IDF dengan seluruh data (115k) mengalahkan semuanya (0,923)**, dengan inference ±300× lebih cepat. Data berlabel yang banyak sering lebih berharga daripada model yang lebih besar.
3. Pilihan yang wajar untuk produksi di sini: **TF-IDF + LogReg (full)**. Kalau data berlabel sedikit (kasus nyata yang umum), transformer atau embedding layak dipertimbangkan. Langkah berikutnya: DistilBERT dengan seluruh data di GPU, yang kemungkinan melewati 0,94.

## Analisis error
| Kelas | F1 TF-IDF full | F1 DistilBERT 8k |
|---|---|---|
| Sports | 0,967 | 0,973 |
| World | 0,901 | 0,908 |
| Sci/Tech | 0,868 | 0,883 |
| Business | 0,856 | 0,864 |
- **Business ↔ Sci/Tech** adalah sumber error terbesar (171 + 188 untuk TF-IDF): berita perusahaan teknologi (Dell, IBM, PeopleSoft) memang berada di dua topik sekaligus. Ini batas yang ambigu di label, bukan hanya kelemahan model.
- Contoh salah: "New $50 bill designed to counter counterfeits" (Business → Sci/Tech), "Bryant Makes First Appearance at Trial" (berlabel Sci/Tech?! → Sports): ada **label noise** di AG News.
- TF-IDF dan DistilBERT berbeda pendapat pada 6,5% berita. Di kasus itu TF-IDF benar 58% dan DistilBERT 35% (sisanya keduanya salah). Ini sejalan dengan perbedaan jumlah data latihnya.

## Inference
```powershell
uv run python projects/p3-news-topics/src/news_topics/predict.py "Apple unveils new chip for laptops"
# Sci/Tech  0.97  Apple unveils new chip for laptops
uv run python projects/p3-news-topics/src/news_topics/predict.py --model distilbert "Inflation slows as central bank holds rates"
# Business  0.96  ...
```
`predict()` mengembalikan label, confidence, dan skor semua kelas; teks kosong ditolak.

## Reproduksi
```powershell
uv sync
uv run python projects/p3-news-topics/src/news_topics/train.py               # semua (±1 jam CPU)
uv run python projects/p3-news-topics/src/news_topics/train.py --skip-bert --skip-emb   # TF-IDF saja (±5 menit)
uv run python projects/p3-news-topics/notebooks/error_analysis.py
uv run pytest projects/p3-news-topics                                        # 7 tes
```
Output: `artifacts/results.json`, `artifacts/test_predictions.csv`, `artifacts/tfidf_logreg.joblib`, `artifacts/distilbert/`.

## Keterbatasan
- Berita tahun 2004–2005 berbahasa Inggris. Topik dan kosakata berita sekarang berbeda (AI, kripto), jadi akurasi di berita baru kemungkinan lebih rendah.
- Label noise dan batas Business/Sci-Tech yang kabur membatasi akurasi maksimal.
- DistilBERT hanya dilatih 1 epoch dengan 8k data karena keterbatasan CPU; perbandingan dengan TF-IDF full tidak setara dalam jumlah data (justru itu yang ingin ditunjukkan).
- Lisensi non-komersial.
