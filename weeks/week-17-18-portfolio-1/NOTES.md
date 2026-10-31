# Week 17–18 — Portfolio Project 1: Tabular ML

Kode proyek ada di [projects/p1-bank-marketing/](../../projects/p1-bank-marketing/) — baca [README-nya](../../projects/p1-bank-marketing/README.md) dulu. Catatan ini menjelaskan **kenapa** proyeknya dibangun seperti itu.

> Panduan GETTING_STARTED menyarankan portfolio di repo terpisah. Folder `projects/p1-bank-marketing/` sudah dibuat mandiri (punya `src/`, `tests/`, README), jadi nanti tinggal disalin ke repo baru.

## Week 17 — Data dan modeling
| Langkah | Di mana | Catatan |
|---|---|---|
| Pilih dataset berlisensi | `config.py` (`DATA_URL`) | UCI Bank Marketing, **CC BY 4.0** (boleh dipakai ulang dengan atribusi). Selalu cek lisensi sebelum memakai dataset Kaggle |
| Target & metrik | README | 11,3% positif → PR-AUC, bukan accuracy. Metrik bisnis: *capture @ top 20%* karena kapasitas call center terbatas |
| EDA | `notebooks/01_eda.py` | Menemukan leakage `duration`, kode `pdays=999`, data terurut waktu |
| Preprocessing | `features.py` | Semua di dalam Pipeline |
| Baseline + ≥3 model | `train.py` | Dummy, LogReg, RandomForest, HistGB |
| CV + test terpisah | `train.py` | 5-fold stratified di 80% dev; test 20% sekali |

### Leakage yang paling penting: `duration`
Lama telepon sangat prediktif (durasi 0 → selalu "no"; kuintil durasi tertinggi → 33% "yes"). Tapi nilainya baru ada **setelah** telepon. Model yang memakainya tidak bisa menjawab "siapa yang perlu ditelepon?". Pertanyaan yang selalu diajukan untuk setiap fitur: **"Apakah nilai ini sudah tersedia pada saat prediksi dibutuhkan?"**

### Kenapa Logistic Regression, bukan Random Forest?
RF 0,467 ± 0,018 vs LogReg 0,450. Selisih 0,017 < 1 std → secara statistik tidak bisa dibedakan dengan yakin. Model sederhana dipilih (aturan 1-SE): lebih cepat, lebih kecil, lebih mudah di-deploy (Phase 6), lebih mudah dijelaskan. Kalau selisihnya besar dan konsisten, RF/HGB yang dipilih.

## Week 18 — Analisis, kualitas kode, dokumentasi
### Analisis error
Lihat bagian "Analisis error" di README: model lemah untuk kontak telephone, bulan Mei, dan nasabah baru. **Uji pergeseran waktu** (latih di 80% awal, uji di 20% akhir) memperlihatkan keterbatasan terbesar: ROC-AUC turun 0,80 → 0,75 dan base rate berubah 6% → 31%.

### Dari notebook ke modul `src/`
| Notebook (dulu) | Modul (sekarang) |
|---|---|
| Path ditulis di mana-mana | `config.py` — satu tempat |
| Sel baca + bersihkan | `data.py` — `load_dataset()` dengan `validate()` yang gagal cepat |
| Sel feature engineering | `features.py` — `add_features()` yang **tidak mengubah input** dan dipakai sama persis saat training & inference |
| Sel training | `train.py` — `main()` menyimpan model + `metadata.json` |
| — | `predict.py` — fungsi `predict(records)` yang dipanggil API di Week 32 |

Type hints (`def predict(records: list[dict]) -> list[dict]`) memperjelas kontrak fungsi dan membantu editor menemukan kesalahan.

### Tes (`uv run pytest projects/p1-bank-marketing` → 11 passed)
- `test_data.py`: target dipetakan ke 0/1, duplikat dibuang, `validate()` menolak kolom hilang / target aneh / umur mustahil.
- `test_features.py`: `duration` dibuang, `pdays=999` dikodekan benar, input tidak diubah, pipeline tetap jalan dengan kategori yang belum pernah dilihat.
- `test_predict.py`: field wajib dicek; output peluang 0–1; *sanity check* perilaku: `poutcome=success` menaikkan peluang.
Tes yang butuh model terlatih otomatis di-skip kalau `artifacts/` belum ada.

`conftest.py` di folder proyek menambahkan `src/` ke `sys.path`, sehingga tes bisa `import bank_marketing` tanpa instalasi paket.

### Reproducibility
`metadata.json`: seed, SHA-256 file data, versi Python/sklearn/pandas, skor semua kandidat, threshold, skor test. Ditambah `uv.lock` untuk versi library yang persis sama.

## Analogi untuk orang awam
- **Portfolio project** → rumah contoh dari kontraktor: calon klien tidak cukup diberi tahu "saya bisa membangun", tapi ingin melihat rumah yang bisa dimasuki.
- **Leakage `duration`** → menebak siapa pemenang lomba lari setelah melihat foto garis finis. Akurat, tapi tidak berguna untuk memasang taruhan sebelum lomba.
- **Memindah kode notebook ke `src/`** → memindahkan resep dari coretan di sobekan kertas ke buku resep yang rapi dan bernomor, sehingga orang lain bisa memasak hal yang sama.
- **Tes otomatis** → alarm asap di dapur: kamu berharap tidak pernah berbunyi, tapi kalau ada yang salah, ia langsung memberi tahu.
- **Pergeseran waktu** → strategi jualan yang laris di musim kemarau belum tentu laris di musim hujan.

## Exit criteria — tes diri
- [ ] Orang lain bisa menjalankan 4 perintah di README dan mendapat angka yang sama.
- [ ] Logika utama ada di modul yang bisa di-import, bukan hanya di notebook.
- [ ] Kamu bisa menjelaskan dalam 2 menit: masalah, metrik, kenapa `duration` dibuang, kenapa LogReg dipilih, keterbatasan terbesar.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
