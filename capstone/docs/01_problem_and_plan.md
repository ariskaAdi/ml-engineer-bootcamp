# 01 — Problem Definition dan Experiment Plan (Week 39)

## Masalah
Operator/aplikasi SMS ingin menyaring SMS spam (undian palsu, promosi premium, penipuan) **sebelum** sampai ke pengguna, tanpa menahan pesan normal.

**Kenapa ML?** Aturan kata kunci ("FREE", "WIN") mudah diakali spammer dan salah menangkap pesan normal ("free tonight?"). Model teks belajar kombinasi sinyal (angka premium, simbol £, gaya penulisan) dari contoh berlabel.

## Data
- [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — Almeida & Hidalgo (2011), UCI. **Lisensi CC BY 4.0** (boleh dipakai ulang, termasuk komersial, dengan atribusi).
- 5.574 SMS berbahasa Inggris (sebagian besar Singapura & Inggris), label `ham`/`spam`.
- Cek kualitas:
  - 747 spam (13,4%) → **tidak seimbang**.
  - **414 baris duplikat** (teks persis sama; banyak template spam) → harus dihapus **sebelum** split.
  - Tidak ada nilai kosong. Panjang pesan 2–910 karakter.
  - Beberapa label dapat diperdebatkan (notifikasi bank, iklan dari kontak sendiri) → ada *label noise* kecil.
  - Data dikumpulkan ±2011; gaya spam sekarang berbeda (link, e-wallet, bahasa Indonesia) → **keterbatasan utama**.

## Target, metrik, dan biaya kesalahan
| Kesalahan | Akibat | Biaya |
|---|---|---|
| **False positive** (pesan normal ditahan) | Pengguna melewatkan OTP, pesan keluarga, notifikasi bank | **Tinggi** |
| False negative (spam lolos) | Gangguan, risiko penipuan | Sedang |

- **Metrik pemilihan model:** PR-AUC (rata-rata 5-fold CV). Stabil dan tidak bergantung threshold; cocok untuk kelas langka.
- **Titik operasi:** threshold dengan recall spam maksimal pada **precision ≥ 98%** di prediksi out-of-fold data dev.
- **Metrik laporan:** precision, recall, jumlah FP/FN di test; latensi per pesan.

## Strategi validasi
1. Hapus duplikat (`drop_duplicates(text)`).
2. Split stratified 80% dev / 20% test (seed 42). **Test disentuh sekali.**
3. 5-fold StratifiedKFold di dev untuk membandingkan kandidat dan memilih threshold.
4. Aturan keputusan (ditulis sebelum eksperimen): PR-AUC tertinggi; kalau model lebih sederhana berada dalam 1 std, pilih yang sederhana.

## Baseline dan kandidat
| # | Model | Hipotesis |
|---|---|---|
| 0 | Dummy (prior) | Batas bawah: PR-AUC ≈ proporsi spam |
| 1 | TF-IDF kata + Complement Naive Bayes | Baseline klasik spam, sangat cepat |
| 2 | TF-IDF kata (1–2 gram) + LogReg | Baseline linear yang kuat |
| 3 | TF-IDF kata + **karakter (2–5 gram)** + LogReg | Karakter menangkap ejaan aneh (`fr33`, `£1000`, `txt`) dan nomor premium |
| 4 | Sama seperti 3, dengan LinearSVC terkalibrasi | Margin-based, sering sedikit lebih baik di teks |
| 5 | (opsional, `--with-embeddings`) e5-small embedding + LogReg | Makna kalimat; lebih lambat |

## Rencana analisis error
- Baca semua FP dan FN di test.
- Error per panjang pesan.
- Zona abu-abu sekitar threshold → kandidat aturan "warn" (kirim dengan peringatan) daripada menahan.

## Definisi selesai
Model package + service Python + gateway Go + Postgres berjalan dengan satu perintah Docker; tes di CI; dokumentasi reproduksi; demo publik (opsional).
