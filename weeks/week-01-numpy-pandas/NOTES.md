# Week 01 — NumPy dan Pandas

## File di folder ini
| File | Isi |
|---|---|
| `01`–`04` (.ipynb) | Notebook latihanmu sendiri (tidak diubah) |
| `task-numpy.md` | Soal latihan aljabar dasar |
| `05-reference-solution.py` | Kunci jawaban `task-numpy.md` + mini-task sales. Buka setelah mencoba sendiri |

Jalankan: `uv run python scripts/make_sales_data.py` (sekali), lalu `uv run python weeks/week-01-numpy-pandas/05-reference-solution.py`.
Script ini juga menyimpan `data/sales_clean.csv` yang dipakai Week 2 dan 3.

> Semua file `.py` di bootcamp ini memakai penanda `# %%`. Di VS Code, setiap `# %%` menjadi cell yang bisa dijalankan dengan **Run Cell** (seperti notebook), dan hasilnya muncul di panel Interactive.

## Konsep inti (penjelasan singkat)

**Array NumPy vs list Python.** Array menyimpan angka bertipe sama dalam satu blok memori yang rapat. Operasi `q * p` dijalankan oleh kode C dalam satu kali panggilan, bukan oleh loop Python per elemen. Hasilnya ratusan kali lebih cepat.

**Shape dan axis.** `shape=(3, 4)` berarti 3 baris dan 4 kolom. `axis=0` artinya "bergerak turun melewati baris", jadi hasilnya satu angka per kolom. `axis=1` artinya "bergerak ke kanan melewati kolom", jadi hasilnya satu angka per baris.

**Broadcasting.** NumPy membandingkan shape dari kanan. Dua dimensi cocok kalau ukurannya sama atau salah satunya 1. `(3,1) + (4,)` → `(3,1)` vs `(1,4)` → `(3,4)`. `(3,) + (4,)` gagal karena 3 ≠ 4 dan tidak ada yang bernilai 1.

**`*` vs `@`.** `A * B` mengalikan elemen pada posisi yang sama (shape harus sama atau bisa di-broadcast). `A @ B` adalah perkalian matriks (kolom kiri = baris kanan).

**`.loc` vs `.iloc`.** `.loc` memilih berdasarkan label (nama index/kolom). `.iloc` memilih berdasarkan posisi (0, 1, 2, ...). Setelah `drop_duplicates()` atau filter, label index tidak lagi berurutan, jadi `.loc[5]` dan `.iloc[5]` bisa berbeda baris.

## Pembersihan data — setiap langkah dan alasannya
| # | Masalah | Tindakan | Alasan |
|---|---|---|---|
| 1 | 20 baris duplikat | `drop_duplicates()` | Transaksi yang sama terhitung dua kali akan menggelembungkan revenue |
| 2 | `"JAKARTA"`, `"  Jakarta "` | `.str.strip().str.title()` | Satu kota harus jadi satu grup saat `groupby` |
| 3 | `quantity` berisi `"2 pcs"` | buang `" pcs"`, lalu `pd.to_numeric` | Satu nilai teks membuat seluruh kolom bertipe teks |
| 4 | `quantity` negatif (15) | dibuang | Asumsi: salah input. Kalau ternyata retur, seharusnya diperlakukan berbeda; tanyakan ke pemilik data |
| 5 | `order_date` kosong (10) | `to_datetime`, lalu dibuang | Tanpa tanggal, transaksi tidak bisa masuk analisis per bulan |
| 6 | `unit_price` kosong | diisi median harga **produk yang sama** | Harga ditentukan oleh produk; median semua produk akan salah besar untuk Madu Hutan |

Hasil: **1020 → 975 baris**, tanpa nilai kosong.

## Hasil mini-task
- Bulan tertinggi: **2026-06** (Rp 21,7 juta).
- Produk teratas: **Madu Hutan** (Rp 48,9 juta) — 1 dari 5 produk, tapi ±45% revenue, karena harganya paling mahal.
- Kota teratas: **Jakarta** (Rp 29,8 juta, 258 order). Selisih antar kota kecil karena data sintetis memilih kota secara acak merata.
- Kontribusi kategori: Lainnya 45,1%, Minuman 37,3%, Makanan 17,6%.

## Statistik deskriptif tanpa `.describe()`
- Mean = jumlah / n.
- Variance sampel = Σ(x − mean)² / (n − 1). Dibagi n − 1 (bukan n) karena mean juga dihitung dari sampel yang sama, jadi sebaran cenderung sedikit terlalu kecil; n − 1 mengoreksinya. Pandas memakai `ddof=1`, NumPy `np.std` memakai `ddof=0`. Ini sering menjadi sumber angka yang "beda sedikit".
- Median = nilai tengah setelah diurutkan. Revenue: mean 111.154 vs median 75.000 → distribusinya **miring ke kanan** (ada transaksi besar dari Madu Hutan yang menarik mean ke atas).

## Experiment log
| # | Hipotesis | Yang diubah | Hasil | Kesimpulan |
|---|---|---|---|---|
| 1 | Membuang baris kosong menurunkan revenue | `dropna` vs isi median per produk | 104,2 jt vs 108,4 jt (−3,87%) | Membuang data = kehilangan transaksi nyata. Isi, lalu dokumentasikan |
| 2 | Vectorized lebih cepat | loop Python vs `q * p` | ±1,2 ms vs ±7 µs | Hindari loop per baris di NumPy/Pandas |

## Jebakan umum
- `FileNotFoundError` di notebook: notebook berjalan dari folder minggunya. Pakai `ROOT` atau `mlbootcamp.paths.DATA_DIR`.
- `SettingWithCopyWarning`: terjadi saat mengubah hasil filter. Pakai `df = df[...].copy()` atau `.loc[mask, kolom] = ...`.
- `np.std(x)` ≠ `df["x"].std()` karena `ddof`.

## Analogi untuk orang awam
- **Array NumPy vs list Python** → list Python seperti kasir yang menghitung belanjaan satu per satu; NumPy seperti mesin pemindai yang memproses seluruh keranjang sekaligus.
- **Shape dan axis** → tabel nilai rapor: `axis=0` = menjumlahkan ke bawah (satu hasil per mata pelajaran), `axis=1` = menjumlahkan ke samping (satu hasil per siswa).
- **Broadcasting** → guru berkata "semua siswa dapat tambahan 5 poin": satu angka otomatis "direntangkan" ke seluruh kelas tanpa ditulis berulang.
- **Data cleaning** → mencuci sayur sebelum dimasak: buang yang busuk (duplikat), rapikan potongannya (nama kota), dan isi yang kurang dengan perkiraan yang masuk akal (harga median per produk).
- **Mean vs median** → di ruangan berisi 9 karyawan dan 1 miliarder, rata-rata gaji tampak tinggi sekali, padahal median (orang di tengah) lebih jujur menggambarkan "orang biasa".

## Exit criteria — tes diri
- [ ] Bisa membersihkan `sales_raw.csv` dari notebook kosong dan menjelaskan 6 langkah di atas.
- [ ] Bisa menghitung mean, median, dan std tanpa `.describe()`.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
