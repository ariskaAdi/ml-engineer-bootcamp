# Week 04 — Statistik dan Peluang

## File dan cara menjalankan
`uv run python weeks/week-04-statistics/stats_simulations.py` → angka di terminal + 4 plot di `outputs/week04_*.png`.
Pendekatan minggu ini: setiap rumus **dibuktikan dengan simulasi**. Kalau rumus dan simulasi memberi angka yang sama, kamu benar-benar paham rumusnya.

## 1. Satu angka ringkasan tidak cukup
Tiga toko punya rata-rata transaksi sama (±100), tapi:
- A: normal, sebagian besar transaksi dekat 100.
- B: miring, median ±69; mean ditarik ke atas oleh sedikit transaksi besar.
- C: dua puncak (50 dan 150); **tidak ada** transaksi yang nilainya dekat rata-rata.

Selalu lihat distribusinya (histogram) + beberapa statistik sekaligus: mean, median, std, persentil.
- **Variance** = rata-rata kuadrat jarak ke mean. **Std** = akarnya, satuannya sama dengan data.
- Median tahan outlier, mean tidak.

## 2. Kenapa sampel bervariasi (CLT dan standard error)
Setiap sampel acak berisi orang/transaksi yang berbeda, jadi mean-nya juga berbeda. Hasil simulasi:
- Std dari mean sampel sama dengan rumus **σ/√n** (standard error).
- Sampel 4× lebih besar → error 2× lebih kecil (karena √4 = 2).
- **Central Limit Theorem**: distribusi *mean* sampel mendekati normal walaupun populasinya miring (lihat `week04_2_clt.png`, n=30 sudah cukup normal). Ini alasan banyak uji statistik memakai distribusi normal.

## 3. Sampling bias
Survei lewat aplikasi: 13.287 jawaban → kepuasan 7,27. Sampel acak 500 orang → 6,75. Nilai sebenarnya 6,73.
Sampel besar yang **bias** tetap salah; sampel kecil yang **acak** justru akurat. Menambah data tidak memperbaiki cara pengambilan sampel yang salah.
Di ML, ini muncul sebagai *training data tidak mewakili data produksi*.

## 4. Peluang bersyarat dan Bayes
`P(A|B)` = peluang A **kalau kita tahu** B terjadi.

Bayes: `P(fraud | alarm) = P(alarm | fraud) · P(fraud) / P(alarm)`

Dengan fraud 1%, sensitivitas 95%, salah alarm 5%: P(fraud | alarm) = **16,1%** (rumus = simulasi).
Intuisi: dari 10.000 transaksi, 100 fraud → 95 alarm benar. 9.900 normal → 495 alarm palsu. Jadi 95 / (95 + 495) ≈ 16%.
**Base rate** (seberapa langka kejadiannya) sangat menentukan. Ini dasar dari *precision* di Week 8 dan 15.

## 5. Confidence interval
CI 95% = interval yang dibuat dengan metode yang, kalau diulang pada banyak sampel, menangkap nilai sebenarnya ±95% dari waktu (plot `week04_3_ci.png`: ±95 dari 100 garis biru).
- ✅ Benar: "Metode ini menangkap μ pada 95% sampel."
- ❌ Salah: "Ada peluang 95% μ berada di interval ini." (μ itu tetap; intervalnya yang berubah-ubah.)

**Bootstrap**: ambil ulang sampel dari sampel itu sendiri (dengan pengembalian) ribuan kali → sebaran statistik → ambil persentil 2,5 dan 97,5. Bisa dipakai untuk statistik apa pun (median, rasio) tanpa rumus.

## 6. Uji hipotesis, p-value, A/B test
- **H0** (hipotesis nol): tidak ada perbedaan.
- **p-value** = peluang melihat selisih sebesar ini atau lebih ekstrem **kalau H0 benar**.
- Uji permutasi (acak label A/B 10.000 kali) dan uji z memberi p-value hampir sama (0,76 vs 0,74).

**Apa yang TIDAK berarti p-value:**
- Bukan peluang H0 benar.
- Bukan ukuran besar efek (efek kecil + sampel sangat besar → p kecil).
- p > 0,05 **bukan** bukti "tidak ada efek". Di simulasi, B *benar-benar* lebih baik (11% vs 10%), tapi p = 0,76, karena sampelnya kurang.

**False positive:** kalau tidak ada efek sama sekali, tetap 5% uji memberi p < 0,05 (hasil simulasi: 5,0%). Menguji 20 metrik sekaligus → hampir pasti ada yang "signifikan" secara kebetulan.

**Power** = peluang mendeteksi efek yang memang ada. Untuk 10% → 11%:
| n per grup | power |
|---|---|
| 1.000 | ±11% |
| 5.000 | ±38% |
| 15.000 | ±83% |

Pelajaran A/B test: tentukan ukuran sampel **sebelum** eksperimen, jangan mengintip lalu berhenti saat p < 0,05, dan laporkan besar efek + CI, bukan hanya p-value.

## Exit criteria — tes diri
- [ ] Jelaskan kenapa dua sampel dari populasi yang sama memberi mean berbeda, dan apa yang terjadi kalau n diperbesar.
- [ ] Jelaskan kenapa mean saja tidak cukup (contoh toko C).
- [ ] Jelaskan apa arti dan apa yang **bukan** arti p-value.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
