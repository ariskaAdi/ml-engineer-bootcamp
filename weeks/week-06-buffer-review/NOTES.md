# Week 06 — Buffer dan Review Phase 1

Minggu ini **tidak ada materi baru**. Pakai untuk menutup exit criteria yang belum lulus. Kalau semuanya sudah lulus, lewati minggu ini.

## Checklist exit criteria Phase 1
| Week | Exit criteria | Cara mengetes |
|---|---|---|
| 0 | Project baru dari nol + push | Buat folder kosong, `uv init`, `uv add pandas`, `git init`, push ke repo baru |
| 1 | Bersihkan dataset dan jelaskan setiap transformasi | Ulangi mini-task sales di notebook kosong (target: 975 baris) |
| 2 | Query multi-tabel tanpa melihat sintaks | 3 soal di `week-02-sql/NOTES.md` |
| 3 | Jelaskan temuan, bedakan korelasi dan kausalitas | Jelaskan contoh kamar tidur vs luas bangunan |
| 4 | Kenapa sampel bervariasi; arti p-value | Kuis di bawah |
| 5 | Bagaimana GD mengurangi loss | Tulis ulang loop GD dari nol |

## Kuis cepat (jawab dengan suara dulu, baru buka jawabannya)

<details><summary>1. Kenapa <code>(3,1) + (4,)</code> berhasil tapi <code>(3,) + (4,)</code> gagal?</summary>

Broadcasting membandingkan shape dari kanan. `(3,1)` vs `(4,)` → 1 vs 4 (boleh, 1 direntangkan), 3 vs (kosong, dianggap 1) → hasil `(3,4)`. `(3,)` vs `(4,)` → 3 vs 4, keduanya bukan 1 → gagal.
</details>

<details><summary>2. Kenapa harga kosong diisi median per produk, bukan median semua data?</summary>

Harga ditentukan oleh produk. Median gabungan (25.000) akan membuat harga Madu Hutan (85.000) jauh terlalu rendah, sehingga revenue salah.
</details>

<details><summary>3. Bedanya WHERE dan HAVING?</summary>

WHERE menyaring baris sebelum GROUP BY; HAVING menyaring grup setelah agregasi (boleh memakai `COUNT`, `SUM`).
</details>

<details><summary>4. Kenapa LEFT JOIN + COUNT(*) memberi angka 1 untuk kota tanpa order?</summary>

LEFT JOIN tetap membuat satu baris untuk kota itu (kolom order bernilai NULL). `COUNT(*)` menghitung baris, `COUNT(o.order_id)` mengabaikan NULL → 0.
</details>

<details><summary>5. Tulis query top-1 produk per kota.</summary>

```sql
WITH r AS (
  SELECT city_id, product_id, SUM(quantity*unit_price) AS rev,
         ROW_NUMBER() OVER (PARTITION BY city_id ORDER BY SUM(quantity*unit_price) DESC) AS rn
  FROM orders GROUP BY city_id, product_id)
SELECT * FROM r WHERE rn = 1;
```
</details>

<details><summary>6. Korelasi kamar tidur–harga 0,68. Apakah menambah kamar menaikkan harga?</summary>

Tidak bisa disimpulkan. Luas bangunan adalah confounder: rumah luas punya lebih banyak kamar **dan** lebih mahal. Dalam kelompok luas yang sama, korelasinya hampir hilang.
</details>

<details><summary>7. Apa itu standard error dan bagaimana ia berubah kalau n dikali 4?</summary>

Std dari statistik (misalnya mean) antar sampel = σ/√n. n × 4 → SE ÷ 2.
</details>

<details><summary>8. p = 0,03. Artinya "97% yakin B lebih baik"?</summary>

Bukan. Artinya: kalau sebenarnya tidak ada beda, selisih sebesar ini atau lebih ekstrem hanya muncul ±3% dari waktu. p-value tidak memberi peluang hipotesis benar dan tidak mengukur besar efek.
</details>

<details><summary>9. Kenapa sampel survei 13.000 orang bisa lebih buruk daripada 500 orang?</summary>

Sampel besar yang bias (hanya pengguna aplikasi) secara sistematis salah. Menambah data tidak menghilangkan bias; sampel acak kecil tidak bias.
</details>

<details><summary>10. Kenapa update gradient descent memakai tanda minus?</summary>

Gradien menunjuk ke arah loss naik paling cepat. Kita ingin loss turun, jadi melangkah ke arah sebaliknya: `w ← w − lr·∇L`.
</details>

<details><summary>11. Bagaimana kamu tahu turunan yang kamu tulis benar?</summary>

Bandingkan dengan turunan numerik `(f(w+h) − f(w−h)) / 2h` untuk h kecil (gradient check).
</details>

## Rapikan repo
- [ ] Setiap notebook: Restart Kernel → Run All tanpa error.
- [ ] `git status` tidak berisi `data/`, `.venv/`, `outputs/`, `models/`.
- [ ] Setiap folder week punya `NOTES.md` dengan weekly review terisi.
- [ ] Commit: `week-06: phase 1 review`.

## Catatan pribadi (isi sendiri)
- Exit criteria yang diulang:
- Yang masih membingungkan:
