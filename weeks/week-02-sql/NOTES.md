# Week 02 — SQL untuk Data

## File dan cara menjalankan
```powershell
uv run python weeks/week-01-numpy-pandas/05-reference-solution.py   # data/sales_clean.csv
uv run python scripts/load_sales_to_sqlite.py                       # data/sales.db (3 tabel)
uv run python weeks/week-02-sql/01_run_sql_practice.py              # semua latihan di sql/01–03
uv run python weeks/week-02-sql/02_business_questions.py            # mini-task: SQL vs Pandas
```
| File | Isi |
|---|---|
| `sql/01_basics.sql` | SELECT, WHERE, ORDER BY, GROUP BY, HAVING, agregat |
| `sql/02_joins.sql` | INNER JOIN, LEFT JOIN, subquery, CTE |
| `sql/03_window_functions.sql` | ROW_NUMBER, RANK, DENSE_RANK, running total, LAG, moving average |
| `sql/04_business_questions.sql` | 5 pertanyaan bisnis |
| `03_postgres_optional.py` + `docker-compose.yml` | Data yang sama di PostgreSQL (Docker) |

## Skema database
```
products (product_id PK, product_name, category, list_price)
cities   (city_id PK, city_name, province)          ← Medan sengaja tanpa order
orders   (order_id PK, order_date, product_id FK, city_id FK, quantity, unit_price)
```
Data Week 1 dipecah menjadi 3 tabel (**normalisasi**): nama produk dan kategori disimpan sekali di `products`, dan `orders` hanya menyimpan `product_id`. Keuntungannya: tidak ada nama yang ditulis berbeda-beda (masalah `"JAKARTA"` di Week 1 tidak bisa terjadi), dan mengganti nama cukup di satu tempat. Konsekuensinya: untuk analisis kita perlu **JOIN**.

`PRIMARY KEY` = identitas unik setiap baris. `FOREIGN KEY` (`REFERENCES`) = nilai harus ada di tabel lain. `CHECK (quantity > 0)` = database menolak data tidak valid; aturan validasi Week 1 dipindahkan ke database.

## Konsep inti

**Urutan eksekusi SQL** berbeda dari urutan menulis:
`FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT`.
Karena itu, alias dari SELECT tidak bisa dipakai di WHERE, dan **WHERE** menyaring baris *sebelum* dikelompokkan, sedangkan **HAVING** menyaring grup *sesudah* agregasi.

**INNER vs LEFT JOIN.** INNER hanya menyimpan baris yang punya pasangan. LEFT menyimpan semua baris tabel kiri; kolom dari kanan menjadi `NULL` kalau tidak ada pasangan. Lihat `inner_join_revenue_per_kota` vs `left_join_revenue_per_kota`: Medan hilang di INNER dan muncul dengan 0 di LEFT.
Jebakan: `COUNT(*)` menghitung baris (Medan = 1), sedangkan `COUNT(o.order_id)` mengabaikan NULL (Medan = 0).

**Subquery dan CTE.** Subquery = query di dalam query. CTE (`WITH nama AS (...)`) = subquery yang diberi nama, supaya bisa dibaca dari atas ke bawah dan dipakai ulang.

**Window function** menghitung sesuatu dari baris lain *tanpa* menggabungkan baris.
- `GROUP BY` → 975 order jadi 5 baris (satu per produk).
- `SUM(...) OVER (PARTITION BY ...)` → tetap 975 baris, setiap baris mendapat total grupnya.
- `ROW_NUMBER` (1,2,3,4), `RANK` (1,2,2,4), `DENSE_RANK` (1,2,2,3) berbeda hanya kalau ada nilai seri.
- `LAG(x)` = nilai baris sebelumnya → dipakai untuk pertumbuhan bulan-ke-bulan.
- `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` = jendela 7 baris → moving average.
- "Top-N per grup" (produk terlaris per kota) = `ROW_NUMBER() OVER (PARTITION BY kota ORDER BY revenue DESC)` lalu `WHERE rn = 1`. Pola ini sangat sering ditanyakan di interview.

**SQL ↔ Pandas**
| SQL | Pandas |
|---|---|
| `WHERE` | `df[mask]` |
| `GROUP BY ... SUM` | `groupby().sum()` |
| `HAVING` | filter setelah `groupby` |
| `JOIN` / `LEFT JOIN` | `merge(how="inner")` / `merge(how="left")` |
| `SUM() OVER (PARTITION BY)` | `groupby().transform("sum")` |
| `ROW_NUMBER ... WHERE rn=1` | `sort_values().groupby().head(1)` |
| `LAG` | `shift(1)` |

**Keamanan: parameter, bukan f-string.** `pd.read_sql_query("... WHERE city = ?", con, params=(city,))`. Kalau nilai dari user dimasukkan dengan f-string, input seperti `x' OR '1'='1` bisa mengubah arti query (*SQL injection*).

## Hasil mini-task (SQL == Pandas, dicek dengan `assert`)
1. Revenue per bulan: tertinggi 2026-06 (21,7 jt); Juni naik 24,1% dari Mei (dari `LAG`).
2. Madu Hutan = 45,1% revenue.
3. Jakarta teratas; Medan 0 order (hanya terlihat dengan LEFT JOIN).
4. Di **setiap** kota, produk terlaris adalah Madu Hutan.
5. Pangsa kategori per bulan berubah-ubah: Minuman mengalahkan Lainnya di April dan Juni.

## SQLite vs PostgreSQL (opsional)
| | SQLite | PostgreSQL |
|---|---|---|
| Bentuk | 1 file (`data/sales.db`), tanpa server | Server, banyak user bersamaan |
| Tanggal per bulan | `strftime('%Y-%m', d)` | `date_trunc('month', d)` |
| Parameter | `?` | `%s` (psycopg) atau `:nama` (SQLAlchemy) |
| Tipe uang | REAL (float) | `NUMERIC(12,2)` (tepat, tanpa error pembulatan) |

Cara mencoba: nyalakan Docker Desktop, lalu `docker compose -f weeks/week-02-sql/docker-compose.yml up -d`, lalu `uv run python weeks/week-02-sql/03_postgres_optional.py`. Koneksi bisa diubah lewat env var `DATABASE_URL`.

## Jebakan umum
- Lupa kolom non-agregat di `GROUP BY` → SQLite diam saja (hasil acak); PostgreSQL error. PostgreSQL lebih benar.
- `NULL = NULL` bukan TRUE. Pakai `IS NULL`.
- Pembagian integer: `1/2 = 0`. Tulis `100.0 * a / b`.
- `pd.read_sql` tanpa `parse_dates` → tanggal terbaca sebagai teks.

## Analogi untuk orang awam
- **Database dengan beberapa tabel** → lemari arsip dengan laci terpisah: laci produk, laci kota, laci pesanan. Pesanan cukup menulis "produk nomor 3" daripada menulis ulang semua detail produk.
- **JOIN** → menyatukan kembali informasi dari laci-laci itu dengan mencocokkan nomor ID, seperti mencocokkan nomor kursi di tiket dengan daftar penonton.
- **INNER vs LEFT JOIN** → daftar hadir: INNER hanya mencatat orang yang diundang **dan** datang; LEFT mencatat semua undangan, yang tidak datang ditandai kosong.
- **WHERE vs HAVING** → WHERE menyaring orang di pintu masuk; HAVING menyaring kelompok setelah semua orang dikelompokkan per meja.
- **Window function** → papan klasemen liga: setiap tim tetap punya barisnya sendiri, tapi ditambah informasi peringkat dan total poin dibandingkan tim lain.
- **SQL injection** → formulir yang menyuruh petugas "tulis apa pun yang dikatakan tamu sebagai perintah". Parameter `?` membuat petugas memperlakukan ucapan tamu hanya sebagai data, bukan perintah.

## Exit criteria — tes diri
Tanpa melihat file `sql/`, tulis query untuk:
- [ ] Revenue per kategori per kota (2 JOIN + GROUP BY).
- [ ] Kota yang revenue-nya di atas rata-rata semua kota (CTE/subquery).
- [ ] Order ke-N setiap kota menurut tanggal (ROW_NUMBER + PARTITION BY).

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
