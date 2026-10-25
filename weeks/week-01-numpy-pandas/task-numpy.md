# Task: Aritmatika dan Aljabar Linear Dasar dengan NumPy

**Level:** Pemula\
**Tujuan:** Berlatih menghitung secara manual terlebih dahulu, lalu
memeriksa hasilnya menggunakan NumPy.\
**Aturan:** Kerjakan bagian *Perhitungan Manual* di kertas atau Markdown
sebelum menjalankan kode Python. Jangan langsung memakai NumPy untuk
mencari jawaban.

## Data awal

Gunakan matriks berikut:

\[ A =
```{=tex}
\begin{bmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12
\end{bmatrix}
```
\]

Di Python:

``` python
import numpy as np

A = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
```

Ingat: indeks NumPy dimulai dari **0**, bukan 1.

------------------------------------------------------------------------

## Bagian 1 --- Memahami bentuk matriks dan indexing

### Perhitungan manual

1.  Berapa jumlah baris dan kolom matriks `A`?
2.  Berapa nilai pada baris ke-2 dan kolom ke-2 jika menghitung seperti
    manusia (mulai dari 1)?
3.  Berapa nilai pada indeks `A[1, 1]`?
4.  Tuliskan semua elemen pada baris pertama.
5.  Tuliskan semua elemen pada kolom ketiga.
6.  Apa hasil jika mengambil baris terakhir?
7.  Apa hasil jika mengambil dua kolom pertama?

### Periksa dengan NumPy

Jalankan satu per satu:

``` python
A.shape
A[1, 1]
A[0, :]
A[:, 2]
A[-1, :]
A[:, :2]
```

Catat apakah hasilnya sama dengan jawaban manual.

------------------------------------------------------------------------

## Bagian 2 --- Aritmatika skalar

Kalikan atau tambahkan **setiap elemen** matriks sesuai instruksi.

### Perhitungan manual

Untuk matriks `A`, hitung:

1.  (A + 2)
2.  (A - 3)
3.  (A `\times 2`{=tex})
4.  (A / 2)

Tuliskan matriks hasil untuk setiap operasi. Contoh: pada (A + 2),
tambahkan 2 ke setiap elemen matriks.

### Periksa dengan NumPy

``` python
A + 2
A - 3
A * 2
A / 2
```

**Pertanyaan:** Apa perbedaan antara `A * 2` dan perkalian matriks yang
akan dipelajari nanti?

------------------------------------------------------------------------

## Bagian 3 --- Penjumlahan dan pengurangan matriks

Gunakan matriks `B` berikut:

\[ B =
```{=tex}
\begin{bmatrix}
2 & 1 & 0 & 1 \\
1 & 2 & 1 & 0 \\
0 & 1 & 2 & 1
\end{bmatrix}
```
\]

``` python
B = np.array([
    [2, 1, 0, 1],
    [1, 2, 1, 0],
    [0, 1, 2, 1]
])
```

### Perhitungan manual

1.  Hitung (A + B).
2.  Hitung (A - B).
3.  Apakah `A` dan `B` memiliki ukuran yang sama?
4.  Menurutmu, mengapa kedua matriks tersebut bisa dijumlahkan dan
    dikurangkan elemen demi elemen?

### Periksa dengan NumPy

``` python
A + B
A - B
```

------------------------------------------------------------------------

## Bagian 4 --- Operasi elemen demi elemen

### Perhitungan manual

Hitung hasil perkalian elemen demi elemen (A `\odot `{=tex}B). Kalikan
elemen yang berada pada posisi yang sama.

Contoh aturan:

\[ (A `\odot `{=tex}B)*{i,j} = A*{i,j} `\times `{=tex}B\_{i,j} \]

1.  Hitung baris pertama hasilnya.
2.  Lanjutkan untuk baris kedua dan ketiga.
3.  Bandingkan hasilnya dengan penjumlahan matriks pada Bagian 3.

### Periksa dengan NumPy

``` python
A * B
```

**Catatan:** `A * B` di NumPy berarti perkalian elemen demi elemen,
bukan perkalian matriks biasa.

------------------------------------------------------------------------

## Bagian 5 --- Transpose matriks

Transpose menukar baris menjadi kolom dan kolom menjadi baris.

### Perhitungan manual

Untuk matriks `A`:

1.  Tuliskan baris pertama `A` sebagai kolom pertama hasil transpose.
2.  Tuliskan baris kedua `A` sebagai kolom kedua hasil transpose.
3.  Tuliskan baris ketiga `A` sebagai kolom ketiga hasil transpose.
4.  Berapa ukuran matriks hasil transpose?

### Periksa dengan NumPy

``` python
A.T
A.T.shape
```

------------------------------------------------------------------------

## Bagian 6 --- Jumlah elemen per baris dan kolom

### Perhitungan manual

Untuk matriks `A`:

1.  Hitung jumlah semua elemen pada baris pertama.
2.  Hitung jumlah semua elemen pada baris kedua.
3.  Hitung jumlah semua elemen pada baris ketiga.
4.  Hitung jumlah elemen pada setiap kolom.
5.  Hitung jumlah seluruh elemen matriks.

### Periksa dengan NumPy

``` python
A.sum(axis=1)  # jumlah per baris
A.sum(axis=0)  # jumlah per kolom
A.sum()        # jumlah seluruh elemen
```

**Petunjuk:** `axis=1` menggabungkan nilai sepanjang kolom untuk
mendapatkan satu hasil per baris. `axis=0` menggabungkan nilai sepanjang
baris untuk mendapatkan satu hasil per kolom.

------------------------------------------------------------------------

## Bagian 7 --- Tantangan campuran

Kerjakan secara manual terlebih dahulu:

1.  Hitung (C = 2A + B).
2.  Hitung (D = A - 2B).
3.  Hitung jumlah semua elemen pada `C`.
4.  Tentukan elemen baris ke-3, kolom ke-4 dari `D` (penomoran manusia
    dimulai dari 1).
5.  Jelaskan langkah yang kamu lakukan untuk memperoleh jawaban nomor 4.

Periksa dengan:

``` python
C = 2 * A + B
D = A - 2 * B

C
D
C.sum()
D[2, 3]
```

------------------------------------------------------------------------

## Bagian 8 --- Refleksi

Jawab dengan kata-katamu sendiri:

1.  Apa perbedaan **baris**, **kolom**, **elemen**, dan **shape**?
2.  Mengapa `A[1, 1]` bukan elemen baris pertama kolom pertama?
3.  Apa perbedaan `A * B` dengan `A + B`?
4.  Apa yang berubah ketika matriks ditranspose?
5.  Kapan operasi aritmatika skalar berguna dalam pengolahan data atau
    machine learning?

------------------------------------------------------------------------

## Checklist penyelesaian

-   [ ] Saya menghitung jawaban secara manual sebelum menjalankan NumPy.
-   [ ] Saya memahami indexing yang dimulai dari 0.
-   [ ] Saya bisa melakukan penjumlahan dan pengurangan matriks
    berukuran sama.
-   [ ] Saya memahami operasi skalar dan operasi elemen demi elemen.
-   [ ] Saya bisa membuat transpose matriks secara manual.
-   [ ] Saya bisa menghitung jumlah elemen per baris, per kolom, dan
    seluruh matriks.
-   [ ] Saya membandingkan jawaban manual dengan hasil NumPy.

**Kriteria selesai:** Usahakan menjawab semua soal tanpa melihat jawaban
orang lain. Jika hasil manual dan NumPy berbeda, periksa kembali operasi
per elemen dan penggunaan indeks.
