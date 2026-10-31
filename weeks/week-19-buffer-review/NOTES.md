# Week 19 — Buffer dan Review Phase 3

## Checklist exit criteria Phase 3
| Week | Exit criteria | Bukti |
|---|---|---|
| 14 | Kenapa cluster ≠ segmen bisnis | K-Means memberi 4 cluster bahkan pada data acak |
| 15 | Identifikasi kegagalan + perbaikan yang bisa diuji | FN malam hari → `is_night` → PR-AUC +0,05 di 5/5 fold |
| 16 | Bedakan asosiasi, penjelasan model, kausalitas | Demo `visited_cancel_page` |
| 17–18 | Proyek reproducible dengan modul + tes | `projects/p1-bank-marketing` |

## Polish Portfolio 1 (pilih yang belum)
- [ ] Salin ke repo GitHub terpisah, lalu tautkan dari README bootcamp.
- [ ] Tambah 1–2 plot di README (PR curve, capture curve).
- [ ] Coba validasi berbasis waktu (TimeSeriesSplit) dan laporkan bersama split acak.
- [ ] Kalibrasi peluang (`CalibratedClassifierCV`) dan bandingkan dengan periode akhir.

## Mulai kebiasaan latihan coding interview (30–45 menit/minggu)
Mulai dari soal easy di [NeetCode](https://neetcode.io/) kategori *Arrays & Hashing*. Pola yang paling sering keluar di ML engineer junior:
- Hash map untuk menghitung frekuensi (`collections.Counter`).
- Two pointers / sliding window.
- Sorting + binary search.
- Ditambah SQL: top-N per grup (`ROW_NUMBER`), running total, `LEFT JOIN ... IS NULL`.

Contoh latihan pertama (tulis tanpa melihat):
```python
def top_k_frequent(nums: list[int], k: int) -> list[int]:
    from collections import Counter
    return [x for x, _ in Counter(nums).most_common(k)]

assert top_k_frequent([1, 1, 1, 2, 2, 3], 2) == [1, 2]
```

## Analogi untuk orang awam
- **Polish portfolio** → mengecat ulang dan merapikan rumah contoh sebelum dibuka untuk pengunjung.
- **Latihan coding interview** → latihan tangga nada untuk pemain musik: membosankan, tapi membuat jarimu lancar saat tampil.

## Catatan pribadi (isi sendiri)
- Exit criteria yang diulang:
- Rencana Phase 4:
