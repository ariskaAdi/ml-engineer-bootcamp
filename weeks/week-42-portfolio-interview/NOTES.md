# Week 42 — Portfolio dan Latihan Interview

Bahan utama: [capstone/docs/04_walkthrough_and_interview.md](../../capstone/docs/04_walkthrough_and_interview.md) — skrip walkthrough 5 menit, 9 pertanyaan interview dengan jawaban, latihan SQL/Python dengan batas waktu, dan tabel gap analysis.

## Checklist portfolio
- [ ] README capstone: masalah, setup, metrik, keterbatasan, **link demo** (isi setelah deploy).
- [ ] README root bootcamp menautkan P1, P2, P3, capstone (sudah ada di [README.md](../../README.md)).
- [ ] Pindahkan P1/P2/P3/capstone ke repo GitHub terpisah kalau ingin dilihat terpisah oleh recruiter, dan tautkan dari CV.
- [ ] Badge CI hijau di README.
- [ ] Hapus file besar/rahasia dari histori sebelum repo dibuat publik (`.env`, data, model besar).

## Latihan menjelaskan (tanpa catatan, rekam diri sendiri)
| Topik | Contoh dari repo ini |
|---|---|
| Overfitting | Week 10 polinomial (train 0,010 vs val 1,46); Week 23 val loss 0,53 → 1,09 |
| Data leakage | `duration` (P1), duplikat SMS (capstone), SelectKBest di seluruh data (Week 9), split acak di data waktu (Week 10) |
| Cross-validation | Week 10 (KFold vs TimeSeriesSplit 5,7 vs 24,9), paired fold comparison (Week 16) |
| Model selection | Aturan 1-SE: P1 (LogReg vs RF), capstone (LogReg vs SVM) |
| Metric trade-off | Week 8 (threshold biaya 0,09), Week 15 (precision 30%), capstone (precision 98% + zona warn) |

## Analogi untuk orang awam
- **Walkthrough 5 menit** → *elevator pitch*: kalau kamu tidak bisa menjelaskan proyekmu selama naik lift, pendengar kehilangan minat.
- **Gap analysis** → peta pendakian: menandai pos yang sudah dilewati dan pos yang masih harus dicapai sebelum puncak (lowongan target).
- **Portfolio di GitHub** → etalase toko: recruiter melihat barang yang sudah jadi, bukan hanya membaca brosur (CV).

## Penilaian akhir (isi sendiri)
- [ ] Reviewer bisa mereproduksi inference capstone dari README.
- [ ] Setiap keputusan model didukung angka eksperimen.
- [ ] Peran Go dan Python jelas.
- [ ] Bisa menjelaskan trade-off dan kegagalan tanpa bantuan AI.

## Rencana setelah bootcamp (isi sendiri dari gap analysis)
1.
2.
3.
