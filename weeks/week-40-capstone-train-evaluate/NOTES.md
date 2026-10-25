# Week 40 — Capstone: Training dan Evaluasi

Laporan: [capstone/docs/02_model_report.md](../../capstone/docs/02_model_report.md). Kode: [capstone/ml/src/spam/train.py](../../capstone/ml/src/spam/train.py), analisis error: [capstone/ml/notebooks/error_analysis.py](../../capstone/ml/notebooks/error_analysis.py).

```powershell
uv run python capstone/ml/src/spam/train.py
uv run python capstone/ml/notebooks/error_analysis.py
```

## Ringkasan
| Langkah | Hasil |
|---|---|
| Baseline + kandidat | Dummy 0,125 · NB 0,969 · LogReg kata 0,972 · **LogReg kata+karakter 0,984** · SVM 0,985 (PR-AUC CV) |
| Pemilihan | LogReg kata+karakter (dalam 1 std dari SVM, lebih sederhana) |
| Threshold (OOF dev) | 0,43 untuk precision ≥ 98% |
| Test (sekali) | PR-AUC 0,992 · precision 0,969 · recall 0,969 · 4 FP · 4 FN |
| Analisis error | FP: notifikasi bank, pesan sangat pendek, label yang bisa diperdebatkan. FN: spam bergaya percakapan |

## Pelajaran
1. **N-gram karakter** menangkap ejaan spam (`£1000`, `txt`, nomor premium) yang tidak tertangkap oleh kata saja: peningkatan di luar 1 std.
2. **Metrik pemilihan harus stabil.** Recall@P98 per fold bervariasi ±5–7 poin (±100 spam per fold), sehingga aturan 1-SE memilih model yang lebih lemah. PR-AUC lebih stabil; target precision dipakai untuk threshold.
3. **Angka test yang di bawah target dilaporkan apa adanya** (96,9% vs 98%), lalu sistem dirancang untuk menanggulanginya (zona *warn*), bukan dengan memutar threshold di test.
4. **Analisis error memengaruhi desain produk**: 26 pesan di zona abu-abu, 58% spam → terlalu campur untuk keputusan otomatis → 3 aksi di gateway.

## Exit (pribadi)
- [ ] Jalankan ulang training dan dapatkan angka yang sama (seed + hash data di `metadata.json`).
- [ ] Jelaskan setiap FP/FN di laporan dengan kata-kata sendiri.
