# Week 30 — Portfolio Project 3: NLP

Kode: [projects/p3-news-topics/](../../projects/p3-news-topics/) — baca [README-nya](../../projects/p3-news-topics/README.md).

## Hasil singkat (AG News, 4 topik, test 7.600)
| Pendekatan | Data | Accuracy | Inference CPU |
|---|---|---|---|
| TF-IDF + LogReg | 115k | **0,923** | 0,3 ms |
| TF-IDF + LogReg | 8k | 0,883 | 0,3 ms |
| e5 embedding + LogReg | 8k | 0,898 | 51 ms |
| DistilBERT fine-tune | 8k | 0,907 | 87 ms |

## Pelajaran utama
1. **Baseline dulu.** TF-IDF dengan seluruh data adalah model terbaik sekaligus termurah. Tanpa baseline ini, DistilBERT (0,907) akan terlihat "bagus" padahal kalah.
2. **Data vs model.** Dengan data sama (8k), pretrained model menang (+2,4 poin). Dengan data 14× lebih banyak, model sederhana menang. Saat data berlabel mahal, pretrained model sangat berharga.
3. **Embedding beku + LogReg** adalah jalan tengah: tidak perlu fine-tuning/GPU, hasil di antara keduanya.
4. **Analisis error menemukan masalah label**: Business ↔ Sci/Tech ambigu, dan ada label yang jelas salah. Tidak semua error bisa diperbaiki dengan model yang lebih besar.

## Checklist TODO Week 30
- [x] Dataset berlisensi (AG News, non-komersial — dicatat di README)
- [x] Baseline sederhana (TF-IDF, 2 ukuran data)
- [x] Fine-tune pretrained model (DistilBERT) + pendekatan retrieval-style (embedding)
- [x] Metrik yang sesuai (accuracy + macro-F1, kelas seimbang) dan contoh benar/salah
- [x] Dokumentasi setup dan inference

## Exit criteria — tes diri
- [ ] Jalankan versi cepat (`--skip-bert --skip-emb`) dan inference dari README.
- [ ] Jelaskan kenapa TF-IDF full mengalahkan DistilBERT di sini, dan kapan hasilnya akan terbalik.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
