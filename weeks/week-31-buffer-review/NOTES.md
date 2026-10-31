# Week 31 — Buffer dan Review Phase 5 (Track B: NLP)

## Checklist exit criteria
| Week | Exit criteria | Bukti |
|---|---|---|
| 27 | Teks → angka, baseline reproducible | TF-IDF + LogReg test acc 0,82 (SST-2) |
| 28 | Apakah peningkatan transformer sepadan dengan biayanya? | Tabel akurasi vs latensi vs ukuran di Week 28 |
| 29 | Kapan fine-tuning vs retrieval vs prompt | Tabel keputusan di Week 29; RAG atas catatan sendiri |
| 30 | Proyek reproducible + evaluasi + keterbatasan | [projects/p3-news-topics](../../projects/p3-news-topics/) |

## Kuis
<details><summary>Kenapa bag-of-words tidak bisa membedakan "not good, the film" dan "the film, not good"?</summary>
BoW hanya menghitung kata; urutan hilang, jadi vektornya identik. Bigram/transformer mengembalikan informasi urutan.
</details>
<details><summary>Kenapa subword tokenization tidak punya masalah kata di luar vocabulary?</summary>
Kata langka dipecah menjadi potongan yang dikenal ("unwatchable" → un, ##watch, ##able), sampai tingkat karakter kalau perlu.
</details>
<details><summary>Hit@3 retrieval tinggi tapi jawaban RAG masih salah. Kemungkinan penyebab?</summary>
Chunk benar tapi terlalu panjang/terpotong, prompt tidak membatasi jawaban ke konteks, LLM terlalu kecil, atau evaluasi retrieval tidak mewakili pertanyaan sungguhan.
</details>
<details><summary>Kapan TF-IDF lebih baik daripada transformer?</summary>
Data sedikit, latensi/biaya ketat, butuh interpretabilitas, teks dengan kata kunci kuat (spam), atau peningkatan akurasinya kecil dibanding biayanya.
</details>

## Opsional: lihat Track A
Skim [week-27-29-track-a-cv/](../week-27-29-track-a-cv/) (konvolusi dengan tangan, transfer learning ResNet18, OpenCV + analisis error), supaya paham dasar computer vision.

## Analogi untuk orang awam
- **Review NLP** → seperti mengulang pelajaran bahasa: dari menghitung kata (TF-IDF), memahami konteks (transformer), sampai mencari referensi sebelum menjawab (RAG).

## Catatan pribadi (isi sendiri)
- Exit criteria yang diulang:
