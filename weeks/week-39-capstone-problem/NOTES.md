# Week 39 — Capstone: Definisi Masalah

Dokumen utama: [capstone/docs/01_problem_and_plan.md](../../capstone/docs/01_problem_and_plan.md). Proyek: [capstone/](../../capstone/).

## Keputusan minggu ini
| Item TODO | Keputusan |
|---|---|
| Masalah praktis (terkait track NLP) | Filter SMS spam |
| Lisensi & kualitas data | UCI SMS Spam Collection, **CC BY 4.0**; 414 duplikat, 13% spam, sedikit label noise |
| Baseline | Dummy (PR-AUC ≈ 0,12) dan TF-IDF + Naive Bayes |
| Metrik | Pemilihan: PR-AUC (5-fold CV). Operasional: recall pada precision ≥ 98% |
| Validasi | Dedup → split stratified 80/20 → 5-fold CV di dev → test sekali |
| Rencana eksperimen | 5 kandidat + hipotesis masing-masing, ditulis **sebelum** menjalankan |

## Kenapa menulis rencana sebelum eksperimen?
Supaya tidak "mencari-cari" metrik atau split yang kebetulan membuat model terlihat bagus. Kalau rencana berubah (di Week 40, metrik pemilihan diganti dari recall@P98 ke PR-AUC), perubahan dan alasannya dicatat terbuka di laporan.

## Kenapa spam SMS cocok untuk capstone
- Dataset kecil dan berlisensi jelas → bisa dilatih di laptop.
- Trade-off metrik yang nyata (FP mahal) → latihan threshold dan desain aksi bisnis.
- Model ringan → bisa di-deploy di free tier, latensi rendah.
- Ada jebakan nyata (duplikat, label noise, data lama) untuk dibahas di interview.

## Exit (pribadi)
- [ ] Bisa menjelaskan masalah, data, metrik, dan validasi dalam 1 menit.
