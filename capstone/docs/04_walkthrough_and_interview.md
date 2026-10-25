# 04 — Technical Walkthrough dan Persiapan Interview (Week 42)

## Walkthrough 5 menit (latih dengan suara, pakai timer)
1. **Masalah (30 detik).** "Menyaring SMS spam sebelum sampai ke pengguna. Kesalahan paling mahal adalah menahan pesan normal seperti OTP, jadi saya mengoptimalkan recall spam dengan precision minimal 98%."
2. **Data (45 detik).** "UCI SMS Spam Collection, CC BY 4.0, 5.574 pesan, 13% spam. Saya menemukan 414 duplikat dan menghapusnya sebelum split, karena 12,5% pesan test ternyata juga ada di train."
3. **Eksperimen (60 detik).** "Dummy, Naive Bayes, LogReg kata, LogReg kata+karakter, SVM. 5-fold CV, PR-AUC. N-gram karakter memberi peningkatan nyata (0,972 → 0,984). SVM sedikit lebih tinggi tapi dalam 1 std, jadi saya memilih LogReg yang lebih sederhana. Threshold dipilih dari prediksi out-of-fold."
4. **Hasil (45 detik).** "Test PR-AUC 0,99; precision dan recall 96,9%: 4 FP dan 4 FN dari 1.032 pesan. Precision sedikit di bawah target. Dengan 128 pesan ditandai, itu setara 1–2 pesan, jadi saya menambahkan zona 'warn' supaya pesan yang meragukan tetap terkirim."
5. **Sistem (60 detik).** "Go gateway untuk validasi dan aturan bisnis, Python untuk model, Postgres untuk log tanpa isi pesan. Timeout 2 detik, fail-open kalau model mati. Docker Compose, tes di CI, versi model di setiap respons untuk penelusuran."
6. **Keterbatasan (45 detik).** "Data 2011 berbahasa Inggris; spam bergaya percakapan lolos; ada label noise; tidak memakai sinyal pengirim. Langkah berikutnya: data baru dan monitoring drift."

## Pertanyaan yang hampir pasti muncul
<details><summary>Kenapa tidak memakai accuracy?</summary>
87% pesan adalah ham; model yang selalu menjawab "ham" mendapat accuracy 87% dengan recall spam 0. PR-AUC dan precision/recall pada threshold mencerminkan kemampuan menangkap kelas langka dan biaya FP.
</details>

<details><summary>Bagaimana kamu tahu modelmu tidak overfit?</summary>
Skor CV (PR-AUC 0,984 ± 0,010) dan test (0,992) konsisten. Test hanya dipakai sekali. Duplikat dibuang sebelum split. Seluruh preprocessing ada di dalam pipeline, sehingga TF-IDF di-fit hanya pada data train setiap fold.
</details>

<details><summary>Apa itu data leakage? Contoh dari proyekmu.</summary>
Informasi yang tidak tersedia saat prediksi masuk ke training/evaluasi. Di capstone: pesan duplikat di train dan test. Di P1: `duration` (lama telepon baru diketahui setelah menelepon). Di Week 9: seleksi fitur di seluruh data memberi akurasi 0,76 pada data acak.
</details>

<details><summary>Cross-validation: kenapa dan jenis apa?</summary>
Satu split bisa beruntung atau sial; K-fold memberi mean ± std. StratifiedKFold karena spam hanya 12%. Untuk data waktu: TimeSeriesSplit; untuk entitas berulang: GroupKFold.
</details>

<details><summary>Bagaimana memilih threshold?</summary>
Dari prediksi out-of-fold di dev: recall tertinggi dengan precision ≥ 98%. Bukan dari test. Kalau biaya FP/FN diketahui dalam uang, minimalkan biaya total (seperti Week 8).
</details>

<details><summary>Kenapa Go + Python, bukan Python saja?</summary>
Python untuk ekosistem ML; Go untuk API publik (performa, binary 16 MB, concurrency, tim backend). Aturan bisnis bisa berubah tanpa menyentuh model. Untuk tim kecil, FastAPI saja juga cukup; ini trade-off kompleksitas operasional.
</details>

<details><summary>Model mati di produksi. Apa yang terjadi?</summary>
Readiness gagal → orchestrator berhenti mengirim traffic. Gateway timeout 2 detik → fail-open (pesan terkirim dengan flag degraded) dan log warning. Kebijakan fail-open dipilih karena memblokir semua SMS lebih buruk daripada meloloskan spam sementara.
</details>

<details><summary>Bagaimana tahu model memburuk tanpa label?</summary>
Pantau distribusi skor dan % quarantine/warn, panjang pesan (PSI), proporsi pesan dalam zona abu-abu, laporan "bukan spam" dari pengguna. Label baru (dari laporan pengguna/sampling manual) untuk precision/recall berkala.
</details>

<details><summary>Kenapa tidak memakai transformer/LLM?</summary>
Baseline TF-IDF sudah PR-AUC 0,99 dengan latensi 1,7 ms di CPU dan model 1,2 MB. Transformer mungkin membantu pada spam bergaya percakapan, tapi biayanya ±50–100× latensi + GPU/RAM lebih besar. Saya akan mencobanya sebagai kandidat (`--with-embeddings`) dan hanya memakainya kalau peningkatan recall pada precision yang sama sepadan dengan biayanya (Week 28).
</details>

## Latihan teknis dengan batas waktu
| Topik | Latihan | Waktu |
|---|---|---|
| SQL | Top-1 produk per kota (`ROW_NUMBER`), pertumbuhan bulanan (`LAG`), kota tanpa order (`LEFT JOIN ... IS NULL`) | 10 menit/soal |
| SQL | Dari tabel `sms_predictions`: % quarantine per jam, rata-rata skor per versi model | 10 menit |
| Python | Two Sum, Valid Anagram, Top K Frequent, Group Anagrams, Merge Intervals ([NeetCode](https://neetcode.io/)) | 20 menit/soal |
| Pandas | Bersihkan `sales_raw.csv` dari nol (Week 1) | 30 menit |
| ML | Tulis pipeline + CV + GridSearch untuk churn dari nol | 30 menit |

Contoh SQL monitoring (jalankan di Postgres capstone, `localhost:5434`):
```sql
SELECT date_trunc('hour', created_at) AS jam,
       model_version,
       COUNT(*)                                  AS n,
       ROUND(AVG(spam_probability)::numeric, 3)  AS skor_rata2,
       ROUND(100.0 * AVG(is_spam), 1)            AS pct_spam,
       PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms
FROM sms_predictions
GROUP BY 1, 2
ORDER BY 1 DESC;
```

## Gap analysis vs lowongan Junior ML Engineer
Isi setelah membaca 10–20 lowongan target:
| Skill yang sering diminta | Status | Bukti di repo | Rencana |
|---|---|---|---|
| Python, pandas, SQL | ✅ | Week 1–2, P1 | — |
| scikit-learn, evaluasi, CV | ✅ | Week 7–12, P1, capstone | — |
| PyTorch | ✅ dasar | Week 20–25, P2 | Model lebih besar di GPU |
| NLP / Hugging Face / RAG | ✅ dasar | Week 27–30 | Proyek RAG dengan evaluasi |
| Docker, CI, REST API | ✅ | Phase 6, capstone | — |
| Cloud (AWS/GCP/Azure) | ⬜ | — | Deploy di 1 cloud besar (free tier) |
| Kubernetes | ⬜ | — | Opsional: minikube |
| Airflow / orchestration | ⬜ | — | Opsional |
| Feature store, streaming | ⬜ | — | Tidak wajib untuk junior |
