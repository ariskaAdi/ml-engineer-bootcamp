# 02 — Model Report (Week 40)

Reproduksi: `uv run python capstone/ml/src/spam/train.py` (±2 menit CPU) → `capstone/ml/artifacts/`.

## Data setelah pembersihan
5.574 → **5.160** pesan (414 duplikat dibuang), spam 12,4%. Dev 4.128 / test 1.032.

## Perbandingan kandidat (5-fold CV di data dev)
| Model | PR-AUC | ± std | Recall @ precision ≥ 98% | ± std | Waktu CV |
|---|---|---|---|---|---|
| Dummy | 0,125 | 0,000 | 0,000 | 0 | 6 s |
| TF-IDF kata + NB | 0,969 | 0,014 | 0,893 | 0,028 | 7 s |
| TF-IDF kata + LogReg | 0,972 | 0,013 | 0,870 | 0,072 | 6 s |
| **TF-IDF kata+karakter + LogReg** | 0,984 | 0,010 | **0,936** | 0,047 | 47 s |
| TF-IDF kata+karakter + SVM terkalibrasi | **0,985** | 0,008 | 0,930 | 0,047 | 50 s |

**Keputusan:** SVM PR-AUC tertinggi (0,9854). LogReg kata+karakter (0,9835) berada dalam 1 std (≥ 0,9771) dan lebih sederhana (tanpa kalibrasi tambahan, recall@P98 sedikit lebih tinggi), jadi **LogReg kata+karakter dipilih**. Model hanya-kata (NB, LogReg) berada di luar 1 std → n-gram karakter memberi peningkatan nyata.

**Catatan proses:** percobaan pertama memilih berdasarkan recall@P98, dan aturan 1-SE memilih Naive Bayes karena metrik itu sangat bervariasi antar fold (std 0,05–0,07: dengan ±100 spam per fold, satu pesan bisa menggeser recall beberapa persen). Metrik pemilihan diganti ke PR-AUC yang lebih stabil, dan target precision dipakai untuk memilih **threshold**. Keputusan ini dicatat supaya tidak tersembunyi.

## Threshold
Dari prediksi out-of-fold di dev: **0,430** = recall tertinggi dengan precision ≥ 98%.

## Hasil akhir di test (sekali)
| PR-AUC | Precision | Recall | TP | FP | FN | TN | Latensi |
|---|---|---|---|---|---|---|---|
| **0,9915** | 0,969 | 0,969 | 124 | **4** | 4 | 900 | 1,7 ms/pesan (CPU) |

Precision di test (96,9%) **sedikit di bawah target 98%**. Dengan hanya 128 pesan yang ditandai, 1 FP = 0,8 poin precision, jadi selisih ini setara ±1–2 pesan dan masih dalam variasi yang terlihat di CV. Sistem tetap dirancang untuk melindungi pengguna dari FP (lihat zona *warn*).

## Analisis error
**False positive (pesan normal yang akan ditahan):**
| p | Pesan | Analisis |
|---|---|---|
| 0,81 | "I (Career Tel) have added u as a contact on INDYAROCKS.COM to send FREE SMS. To remove ... sms NO to ..." | Secara isi memang mirip spam; label `ham` bisa diperdebatkan (*label noise*) |
| 0,70 | "Glad to see your reply." | Pesan sangat pendek → sedikit fitur, model ragu |
| 0,57 | "Your daily text from me – a favour this time" | Frasa "daily text" sering muncul di spam langganan |
| 0,54 | "Please protect yourself from e-threats. SIB never asks for ... PIN ..." | Notifikasi bank asli memakai kosakata yang sama dengan penipuan (PIN, password) |

**False negative (spam yang lolos):**
| p | Pesan | Analisis |
|---|---|---|
| 0,06 | "Hi this is Amy, we will be sending you a free phone number ... adult parties" | Gaya percakapan personal, tanpa nomor premium |
| 0,06 | "Sorry I missed your call let's talk when you have the time. I'm on 07090201529" | Meniru pesan normal; satu-satunya sinyal adalah nomor 070 |
| 0,14 | "dating:i have had two of these. Only started after i sent a text ..." | Label `spam` dapat diperdebatkan |
| 0,31 | "Babe: U want me dont u baby! ... Txt XXX SLO(4msgs)" | Di zona abu-abu; tertangkap kalau threshold sedikit lebih rendah |

**Pola:**
- Error rate naik untuk pesan panjang 80–160 karakter (1,5%) dibanding < 40 karakter (0,3%).
- 26 pesan berada di zona abu-abu (0,22–0,86), dan 58% di antaranya spam. Zona ini terlalu campur untuk keputusan otomatis → gateway memakai **3 aksi**: `quarantine` (p ≥ 0,90), `warn` (threshold ≤ p < 0,90: tetap dikirim dengan label), `deliver`.
- Spam yang meniru percakapan personal adalah kelemahan utama model berbasis kata/karakter.

## Cek leakage duplikat
Tanpa dedup, **12,5%** pesan test juga ada persis di train. PR-AUC tanpa dedup 0,9907 vs dengan dedup 0,9915 (test set berbeda, jadi tidak bisa dibandingkan langsung). Efeknya kecil di sini karena duplikat kebanyakan template spam yang memang mudah. Tetap dibuang: kalau tidak, angka evaluasi bisa menilai hafalan, bukan generalisasi.

## Keterbatasan dan perbaikan yang bisa diuji
1. **Data lama (2011), bahasa Inggris.** Spam modern (link pendek, e-wallet, bahasa Indonesia/campuran) tidak terwakili. → Kumpulkan dan beri label data baru, pantau drift panjang pesan & skor.
2. **Spam bergaya personal** lolos. → Fitur nomor telepon (prefix premium), atau model transformer kecil; uji di fold yang sama.
3. **Label noise.** → Tinjau ulang label FP/FN dengan aturan pelabelan tertulis.
4. Model tidak melihat pengirim, frekuensi pengiriman, atau reputasi nomor. Sinyal non-teks ini biasanya sangat kuat di sistem nyata.
5. Threshold berbasis ±130 spam di test. → Perkirakan precision dengan interval kepercayaan (bootstrap) sebelum menjanjikan angka ke stakeholder.
