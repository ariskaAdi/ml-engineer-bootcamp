# Week 08 — Klasifikasi

## File dan cara menjalankan
`uv run python weeks/week-08-classification/classification_churn.py` → 4 plot `outputs/week08_*.png`.
Data: `make_churn()` — 5000 pelanggan, 28,2% churn (mirip dataset Telco Customer Churn di Kaggle).

## Tiga model
| Model | Cara kerja | Perlu scaling? |
|---|---|---|
| **Logistic Regression** | Skor linear `z = X·w + b` → sigmoid `1/(1+e^-z)` → peluang 0–1 | Ya (supaya regularisasi adil & cepat konvergen) |
| **KNN** | Lihat k tetangga terdekat di train, ambil suara mayoritas | **Wajib** — berbasis jarak |
| **Decision Tree** | Serangkaian pertanyaan ya/tidak (`contract == Month-to-month?`) yang dipilih untuk memisahkan kelas sebaik mungkin | Tidak |

## Hasil validation (threshold 0,5)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Dummy (selalu "tidak churn") | **0,718** | 0 | 0 | 0 | 0,50 | 0,28 |
| LogisticRegression | 0,788 | 0,665 | 0,500 | 0,571 | **0,842** | **0,677** |
| KNN (k=15) | 0,770 | 0,629 | 0,450 | 0,525 | 0,810 | 0,599 |
| DecisionTree (depth 4) | 0,770 | 0,610 | 0,511 | 0,556 | 0,792 | 0,533 |

Dummy mendapat accuracy 71,8% **tanpa menangkap satu pun pelanggan churn**. Dengan kelas tidak seimbang, accuracy hampir tidak berguna.

## Confusion matrix dan metrik
```
                 tebak: stay   tebak: churn
benar: stay        TN=647        FP=71
benar: churn       FN=141        TP=141
```
- **Precision** = TP/(TP+FP) = 0,665 → dari pelanggan yang *kita tandai* churn, 66,5% benar. Penting kalau tindakan untuk positif mahal.
- **Recall** = TP/(TP+FN) = 0,500 → dari pelanggan yang *benar-benar* churn, hanya setengah yang tertangkap. Penting kalau melewatkan positif mahal.
- **F1** = rata-rata harmonik precision & recall. Berguna kalau keduanya sama penting dan biaya tidak diketahui.
- **ROC-AUC** = peluang model memberi skor lebih tinggi ke pelanggan churn acak daripada pelanggan non-churn acak. Tidak bergantung threshold.
- **PR-AUC** (average precision) = lebih jujur daripada ROC-AUC kalau positif sedikit. Baseline-nya = proporsi positif (0,28), bukan 0,5.

## Threshold dan biaya (exit criteria)
Model memberi **peluang**, kita yang memilih batasnya. Biaya: FP (penawaran sia-sia) = Rp 50 rb, FN (pelanggan hilang) = Rp 600 rb.
| Threshold | Precision | Recall | Biaya di TEST |
|---|---|---|---|
| 0,50 (default) | 0,664 | 0,534 | Rp 82,4 jt |
| **0,09** (dipilih di validation) | 0,386 | 0,947 | **Rp 30,2 jt** |

Karena FN 12× lebih mahal daripada FP, lebih baik "banyak salah alarm" asal hampir semua churn tertangkap. Secara teori, threshold optimal untuk peluang yang terkalibrasi = `FP / (FP + FN)` = 50/650 ≈ 0,077 — dekat dengan 0,09 yang ditemukan.
Kalau biayanya terbalik (misalnya penawaran berupa diskon besar), threshold akan naik dan precision lebih penting.

Aturan memilih metrik:
- FN mahal (fraud, penyakit, churn bernilai tinggi) → kejar **recall**, threshold rendah.
- FP mahal (spam filter yang membuang email penting, menolak kredit nasabah baik) → kejar **precision**, threshold tinggi.
- Tahu biayanya → minimalkan **biaya total** langsung.

Threshold dipilih di **validation**, lalu dicek sekali di **test**.

## Class imbalance
28% positif tergolong "sedikit tidak seimbang". Dampaknya: accuracy menyesatkan, dan threshold 0,5 cenderung meremehkan kelas minoritas. Cara menangani: `stratify=y` saat split, metrik yang tepat (PR-AUC, recall, biaya), atur threshold, `class_weight` (Week 15).

## Apa yang dipelajari model
Koefisien LogReg (fitur distandarkan): kontrak 2 tahun (−0,98), tenure (−0,68), tech support (−0,36) menurunkan peluang churn. Tagihan bulanan (+0,54), jumlah komplain (+0,45), fiber (+0,31), e-check (+0,22) menaikkannya. Ini cocok dengan pola yang ditanam di generator, jadi model menemukan pola yang benar.

## KNN dan scaling
| k | AUC dengan scaling | tanpa scaling |
|---|---|---|
| 1 | 0,656 | 0,592 |
| 15 | 0,810 | 0,713 |
| 51 | 0,831 | 0,712 |
Tanpa scaling, jarak didominasi `total_charges` (ribuan), sehingga fitur lain diabaikan. k=1 terlalu sensitif terhadap noise (overfit); k besar lebih halus.

## Analogi untuk orang awam
- **Logistic regression** → juri yang memberi skor, lalu skor diubah menjadi peluang 0–100%.
- **KNN** → menilai seseorang dari 15 tetangga terdekatnya: kalau kebanyakan tetangganya pindah langganan, dia juga kemungkinan pindah.
- **Decision tree** → permainan "20 pertanyaan": "Kontraknya bulanan? Ya. Sudah langganan < 6 bulan? Ya. → kemungkinan besar churn."
- **Precision vs recall** → jala ikan: recall = berapa banyak ikan di laut yang tertangkap; precision = berapa banyak isi jala yang benar-benar ikan (bukan sampah).
- **Threshold** → tinggi palang di pintu: palang rendah, banyak yang masuk (recall tinggi, banyak alarm palsu); palang tinggi, sedikit yang masuk (precision tinggi, banyak yang terlewat).
- **Accuracy menyesatkan** → satpam yang selalu bilang "aman" di kota yang jarang ada maling: benar 99% waktu, tapi tidak pernah menangkap satu maling pun.

## Exit criteria — tes diri
- [ ] Untuk kasus spam filter, fraud, dan diagnosis penyakit: metrik apa dan threshold tinggi atau rendah? Kenapa?
- [ ] Hitung precision, recall, F1 dari confusion matrix di atas dengan tangan.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
