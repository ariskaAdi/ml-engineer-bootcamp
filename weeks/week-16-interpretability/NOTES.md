# Week 16 — Interpretabilitas dan Eksperimen Terkontrol

## File dan cara menjalankan
`uv run python weeks/week-16-interpretability/interpretability.py` → `outputs/week16_*.png` + log eksperimen `outputs/experiments_week16.jsonl`.
Model: HistGradientBoosting (lr 0,05, depth 3) untuk churn, val ROC-AUC ±0,82.
Helper baru: [src/mlbootcamp/experiments.py](../../src/mlbootcamp/experiments.py) — `log_experiment(...)` menulis satu baris JSON per eksperimen.

## Tiga hal yang berbeda (exit criteria)
| | Pertanyaan | Alat | Contoh |
|---|---|---|---|
| **Asosiasi prediktif** | Apakah X bergerak bersama y di data? | Korelasi, plot | Pelanggan dengan banyak komplain lebih sering churn |
| **Penjelasan model** | Fitur apa yang dipakai **model** untuk prediksi ini? | Permutation importance, PDP, SHAP | Model menaikkan skor churn karena `support_calls = 5` |
| **Kausalitas** | Kalau kita **mengubah** X, apakah y berubah? | Eksperimen acak (A/B test), atau asumsi kausal yang kuat | Kalau komplain diselesaikan lebih cepat, apakah churn turun? |

Alat interpretabilitas hanya menjawab baris kedua. Penjelasan itu jujur tentang **model**, bukan tentang **dunia**.

## Demo: fitur akibat, bukan sebab
Ditambahkan `visited_cancel_page` (pelanggan membuka halaman berhenti langganan). Fitur ini adalah **akibat** dari niat churn.
- ROC-AUC melonjak 0,82 → **0,979**; permutation importance 0,49 (jauh di atas fitur lain).
- "Intervensi" memblokir halaman (fitur = 0 untuk semua): **model** memprediksi churn turun 28,5% → 3,2%.
- **Kenyataan** (dari generator): churn tetap 28,1%, karena niat berhenti tidak berubah.

Penjelasan model benar ("model memakai fitur ini"), tapi kalau dipakai sebagai saran tindakan, hasilnya salah total. Fitur seperti ini juga sering merupakan **leakage**: kalau kita butuh prediksi sebulan sebelumnya, data kunjungan halaman cancel belum ada.

## Permutation importance
Acak satu kolom di data **validasi**, lalu ukur seberapa besar ROC-AUC turun. Diulang 20×, sehingga ada mean ± std.
| Fitur | Penurunan AUC |
|---|---|
| contract | 0,106 |
| tenure_months | 0,056 |
| internet_service | 0,031 |
| support_calls | 0,018 |
| monthly_charges | 0,016 |
| total_charges | ≈ 0 |

`total_charges` ≈ 0 **bukan** berarti tidak berhubungan dengan churn. Ia berkorelasi 0,79 dengan `tenure_months`, sehingga ketika satu diacak, yang lain masih membawa informasinya. Fitur yang berkorelasi berbagi (dan saling menyembunyikan) importance.
Kelebihan: berlaku untuk model apa pun dan dihitung pada kolom asli (sebelum one-hot).

## Partial dependence plot (PDP)
Rata-rata prediksi kalau satu fitur diubah ke nilai tertentu untuk semua baris, sementara fitur lain tetap (`week16_2_pdp.png`): peluang churn turun seiring tenure, naik seiring tagihan dan jumlah komplain.
Keterbatasan: menghasilkan kombinasi yang tidak realistis kalau fitur berkorelasi (tenure 70 bulan dengan total_charges 300), dan rata-rata bisa menyembunyikan efek yang berbeda antar kelompok.

## SHAP
SHAP membagi **satu prediksi** menjadi kontribusi setiap fitur (berbasis Shapley value dari teori permainan):
`base value + Σ kontribusi = output model (log-odds)`.
Pelanggan paling berisiko (tenure 3 bulan, 5 komplain, kontrak bulanan, fiber, tagihan 97):
base −1,352; tenure +1,085; support_calls +0,842; Month-to-month +0,629; monthly_charges +0,478; Fiber +0,381; ... → total **2,118** = logit(p) ✔ (p ≈ 0,89).

Beeswarm plot (`week16_3_shap_beeswarm.png`): setiap titik = satu pelanggan. Warna = nilai fitur, posisi = dorongan ke arah churn (+) atau tidak churn (−).

Keterbatasan SHAP:
- Menjelaskan model, bukan dunia (lihat demo di atas).
- Fitur berkorelasi → kontribusi dibagi dengan cara yang bergantung pada metode/asumsi.
- Satuan log-odds tidak intuitif bagi orang bisnis.
- Mahal untuk model non-pohon dan data besar.
- Catatan teknis: untuk HistGradientBoosting dengan kategori native, TreeExplainer memberi nilai yang tidak konsisten. Di script dipakai one-hot, lalu dicek bahwa `base + Σ = logit(p)`. Selalu lakukan cek penjumlahan ini.

## Eksperimen terkontrol
Aturan: ubah **satu** faktor, gunakan fold CV yang **sama**, lalu bandingkan per fold (berpasangan).
| Hipotesis | Δ AUC | Per fold | Kesimpulan |
|---|---|---|---|
| Hapus `total_charges` (redundan?) | +0,0003 | +/− campur | Tidak berbeda → boleh dihapus (model lebih sederhana) |
| Hapus `support_calls` | −0,0104 | negatif di 5/5 fold | Fitur ini benar-benar dipakai |
| `max_iter` 100 → 300 | −0,0023 | campur | Tidak berbeda; lebih banyak pohon tidak membantu |

Setiap eksperimen dicatat dengan: waktu, hipotesis, apa yang diubah, metrik, kesimpulan (`outputs/experiments_week16.jsonl`, bisa dibaca dengan `pd.read_json(path, lines=True)`).
Kenapa per fold? Variasi **antar** fold (±0,01) lebih besar daripada efek yang diuji. Membandingkan pada fold yang sama menghilangkan variasi itu, sehingga perbedaan kecil yang konsisten bisa terdeteksi.

## Analogi untuk orang awam
- **Asosiasi** → payung yang terbuka sering muncul bersama jalan basah.
- **Penjelasan model** → model memakai "ada payung terbuka" untuk menebak "jalan basah". Itu jujur tentang cara kerja model.
- **Kausalitas** → menutup semua payung tidak akan membuat jalan kering. Sama seperti memblokir halaman cancel yang tidak menghentikan niat pelanggan untuk berhenti.
- **Permutation importance** → mengacak satu bahan resep lalu melihat seberapa rusak rasanya.
- **SHAP** → membagi tagihan makan bersama secara adil: berapa "kontribusi" setiap fitur terhadap skor akhir satu pelanggan.
- **Eksperimen terkontrol** → mengubah satu bumbu saja per percobaan, dengan kompor dan bahan yang sama, supaya tahu persis apa penyebab perubahan rasa.

## Exit criteria — tes diri
- [ ] Jelaskan dengan contoh beda asosiasi, penjelasan model, dan kausalitas.
- [ ] Jelaskan kenapa SHAP tinggi untuk `visited_cancel_page` tidak berarti memblokir halaman itu akan mengurangi churn.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
