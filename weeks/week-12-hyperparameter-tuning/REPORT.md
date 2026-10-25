# Experiment Report — Churn Model (Week 12)

**Tujuan:** memprediksi pelanggan yang akan churn supaya tim retensi bisa menghubungi mereka.
**Data:** `make_churn()` — 5000 pelanggan, 28,2% churn. Split: 80% dev (untuk CV) / 20% test (disimpan), stratified, seed 42.
**Metrik utama:** ROC-AUC (kemampuan meranking pelanggan berisiko; threshold dipilih terpisah berdasarkan biaya, lihat Week 8). Metrik pendukung: PR-AUC.
**Validasi:** 5-fold StratifiedKFold pada data dev.

## Hasil
| Eksperimen | Ruang pencarian | CV ROC-AUC | ± std |
|---|---|---|---|
| Dummy | — | 0,5000 | 0 |
| LogReg default | — | 0,8356 | 0,0079 |
| **LogReg GridSearch** | C ∈ {0,001…10} × class_weight ∈ {None, balanced} (10 kombinasi) | **0,8356** | 0,0078 |
| LogReg + interaksi derajat 2 | C ∈ {0,001…0,03} | 0,8334 | 0,0086 |
| HistGB RandomizedSearch | 40 kombinasi dari 5 hyperparameter | 0,8312 | 0,0104 |
| HistGB default | — | 0,8078 | 0,0155 |

## Keputusan
**Dipilih: Logistic Regression (C=10, tanpa class_weight).**
- Skor CV tertinggi. Semua kandidat teratas berada dalam ±1 std satu sama lain, sehingga berdasarkan aturan *one-standard-error* model paling sederhana dipilih.
- Tuning LogReg praktis tidak mengubah skor (C=0,1 sampai 10 semuanya 0,8355–0,8356). Model default sudah cukup baik.
- Tuning **sangat membantu** HistGB (0,8078 → 0,8312): kombinasi terbaik adalah pohon dangkal (depth 3), learning rate kecil (0,059), daun besar (min 133 sampel). Selisih train–val turun menjadi 0,023. Tuning mengurangi overfit, tapi tetap tidak mengalahkan model linear di data ini.
- `class_weight="balanced"` tidak mengubah AUC. Ini wajar: AUC hanya mengukur ranking, sedangkan class weight terutama menggeser peluang (dan threshold).

## Hasil akhir di test (dijalankan sekali)
| ROC-AUC | PR-AUC |
|---|---|
| **0,8365** | 0,6686 |

Skor test ≈ skor CV (0,8356), jadi perkiraan CV dapat dipercaya dan tidak ada tanda kebocoran.

## Reproduksi
```powershell
uv run python weeks/week-12-hyperparameter-tuning/tuning.py
```
Output: `models/week12_churn_model.joblib` dan `models/week12_experiment.json` (seed, skor CV semua kandidat, parameter terpilih, skor test).

## Keterbatasan dan langkah berikutnya
- Data sintetis dengan struktur linear; di data asli, peringkat model bisa berbeda.
- Threshold operasional belum dipilih ulang untuk model ini (gunakan metode biaya Week 8 di data validasi).
- Belum ada analisis error per segmen (Week 15).
