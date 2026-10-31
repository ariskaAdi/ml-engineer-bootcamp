# Week 37 — Experiment Tracking dan Model Lifecycle

## File dan cara menjalankan
```powershell
uv run python weeks/week-37-experiment-tracking/mlflow_lifecycle.py        # ±5 menit
uv run mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlflow.db     # http://localhost:5000
```
Tracking store: `outputs/mlflow/mlflow.db` (SQLite) + artefak di `outputs/mlflow/artifacts/` — di luar Git.

## Yang dicatat untuk setiap run
| Jenis | Contoh | Kenapa |
|---|---|---|
| **Params** | model, C, max_depth, seed, cv_folds | Mengulang eksperimen yang sama |
| **Metrics** | cv_pr_auc, cv_pr_auc_std, cv_roc_auc, threshold_top20 | Membandingkan run |
| **Tags** | `data_sha256`, `git_commit`, dataset, project | **Versi data + versi kode** → bisa direproduksi |
| **Artifacts** | model (format skops) + `signature` + `input_example` + `code/` + requirements | Model bisa dimuat ulang di mana saja |

Hasil:
| Run | CV PR-AUC | ± |
|---|---|---|
| logreg_C0.5 | 0,4502 | 0,0136 |
| logreg_C0.05 | 0,4488 | 0,0127 |
| hgb_depth4 | 0,4645 | 0,0167 |
Run bisa dicari lewat kode: `mlflow.search_runs(order_by=["metrics.cv_pr_auc DESC"])`.

## Model Registry dan alias
- `register_model(...)` → **versi** berurutan: v1 = logreg_C0.5, v2 = hgb_depth4.
- **Alias** menunjuk ke versi: `champion` (yang di-deploy) dan `challenger` (kandidat).
- Kode produksi memuat `models:/bank-marketing@champion`, bukan nomor versi. Mengganti model = memindah alias, tanpa mengubah kode.
- Model champion dimuat ulang dari registry → PR-AUC test **0,4536**, sama persis dengan P1. Reproducible.

## Menelusuri model yang sedang di-deploy (exit criteria)
`export_alias("champion")` menulis `services/inference-py/model/metadata.json`:
```json
{"model_name": "logreg_C0.5", "mlflow_run_id": "52c7467e...", "registry_version": 1, "alias": "champion",
 "data_sha256": "74adfc57...", "threshold": 0.1169}
```
Rantai penelusuran: `/v1/model-info` atau `model_version` di setiap respons → `metadata.json` → `mlflow_run_id` → run di MLflow (params, metrik, commit Git, hash data) → `git checkout <commit>` + data dengan hash yang sama = konfigurasi yang persis sama.

## Rollback
Skenario di script:
1. Promosi: `champion` → v2 (hgb).
2. Masalah terdeteksi (misalnya monitoring Week 36).
3. **Rollback**: `set_registered_model_alias("bank-marketing", "champion", 1)` → export ulang → build & deploy ulang image.

Tanpa MLflow, rollback tetap bisa dilakukan selama model package lama disimpan: tag image Docker dengan versi model (`bank-inference:logreg-20261025-74adfc57`) dan deploy ulang tag lama. **Jangan menimpa artefak lama.**

## Format skops vs pickle
MLflow 3 menyimpan model scikit-learn dengan **skops**, yang menolak memuat tipe yang belum dipercaya. Pickle bisa menjalankan kode apa pun saat dimuat, jadi file model dari sumber asing berbahaya.
Di script, dua tipe didaftarkan eksplisit di `skops_trusted_types`: fungsi `add_features` milik sendiri dan `TreePredictor` (HistGradientBoosting). Aman di sini karena kita membuat file modelnya sendiri.

## Versi dataset (opsional: DVC)
Di sini versi data = hash SHA-256 file CSV yang dicatat sebagai tag. **DVC** melangkah lebih jauh: file data besar disimpan di storage (S3/GDrive), dan Git hanya menyimpan file kecil `.dvc` berisi hash-nya. `git checkout` versi lama + `dvc pull` = data versi lama kembali.
```powershell
uv add dvc; dvc init; dvc add projects/p1-bank-marketing/data/bank-additional-full.csv
git add projects/p1-bank-marketing/data/*.dvc .dvc; git commit -m "track data with dvc"
```

## Analogi untuk orang awam
- **Experiment tracking** → buku catatan laboratorium: setiap percobaan dicatat bahannya, takarannya, suhu, dan hasilnya, sehingga bisa diulang persis.
- **Model registry** → gudang dengan rak bernomor versi: v1, v2, v3. Tidak ada yang ditimpa.
- **Alias champion** → papan "menu andalan hari ini" yang menunjuk ke salah satu resep. Mengganti menu andalan cukup mengganti papan, tidak perlu mengubah dapur.
- **Rollback** → tombol *undo*: kalau resep baru dikeluhkan pelanggan, papan dikembalikan ke resep sebelumnya dalam hitungan menit.
- **Hash data** → sidik jari dataset: kalau datanya berubah satu baris pun, sidik jarinya berbeda.
- **skops vs pickle** → paket yang diperiksa satpam sebelum dibuka (skops) vs paket yang langsung dibuka tanpa tahu isinya (pickle).

## Exit criteria — tes diri
- [ ] Dari `model_version` di respons API, temukan run MLflow-nya dan sebutkan parameter, metrik, commit, dan hash data.
- [ ] Lakukan rollback champion ke versi sebelumnya dan export ulang.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
