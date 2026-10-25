# Week 32 — Model Inference dan Python Service

Kode: [services/inference-py/](../../services/inference-py/) — FastAPI yang menyajikan model P1 (Bank Marketing).

## Cara menjalankan
```powershell
uv run python projects/p1-bank-marketing/src/bank_marketing/train.py     # kalau belum ada model
uv run python services/inference-py/scripts/export_model.py               # salin model package
cd services/inference-py
uv run uvicorn app.main:app --port 8000
```
Buka **http://localhost:8000/docs**: dokumentasi interaktif otomatis dari schema; bisa langsung mencoba request.
```powershell
curl -X POST localhost:8000/v1/predict -H "Content-Type: application/json" -d '{"instances":[{"age":35,"job":"admin.","marital":"married","education":"university.degree","default":"no","housing":"yes","loan":"no","contact":"cellular","month":"may","day_of_week":"thu","campaign":1,"pdays":999,"previous":0,"poutcome":"nonexistent","emp_var_rate":-1.8,"cons_price_idx":92.893,"cons_conf_idx":-46.2,"euribor3m":1.313,"nr_employed":5099.1}]}'
# -> {"model_version":"logreg-20261025T154559-74adfc57","threshold":0.1169,"predictions":[{"probability":0.1213,"should_call":true}]}
```
Probabilitas 0,1213 **sama persis** dengan `predict.py` P1 → service memakai preprocessing dan model yang identik.

## Pemisahan training vs inference
| Training (`projects/p1-.../train.py`) | Inference (`services/inference-py`) |
|---|---|
| Butuh data lengkap, CV, beberapa menit | Butuh 1 nasabah, harus ±milidetik |
| Dijalankan sesekali | Berjalan terus sebagai server |
| Menghasilkan artefak | Hanya **membaca** artefak |

**Model package** (`services/inference-py/model/`) = semua yang dibutuhkan untuk prediksi:
- `model.joblib`: pipeline lengkap (feature engineering + preprocessing + model).
- `metadata.json`: threshold, versi, hash data, metrik test.
- `code/bank_marketing/`: kode yang dirujuk pickle. Pickle hanya menyimpan **nama** fungsi `add_features`, bukan isinya. Tanpa folder ini, `joblib.load` gagal dengan `ModuleNotFoundError`. Ini jebakan umum saat memindahkan model ke server.
- Versi library di `requirements.txt` **disamakan** dengan saat training (scikit-learn 1.9.1). Pickle dari versi berbeda bisa gagal dimuat atau, lebih buruk, memberi hasil berbeda tanpa error.

## Kontrak API (request/response schema)
`app/schemas.py` (Pydantic):
- Setiap field punya tipe dan batas: `age: int = Field(ge=17, le=100)`, `job: Literal["admin.", ...]`.
- `extra="forbid"`: field tak dikenal ditolak (typo `agee` tidak diam-diam diabaikan).
- Nama field API memakai snake_case (`emp_var_rate`); `model.py` memetakannya ke nama kolom training (`emp.var.rate`). API publik tidak perlu mengikuti nama kolom dataset.
- Input salah → **422** dengan pesan per field, **sebelum** model disentuh.

## Dasar REST
| Konsep | Di service ini |
|---|---|
| Resource + method | `POST /v1/predict` (membuat prediksi), `GET /v1/model-info` |
| Versi di URL | `/v1/...` → kontrak bisa diubah di `/v2` tanpa merusak klien lama |
| Status code | 200 sukses, 413 batch terlalu besar, 422 input salah, 500 inference gagal, 503 model belum siap |
| Stateless | Setiap request berdiri sendiri; model dimuat sekali saat startup (`lifespan`) |
| Health check | `/health` = proses hidup (liveness), `/ready` = model termuat (readiness) |

## Detail desain
- Model dimuat **sekali** saat startup, bukan setiap request (memuat file ratusan ms).
- Kalau model gagal dimuat, service tetap hidup tapi `/ready` = 503 dan log menyebut penyebabnya. Orchestrator (Docker/Render) tidak mengirim traffic ke instance yang belum *ready*.
- Error internal → pesan umum "inference failed"; detailnya hanya di log server (jangan membocorkan stack trace ke klien).
- Setiap prediksi dicatat ke database (`app/db.py`) untuk monitoring (Week 36). Kegagalan mencatat log **tidak** menggagalkan prediksi.

## Exit criteria — tes diri
- [ ] Kirim request langsung ke service Python dan dapat prediksi valid.
- [ ] Jelaskan kenapa folder `code/` ikut di model package dan kenapa versi scikit-learn dikunci.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
