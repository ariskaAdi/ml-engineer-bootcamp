# 03 — Integration dan Deployment (Week 41)

## Arsitektur
```
klien ─▶ Go gateway :8081 ─(HTTP/JSON internal, timeout 2s)─▶ Python inference :8000 ─▶ model.joblib
           │ validasi (id, panjang ≤1000 karakter, UTF-8)          │ normalisasi teks (sama dgn training)
           │ aturan: kontak → deliver tanpa model                  │ predict_proba
           │ p ≥ 0,90 quarantine / ≥ threshold warn / deliver      │ log: panjang + SHA-256 + skor
           │ model mati → fail-open (deliver, degraded=true)       ▼
           ▼                                                     PostgreSQL (sms_predictions)
```
| Komponen | Tanggung jawab | Tidak boleh |
|---|---|---|
| Go gateway | Kontrak publik, validasi, aturan bisnis, kebijakan saat model gagal | Menyimpan isi pesan / nomor di log |
| Python inference | Memuat model, normalisasi, skor, log monitoring | Mengambil keputusan bisnis (quarantine/warn) |
| PostgreSQL | Log prediksi untuk monitoring | Menyimpan isi SMS |

## Menjalankan
```powershell
uv run python capstone/ml/src/spam/train.py
uv run python capstone/services/inference/scripts/export_model.py
Copy-Item capstone/services/.env.example capstone/services/.env      # ganti password
docker compose -f capstone/docker-compose.yml --env-file capstone/services/.env up --build -d --wait
bash capstone/examples/requests.sh                                   # 6 contoh request
```
Hasil yang sudah diuji:
| Request | Respons |
|---|---|
| "URGENT! You have WON a £1000 cash prize. Call 0906..." | `quarantine`, p = 0,9998 |
| "Jadi ketemu jam 7 ya? ..." | `deliver`, p = 0,0128 |
| Pengirim di kontak + teks spam | `deliver`, model tidak dipanggil |
| "Your daily text from me - a favour this time" | `warn`, p = 0,666 |
| `message_id` & `text` kosong | 422 + daftar masalah |
| Field `txt` (typo) | 400 |
| Container inference dimatikan | `deliver`, `degraded: true`; `/ready` = not ready |

## Tes dan CI
```powershell
uv run pytest capstone                                   # 21 tes: data, API, privasi, konsistensi normalisasi
cd capstone/services/gateway-go; go test -race ./...     # aturan bisnis, fail-open/closed, integrasi dgn Python palsu
```
Job `capstone` di [.github/workflows/ci.yml](../../.github/workflows/ci.yml). Tes penting:
- `test_raw_text_is_not_stored`: isi pesan ("PIN 4321") tidak ada di database.
- `test_consistency.py`: normalisasi di service == normalisasi di training (mencegah *training-serving skew*).
- `TestPythonTimeoutFailOpenAndFailClosed`: Python lambat → fail-open mengirim pesan; fail-closed → 504.

## Konfigurasi
| Env | Default | Arti |
|---|---|---|
| `INFERENCE_URL` | `http://inference:8000` | Alamat service Python |
| `INFERENCE_TIMEOUT` | 2s | Naikkan ke 60s di free tier (cold start) |
| `FAIL_OPEN` | true | `false` = kembalikan 503/504 saat model gagal |
| `DATABASE_URL` | SQLite lokal | Postgres di Compose |
| `MAX_BATCH` | 50 | Batas pesan per request ke Python |

## Deploy publik (dengan akunmu)
Sama seperti Week 36 (Render Blueprint atau Hugging Face Spaces):
1. `git add capstone/services/inference/model` (1,2 MB; sudah diizinkan `.gitignore`).
2. Buat 2 web service Docker: `capstone/services/inference` dan `capstone/services/gateway-go`.
3. Isi `INFERENCE_URL` gateway dengan URL publik inference; `INFERENCE_TIMEOUT=60s`.
4. Tulis URL demo di [README capstone](../README.md#demo).

## Rollback
Model package diberi versi `logreg_word_char-<waktu>-<hash data>` (terlihat di setiap respons). Simpan image lama dengan tag versi tersebut. Rollback = deploy ulang tag sebelumnya (atau ganti alias champion di MLflow seperti Week 37, lalu export dan build ulang).
