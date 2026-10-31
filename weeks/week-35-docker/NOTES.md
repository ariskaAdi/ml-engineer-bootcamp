# Week 35 — Docker dan Reproducibility

File: [inference-py/Dockerfile](../../services/inference-py/Dockerfile), [api-go/Dockerfile](../../services/api-go/Dockerfile), [docker-compose.yml](../../services/docker-compose.yml), [.env.example](../../services/.env.example).

## Menjalankan seluruh sistem
```powershell
uv run python services/inference-py/scripts/export_model.py           # model package
Copy-Item services/.env.example services/.env                         # lalu ganti password
docker compose -f services/docker-compose.yml --env-file services/.env up --build -d --wait
curl localhost:8080/ready
curl -X POST localhost:8080/v1/predictions -d '{"client_id":"C001", ... }'
docker compose -f services/docker-compose.yml --env-file services/.env logs -f api inference
docker compose -f services/docker-compose.yml --env-file services/.env down          # stop (data Postgres tetap)
```
Sudah diuji: ketiga container *healthy*, request lewat Go → Python → model, prediksi tercatat di PostgreSQL dengan request id yang sama, dan port 8000 **tidak** bisa diakses dari luar.

## Tiga container
| Service | Image | Ukuran | Port ke luar |
|---|---|---|---|
| `api` (Go) | multi-stage → `distroless/static:nonroot` | **16 MB** | 8080 |
| `inference` (Python) | `python:3.12-slim` + sklearn/pandas | 726 MB | — (internal saja) |
| `postgres` | `postgres:16-alpine` | — | 127.0.0.1:5433 (hanya untuk monitoring lokal) |

## Konsep Dockerfile
- **Urutan layer untuk cache**: `COPY requirements.txt` → `pip install` → baru `COPY app`. Mengubah kode tidak memicu install ulang dependency.
- **Multi-stage (Go)**: tahap 1 memakai image `golang` (±300 MB) untuk compile; tahap 2 hanya menyalin binary statis (`CGO_ENABLED=0`) ke image minimal tanpa shell. Lebih kecil = lebih cepat di-deploy dan lebih sedikit celah keamanan.
- **Non-root user**: kalau aplikasi dibobol, penyerang tidak menjadi root di container.
- **`.dockerignore`**: tes, cache, dan file DB lokal tidak ikut ke image.
- **`HEALTHCHECK`** Python memanggil `/ready`; Compose `depends_on: condition: service_healthy` → Go baru start setelah model siap, Python baru start setelah Postgres siap.

## Konfigurasi lewat environment variable
| Variabel | Service | Default |
|---|---|---|
| `MODEL_DIR` | inference | `/srv/model` |
| `DATABASE_URL` | inference | SQLite lokal; di Compose: PostgreSQL |
| `LOG_LEVEL`, `MAX_BATCH` | inference | INFO, 100 |
| `INFERENCE_URL` | api | `http://inference:8000` (nama service = hostname di jaringan Docker) |
| `INFERENCE_TIMEOUT`, `PORT` | api | 2s, 8080 |
Image yang **sama** bisa dipakai di laptop, staging, dan produksi; hanya env var yang berbeda.

## Rahasia (secrets)
- Password Postgres hanya ada di `services/.env` (di-ignore Git). Yang di-commit hanya `.env.example` dengan nilai palsu.
- Tidak ada `ENV PASSWORD=...` di Dockerfile: env di Dockerfile tersimpan permanen di image dan bisa dibaca siapa pun yang punya image (`docker history`).
- Di cloud: pakai fitur secret dari platform (Render environment, GitHub Actions secrets).
- Kalau rahasia terlanjur ter-commit: **ganti rahasianya**. Menghapus dari commit berikutnya tidak cukup karena histori Git tetap menyimpannya.

## Reproducibility
Docker mengunci OS + versi Python + library. Ditambah: model package dengan hash data dan versi model, `requirements.txt` dengan versi persis. Siapa pun dengan Docker bisa menjalankan sistem yang identik.

## Analogi untuk orang awam
- **Container** → kontainer kapal: isinya dikemas lengkap dan standar, sehingga bisa dipindah dari kapal ke truk ke gudang tanpa dibongkar ulang. "Di laptopku jalan kok" tidak lagi menjadi masalah.
- **Image vs container** → image = cetakan kue; container = kue yang sedang dibuat dari cetakan itu. Satu cetakan bisa membuat banyak kue.
- **Layer cache** → menyimpan adonan dasar yang sudah jadi, sehingga kalau hanya hiasannya yang berubah, kamu tidak mengulang dari tepung.
- **Multi-stage build** → membangun mebel di bengkel besar, lalu hanya mengirim mebelnya ke rumah, bukan seluruh bengkel (image Go 16 MB).
- **Docker Compose** → konduktor orkestra yang memastikan pemain masuk sesuai urutan (database dulu, lalu model, lalu API).
- **Secret di `.env`** → kunci rumah disimpan di saku, bukan ditempel di pintu yang difoto dan disebar.

## Exit criteria — tes diri
- [ ] Jalankan dengan perintah di atas dan verifikasi request end-to-end.
- [ ] Jelaskan kenapa image Go 16 MB tapi image Python 726 MB.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
