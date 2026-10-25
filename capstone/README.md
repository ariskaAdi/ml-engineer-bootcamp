# SMS Spam Filter — Capstone

Sistem end-to-end yang menyaring SMS spam: model teks scikit-learn di service Python, REST API publik di Go, dan log monitoring di PostgreSQL — semuanya dijalankan dengan satu perintah Docker Compose.

## Masalah
Menahan SMS spam sebelum sampai ke pengguna, **tanpa** menahan pesan normal (OTP, pesan keluarga). Karena itu, model dioptimalkan untuk recall spam setinggi mungkin dengan precision ≥ 98%, dan pesan yang meragukan tidak ditahan, melainkan diberi label peringatan.
Detail: [docs/01_problem_and_plan.md](docs/01_problem_and_plan.md).

## Hasil
| | Nilai |
|---|---|
| Data | UCI SMS Spam Collection (CC BY 4.0), 5.160 pesan unik, 12,4% spam |
| Model | TF-IDF kata (1–2 gram) + karakter (2–5 gram) → Logistic Regression |
| CV PR-AUC (5-fold) | 0,984 ± 0,010 |
| **Test PR-AUC** | **0,992** |
| Test precision / recall (threshold 0,43) | 0,969 / 0,969 (4 FP, 4 FN dari 1.032) |
| Latensi model | ±1,7 ms per pesan (CPU) |
| Ukuran model | 1,2 MB |
Laporan lengkap + analisis error: [docs/02_model_report.md](docs/02_model_report.md).

## Arsitektur
```
klien ─▶ Go gateway :8081 ─▶ Python inference ─▶ model
           validasi, aturan bisnis,      skor + log (tanpa isi pesan) ─▶ PostgreSQL
           quarantine / warn / deliver,
           fail-open kalau model mati
```
| Aksi | Kondisi |
|---|---|
| `deliver` | Pengirim ada di kontak (model tidak dipanggil), atau p < 0,43, atau model tidak tersedia (`degraded: true`) |
| `warn` | 0,43 ≤ p < 0,90: tetap dikirim dengan label "mungkin spam" |
| `quarantine` | p ≥ 0,90 |
Detail integrasi dan konfigurasi: [docs/03_deployment.md](docs/03_deployment.md).

## Menjalankan
Prasyarat: [uv](https://docs.astral.sh/uv/), Docker, (opsional) Go 1.24+.
```powershell
uv sync                                                             # dari root repo
uv run python capstone/ml/src/spam/train.py                         # unduh data + latih (±2 menit)
uv run python capstone/services/inference/scripts/export_model.py
Copy-Item capstone/services/.env.example capstone/services/.env     # lalu ganti password
docker compose -f capstone/docker-compose.yml --env-file capstone/services/.env up --build -d --wait
bash capstone/examples/requests.sh
```
Contoh:
```bash
curl -X POST localhost:8081/v1/messages/classify -H "Content-Type: application/json" \
  -d '{"message_id":"m1","sender":"+44870","text":"URGENT! You have WON a £1000 cash prize. Call 09061701461 now"}'
# {"message_id":"m1","action":"quarantine","spam_probability":0.9998,"reason":"sangat mungkin spam",
#  "model_version":"logreg_word_char-20261025T172554-7d039a24","degraded":false}
```

## Tes
```powershell
uv run pytest capstone                                   # 21 tes Python
cd capstone/services/gateway-go; go test -race ./...     # tes Go
```
Berjalan otomatis di GitHub Actions (job `capstone`).

## Demo
URL demo: _(isi setelah deploy — lihat [docs/03_deployment.md](docs/03_deployment.md#deploy-publik-dengan-akunmu))_. Layanan free tier tidur saat tidak dipakai; request pertama bisa butuh ±1 menit.

## Keterbatasan
- Data 2011, berbahasa Inggris. Spam modern (link pendek, e-wallet, bahasa Indonesia) tidak terwakili; kinerja di data baru belum diukur.
- Spam yang meniru percakapan personal sering lolos.
- Beberapa label di dataset dapat diperdebatkan.
- Hanya memakai teks; sistem nyata juga memakai reputasi pengirim dan pola pengiriman.
- Precision test 96,9% sedikit di bawah target 98% (selisih ±1–2 pesan).

## Struktur
```
capstone/
├── ml/                      # data, training, analisis error, tes data
├── services/inference/      # FastAPI + model package + tes API/privasi/konsistensi
├── services/gateway-go/     # Go REST API + tes
├── docker-compose.yml
├── examples/requests.sh
└── docs/                    # 01 rencana · 02 laporan model · 03 deployment · 04 walkthrough & interview
```

## Sitasi data
Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering: New Collection and Results.* DocEng '11.
