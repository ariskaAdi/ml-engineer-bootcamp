# Week 33 — Integrasi Go REST API

Kode: [services/api-go/](../../services/api-go/) — hanya library standar Go (`net/http`, `log/slog`), tanpa framework.

## Cara menjalankan
```powershell
# terminal 1: service Python (Week 32) di port 8000
# terminal 2:
cd services/api-go
$env:INFERENCE_URL="http://localhost:8000"; go run ./cmd/api        # port 8080
curl -X POST localhost:8080/v1/predictions -d '{"client_id":"C001","age":35, ... }'
```
Hasil end-to-end yang sudah diuji:
```json
{"client_id":"C001","probability":0.1213,"action":"call","priority":"medium",
 "reason":"di atas threshold model (20% teratas)","model_version":"logreg-20261025T154559-74adfc57"}
```
`campaign: 12` → `{"action":"do_not_call","reason":"sudah dihubungi 12 kali di kampanye ini (batas 10)"}`, **tanpa memanggil model**.
`age: 5` → **422** `{"error":"validation failed","details":["age: 5 di luar rentang [17, 100]"]}`.

## Arsitektur
```
klien ──HTTP/JSON──▶ Go API :8080 ──HTTP/JSON internal──▶ Python inference :8000 ──▶ model
                     • validasi                            • load model
                     • aturan bisnis                       • predict_proba
                     • keputusan & prioritas               • log prediksi
```
| Lapisan Go | File | Tanggung jawab | Tahu HTTP? |
|---|---|---|---|
| Handler | `internal/httpapi/handler.go` | Decode JSON, panggil service, petakan error → status code, log request | Ya |
| Service (bisnis) | `internal/predict/service.go` | Validasi, aturan *contact fatigue*, prioritas high/medium/low | **Tidak** |
| Client | `internal/inference/client.go` | Panggil Python dengan timeout, teruskan `X-Request-ID` | Ya (sebagai klien) |
| Main | `cmd/api/main.go` | Baca env, rangkai semuanya, timeout server, graceful shutdown | — |

**Kenapa logika bisnis dipisah dari handler?** `Service` hanya bergantung pada interface `Scorer`. Di tes, `Scorer` diganti fake → aturan bisnis bisa dites tanpa HTTP dan tanpa Python. Kalau suatu hari model pindah ke ONNX di dalam Go, cukup buat `Scorer` baru; handler dan aturan bisnis tidak berubah.

**Pembagian tugas Go vs Python:**
- Go: kontrak API publik, validasi, aturan bisnis (batas kontak, prioritas), orkestrasi. Kalau aturan bisnis berubah, model tidak perlu dilatih ulang.
- Python: hanya memuat model dan memberi skor. Ekosistem ML (scikit-learn, PyTorch) ada di Python, jadi tidak perlu menulis ulang model di Go.
- `client_id` hanya dipakai di Go. Model tidak perlu (dan tidak boleh) melihat identitas.

## Detail Go yang penting
- Routing Go 1.22+: `mux.HandleFunc("POST /v1/predictions", ...)`. Method lain otomatis **405**.
- `json.Decoder.DisallowUnknownFields()` + `http.MaxBytesReader` (64 KB): field salah ketik dan body raksasa ditolak.
- `context.WithTimeout` per panggilan ke Python (default 2 detik, `INFERENCE_TIMEOUT`). Tanpa timeout, satu service Python yang macet membuat semua goroutine Go menunggu selamanya.
- Pemetaan error → status: validasi 422, timeout Python 504, Python mati 503, Python error 502.
- `X-Request-ID` diteruskan Go → Python, sehingga satu request bisa dilacak di log kedua service.
- `http.Server` dengan `ReadHeaderTimeout`/`ReadTimeout`/`WriteTimeout` + graceful shutdown (SIGTERM dari Docker → selesaikan request yang sedang berjalan).
- Tanpa message broker: HTTP sinkron cukup untuk satu prediksi per request. Broker (Kafka/RabbitMQ) baru masuk akal untuk batch besar atau pemrosesan asinkron.

## Exit criteria — tes diri
- [ ] Kirim request ke endpoint Go dan dapat prediksi dari model Python.
- [ ] Jelaskan kenapa aturan *contact fatigue* ada di Go, bukan di model.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
