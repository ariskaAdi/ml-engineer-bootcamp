# Week 34 — Testing, Reliability, dan CI

## Cara menjalankan semua tes
```powershell
uv run pytest services/inference-py              # 15 tes API Python
cd services/api-go; go test -race ./...          # tes Go (service, client, handler)
uv run pytest projects/p1-bank-marketing projects/p2-fashion-cnn projects/p3-news-topics
```
CI: [.github/workflows/ci.yml](../../.github/workflows/ci.yml) menjalankan semuanya di GitHub Actions setiap push/PR (4 job paralel: go-api, inference-service, ml-projects, capstone).

## Apa yang dites
### Python inference (`services/inference-py/tests/test_api.py`)
| Kasus | Harapan |
|---|---|
| Request valid (2 nasabah) | 200, peluang 0–1, `poutcome=success` > `nonexistent` |
| Prediksi tercatat di DB dengan `X-Request-ID` yang sama | baris ada di tabel `predictions` |
| Umur 5 / 250, job "astronaut", campaign 0, tipe salah, field asing | **422** |
| Field wajib hilang, list kosong, JSON rusak | **422** |
| 101 instance (> MAX_BATCH) | **413** |
| Model gagal dimuat | `/health` 200, `/ready` dan `/v1/predict` **503** |
| Model melempar error | **500** tanpa membocorkan pesan internal ("secret" tidak muncul) |

Tes memakai **model palsu kecil** (`tests/conftest.py`) dengan kontrak yang sama dengan model P1. Tes API tidak butuh artefak P1 dan cepat (±3 detik) → bisa jalan di CI.

### Go
| File | Yang dites |
|---|---|
| `predict/service_test.go` | Validasi mengumpulkan **semua** masalah; prioritas high/medium/low; aturan *contact fatigue* **tidak memanggil model**; input salah tidak memanggil model; error scorer dibungkus |
| `inference/client_test.go` | `httptest.Server` meniru Python: sukses + `X-Request-ID` diteruskan; **timeout** (server tidur 2 detik, klien menyerah di 200 ms); 500; JSON rusak; service mati |
| `httpapi/handler_test.go` | 200 + header request id; pemetaan error → 422/504/503/502; JSON rusak, field asing, body > 64 KB → 400; GET ke endpoint POST → 405 |

Prinsip: **tes Go dan Python secara terpisah** (masing-masing dengan pengganti/fake untuk pihak lain), lalu satu tes end-to-end manual/otomatis untuk memastikan keduanya cocok (Week 35).

## Reliability
- **Timeout** di setiap panggilan jaringan (Go → Python 2 detik; server Go read/write timeout).
- **Kegagalan eksplisit**: setiap jenis kegagalan punya status code sendiri, sehingga klien bisa memutuskan (retry untuk 503/504, perbaiki input untuk 422).
- **Degradasi yang aman**: gagal menulis log prediksi tidak menggagalkan prediksi; model gagal dimuat → readiness 503, bukan crash berulang.
- **Validasi berlapis**: Go (pesan bisnis yang ramah) + Pydantic di Python (pertahanan terakhir, kalau Python dipanggil langsung).

## Log terstruktur tanpa data sensitif
Kedua service menulis **satu baris JSON per request**:
```json
{"level":"INFO","msg":"request","request_id":"a6182c...","method":"POST","path":"/v1/predictions","status":200,"latency_ms":101.7}
```
Yang dicatat: request id, path, status, latensi, versi model, jenis error. Yang **tidak** dicatat: body request (umur, pekerjaan, status pinjaman = data pribadi), `client_id`, stack trace ke klien. Log JSON mudah difilter (`status >= 500`) dan dihitung (latensi p95) oleh tool seperti Grafana Loki, Datadog, atau CloudWatch.

## CI (GitHub Actions)
- Setiap job memasang dependency **minimal** (torch versi CPU, bukan seluruh `pyproject.toml`) → lebih cepat.
- `gofmt` gagal → CI gagal (format konsisten).
- `go test -race`: mendeteksi data race antar goroutine.
- Tes yang butuh model terlatih otomatis `skip` karena artefak tidak di-commit.
- Setelah push: tab **Actions** di GitHub menampilkan hasilnya. Tambahkan badge di README.

## Analogi untuk orang awam
- **Unit test** → menguji setiap komponen mobil secara terpisah (rem, lampu, klakson) sebelum dirakit.
- **Fake / mock** → boneka uji tabrak: menggantikan manusia sungguhan supaya pengujian aman dan bisa diulang.
- **CI** → inspeksi otomatis di setiap pintu keluar pabrik: setiap mobil (setiap push) dicek sebelum boleh keluar.
- **Fail explicitly** → lampu indikator di dashboard yang jelas ("bensin habis", "rem bermasalah"), bukan mobil yang tiba-tiba mati tanpa tanda.
- **Log tanpa data sensitif** → buku tamu yang mencatat jam datang dan keperluan, tapi tidak mencatat isi dompet tamu.

## Exit criteria — tes diri
- [ ] Semua tes hijau secara lokal; push dan pastikan CI hijau.
- [ ] Sengaja rusak satu aturan (misalnya batas umur) dan lihat tes mana yang gagal.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
