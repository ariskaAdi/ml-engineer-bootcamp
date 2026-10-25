# Week 36 — Cloud Deployment dan Dasar Monitoring

## A. Deploy ke free tier (perlu akunmu sendiri)
Langkah ini **belum dijalankan** karena membutuhkan akun dan push ke GitHub milikmu.

### Opsi 1 — Render (2 service, dari repo GitHub)
1. Commit model package: `git add services/inference-py/model` (sudah diizinkan di `.gitignore`; ukurannya ±20 KB).
2. Salin [services/deploy/render.yaml](../../services/deploy/render.yaml) ke root repo sebagai `render.yaml`, lalu push.
3. Render → **New → Blueprint** → pilih repo.
4. Setelah `bank-inference` hidup, isi env `INFERENCE_URL` di `bank-api` dengan URL publiknya.
5. Tes: `curl https://bank-api-xxxx.onrender.com/ready`.

### Opsi 2 — Hugging Face Spaces (Docker)
Buat Space tipe **Docker**, upload isi `services/inference-py/` (Dockerfile + app + model), lalu ubah `PORT` ke **7860** (port wajib Spaces). Cocok untuk demo service Python saja.

### Batasan free tier
| Batasan | Dampak | Cara menyikapi |
|---|---|---|
| **Sleep** setelah ±15 menit tanpa traffic | Request pertama menunggu *cold start* 30–60 detik | `INFERENCE_TIMEOUT=60s` di Go; jelaskan di README demo |
| RAM 512 MB | Model besar (transformer) tidak muat | Model kecil (LogReg ±8 KB) aman |
| Disk tidak permanen | SQLite di `/tmp` hilang saat restart | Postgres terkelola untuk log yang harus disimpan |
| CPU kecil & dibagi | Latensi lebih tinggi daripada laptop | Ukur ulang p95 setelah deploy |
Kebijakan free tier bisa berubah — cek dokumentasi terbaru sebelum deploy.

## B. Monitoring
Script: [services/monitoring/simulate_and_monitor.py](../../services/monitoring/simulate_and_monitor.py) (stack Docker harus berjalan).
```powershell
uv run python services/monitoring/simulate_and_monitor.py --n 300
```
Mengirim 300 nasabah dari periode **awal** data (mirip training) lalu 300 dari periode **akhir** (kondisi ekonomi berubah), ditambah request rusak. Lalu membaca tabel `predictions` di PostgreSQL. Output: `outputs/week36_monitoring_report.json` + `outputs/week36_monitoring.png`.

### Hasil
| Sinyal | Fase A (normal) | Fase B (drift) |
|---|---|---|
| Latensi API p50 / p95 (klien) | 109 / 168 ms | 73 / 147 ms |
| Latensi model p50 (server) | 77 ms | 46 ms |
| Error rate (422 dari request rusak) | 2,0% | 2,0% |
| Rata-rata peluang | 0,073 | **0,315** |
| % `should_call` | 11,3% | **69%** |
| PSI `euribor3m` | 0,04 | **8,60** |
| PSI `nr_employed` / `emp_var_rate` | 0,00 / 0,01 | **8,63 / 8,80** |
| PSI `month` / `poutcome` / `age` | 0,02 / 0,02 / 0,02 | 1,34 / 0,76 / 0,41 |

(Latensi tinggi untuk LogReg karena laptop sedang menjalankan training lain bersamaan, dan pembuatan DataFrame pandas per request mendominasi. Ukur ulang di server kosong.)

**PSI (Population Stability Index)** membandingkan distribusi fitur di produksi dengan data training: `Σ (cur − ref) · ln(cur / ref)` per bin. Aturan umum: < 0,1 stabil; 0,1–0,2 waspada; > 0,2 drift berarti. Fase A semuanya < 0,05 → monitor tidak memberi alarm palsu. Fase B → alarm keras pada indikator ekonomi.

**Temuan operasional penting:** threshold dirancang supaya ±20% nasabah ditelepon. Saat drift, 69% nasabah melewati threshold, jauh di atas kapasitas call center. Tanpa monitoring distribusi prediksi, masalah ini baru ketahuan saat tim call center kewalahan.

### Sinyal yang dipantau
| Kategori | Metrik | Butuh label? |
|---|---|---|
| Kesehatan sistem | Latensi p50/p95/p99, error rate per status code, readiness | Tidak |
| Prediksi | Rata-rata peluang, % positif, histogram skor | Tidak |
| Data input | PSI/KS per fitur, % nilai kosong, kategori baru | Tidak |
| Kualitas model | PR-AUC, precision di threshold, kalibrasi | **Ya** (baru tersedia setelah hasil kampanye diketahui, bisa berminggu-minggu) |

## Data drift vs concept drift
- **Data drift (covariate shift)**: distribusi input `P(X)` berubah. Contoh: euribor turun dari ±4,9% ke ±1% di periode akhir. Terdeteksi **tanpa label** (PSI).
- **Concept drift**: hubungan input → target `P(y|X)` berubah. Contoh: dengan suku bunga sama, nasabah kini lebih/kurang tertarik deposito karena krisis. Hanya terdeteksi **dengan label** (performa turun).
- Di P1: base rate naik dari 6,4% ke 30,8% dan ROC-AUC turun 0,80 → 0,75 di periode akhir — keduanya terjadi.

## Kapan retraining?
1. **Terjadwal** (misalnya bulanan), kalau data berubah perlahan dan label rutin tersedia.
2. **Dipicu**: PSI fitur penting > 0,2 bertahan beberapa hari, % positif keluar dari rentang yang disepakati, atau metrik berlabel turun di bawah batas.
3. Sebelum mengganti model: bandingkan model baru vs lama pada data terbaru (validasi berbasis waktu), lalu deploy bertahap dan siapkan rollback (Week 37).
Drift tidak selalu berarti retraining. Kadang yang perlu diubah cukup threshold-nya (misalnya kembali ke "top 20%" berdasarkan skor hari ini), atau masalahnya ada di pipeline data (bug, kolom tertukar).

## Exit criteria — tes diri
- [ ] Endpoint publik mengembalikan prediksi (setelah deploy dengan akunmu).
- [ ] Jelaskan 3 sinyal yang menunjukkan model memburuk, mana yang butuh label dan mana yang tidak.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
