# Week 38 — Buffer dan Review Phase 6

## Checklist exit criteria Phase 6
| Week | Exit criteria | Bukti / perintah |
|---|---|---|
| 32 | Request langsung ke Python → prediksi valid | `curl localhost:8000/v1/predict ...` |
| 33 | Request ke Go → prediksi dari model Python | `curl localhost:8080/v1/predictions ...` |
| 34 | Tes sukses + kegagalan penting, jalan di CI | `pytest services/inference-py`, `go test ./...`, tab Actions |
| 35 | Start dengan perintah container + verifikasi end-to-end | `docker compose ... up --build` |
| 36 | Endpoint publik + sinyal degradasi | Render/HF Spaces (akunmu) + laporan monitoring |
| 37 | Telusuri model yang di-deploy + reproduksi konfigurasinya | `metadata.json` → MLflow run |

## Latihan "rusak lalu perbaiki"
Setiap skenario: tebak gejalanya dulu, jalankan, lalu cari penyebabnya **dari log**.
1. Hapus folder `services/inference-py/model/code/` lalu start service → `/ready` 503; log: `model failed to load ... ModuleNotFoundError`.
2. Set `INFERENCE_URL=http://localhost:9999` di Go → `/v1/predictions` **503**.
3. Tambah `time.sleep(5)` di endpoint Python → Go membalas **504** setelah 2 detik.
4. Kirim `"campaign": 0` ke Go → 422 dari validasi Go (Python tidak pernah dipanggil).
5. Ubah versi scikit-learn di `requirements.txt` ke versi lama lalu build image → model gagal dimuat atau muncul peringatan versi.

## Pertanyaan wawancara (MLOps dasar)
<details><summary>Kenapa model tidak dimuat di setiap request?</summary>Memuat file butuh ratusan ms dan memori; dilakukan sekali saat startup, lalu dipakai ulang.</details>
<details><summary>Apa beda liveness dan readiness?</summary>Liveness: proses hidup (kalau gagal → restart). Readiness: siap menerima traffic (kalau gagal → jangan kirim traffic, tapi jangan restart dulu).</details>
<details><summary>Training-serving skew itu apa dan bagaimana mencegahnya?</summary>Preprocessing berbeda antara training dan produksi. Cegah dengan menyimpan seluruh pipeline (bukan hanya model), satu fungsi preprocessing bersama, dan tes yang membandingkan prediksi service dengan prediksi offline.</details>
<details><summary>Drift terdeteksi. Apakah langsung retrain?</summary>Belum tentu. Cek dulu bug pipeline data, lalu dampaknya ke metrik bisnis. Bisa cukup menyesuaikan threshold. Kalau retrain: validasi berbasis waktu, bandingkan dengan champion, deploy bertahap, siapkan rollback.</details>

## Catatan pribadi (isi sendiri)
- Exit criteria yang diulang:
