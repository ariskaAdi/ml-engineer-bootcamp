# Week 41 — Capstone: Integrasi dan Deploy

Dokumen: [capstone/docs/03_deployment.md](../../capstone/docs/03_deployment.md).

## Yang sudah dibangun dan diuji
| Item TODO | Di mana | Status |
|---|---|---|
| Package model untuk inference | `capstone/services/inference/model/` (+ `scripts/export_model.py`) | ✅ |
| Python inference ↔ Go API | `services/inference` + `services/gateway-go` | ✅ end-to-end di Docker |
| Tes, CI, contoh request | 21 tes Python + tes Go; job `capstone` di CI; `examples/requests.sh` | ✅ |
| Deploy & alur end-to-end | Docker Compose lokal ✅; deploy publik → **perlu akunmu** (langkah di dokumen) | ⏳ |
| Dokumentasi deploy & konfigurasi | `docs/03_deployment.md` | ✅ |

## Perbedaan desain dibanding Phase 6 (bank marketing)
| | Phase 6 (bank) | Capstone (SMS) | Kenapa berbeda |
|---|---|---|---|
| Model gagal | 503/504 (fail-closed) | **Fail-open**: kirim pesan, `degraded: true` | Memblokir semua SMS lebih merugikan daripada meloloskan spam sebentar |
| Log prediksi | Fitur lengkap (tanpa identitas) | **Hanya panjang + SHA-256 + skor** | Isi SMS sangat pribadi |
| Keputusan | call/skip + prioritas | deliver/warn/quarantine | Zona abu-abu dari analisis error |
| Bypass model | Kontak ≥ 10 kali → do_not_call | Pengirim di kontak → deliver | Aturan bisnis yang lebih tepat daripada model |

## Analogi untuk orang awam
- **Fail-open** → kalau mesin pemindai di bandara rusak, penumpang tetap boleh lewat dengan pemeriksaan manual, daripada menutup bandara seharian. Untuk SMS, lebih baik spam lolos sebentar daripada semua pesan tertahan.
- **Tidak menyimpan isi SMS** → kantor pos yang mencatat berat dan cap amplop, tapi tidak pernah membuka dan menyalin isi suratnya.
- **Kontak bypass** → surat dari keluarga yang sudah dikenal langsung diantar tanpa diperiksa mesin.

## Exit (pribadi)
- [ ] `docker compose ... up` lalu `bash capstone/examples/requests.sh` memberi 6 respons sesuai tabel di dokumen.
- [ ] Matikan container inference dan jelaskan respons gateway.
