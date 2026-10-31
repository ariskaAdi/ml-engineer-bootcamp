# Week 12 — Hyperparameter Tuning

## File dan cara menjalankan
`uv run python weeks/week-12-hyperparameter-tuning/tuning.py` (±1 menit). Laporan eksperimen: [REPORT.md](REPORT.md).

## Konsep
- **Parameter**: dipelajari dari data (bobot LogReg, titik potong pohon).
- **Hyperparameter**: dipilih sebelum training (`C`, `max_depth`, `learning_rate`). Tidak bisa dipelajari dari data train, karena di data train nilai "paling kompleks" selalu tampak terbaik (Week 10).
- **GridSearchCV**: coba semua kombinasi. 10 kombinasi × 5 fold = 50 kali training. Jumlahnya meledak kalau hyperparameternya banyak (5 parameter × 5 nilai = 3125 kombinasi).
- **RandomizedSearchCV**: ambil N kombinasi acak dari distribusi. Dengan anggaran yang sama, biasanya lebih efisien, karena biasanya hanya 1–2 hyperparameter yang benar-benar penting, dan random search mencoba lebih banyak nilai berbeda untuk masing-masing.
- **`loguniform(0.01, 0.3)`**: untuk hyperparameter yang efeknya bekerja dalam skala kali lipat (learning rate, C, alpha). 0,01→0,1 sama "jauhnya" dengan 0,1→1.
- **Penamaan di pipeline**: `model__C` = parameter `C` di langkah bernama `model`.
- **`refit=True`** (default): setelah pencarian, model terbaik dilatih ulang di seluruh data dev. `best_estimator_` siap dipakai.

## Kenapa test set hanya sekali?
Setiap kali hasil test dipakai untuk mengambil keputusan, test berubah menjadi validation set kedua dan skornya ikut terlalu optimis. Dengan 40 kombinasi, beberapa pasti beruntung di data tertentu. CV dipakai untuk memilih; test hanya untuk *melaporkan*.

Hasil: CV 0,8356 → test 0,8365. Hampir sama, berarti proses validasinya jujur.

## Temuan
1. **Tuning tidak selalu membantu.** LogReg: C 0,1–10 semuanya ±0,8356. Model default sudah berada di "dataran" yang datar.
2. **Tuning sangat membantu model yang mudah overfit.** HistGB naik dari 0,808 ke 0,831 dengan membuatnya lebih "pelan" dan sederhana.
3. **Model yang di-tune pun belum tentu menang** dari baseline sederhana. Kandidat dibandingkan secara adil: CV yang sama, metrik yang sama, data dev yang sama.
4. **Aturan one-standard-error**: kalau beberapa model berada dalam ±1 std dari yang terbaik, pilih yang paling sederhana. Lebih cepat, lebih mudah dijelaskan, dan biasanya lebih stabil di data baru.

## Reproducibility checklist
- [x] Seed tetap (`SEED = 42`) untuk split, CV, dan model.
- [x] Semua kandidat + skor CV disimpan (`models/week12_experiment.json`).
- [x] Model akhir disimpan beserta preprocessing (pipeline joblib).
- [x] Versi library tercatat di `uv.lock`.

## Analogi untuk orang awam
- **Parameter vs hyperparameter** → parameter = rasa masakan yang terbentuk saat memasak; hyperparameter = setelan kompor dan lama memasak yang kamu pilih sebelum mulai.
- **Grid search** → mencoba semua kombinasi suhu oven × waktu pada tabel. Teliti, tapi jumlahnya meledak kalau setelannya banyak.
- **Random search** → mencoba 40 kombinasi acak. Biasanya cepat menemukan setelan yang bagus, karena hanya 1–2 setelan yang benar-benar penting.
- **Test hanya sekali** → mencicipi hidangan final di depan juri hanya satu kali. Kalau kamu memasak ulang setelah melihat reaksi juri, penilaiannya tidak lagi adil.
- **Aturan one-standard-error** → dua mobil yang kecepatannya hampir sama: pilih yang lebih irit dan mudah dirawat.

## Exit criteria — tes diri
- [ ] Jalankan ulang script dan dapatkan angka yang sama.
- [ ] Jelaskan alasan memilih model akhir dengan angka dari REPORT.md.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
