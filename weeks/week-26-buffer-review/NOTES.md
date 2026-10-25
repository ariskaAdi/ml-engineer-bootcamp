# Week 26 — Buffer dan Review Phase 4

## Checklist exit criteria Phase 4
| Week | Exit criteria | Bukti |
|---|---|---|
| 20 | Diagnosis device mismatch; kenapa workload kecil tidak lebih cepat di GPU | Demo `meta` device; tabel matmul |
| 21 | Hubungan prediksi → loss → gradien → update | Autograd = −26/−13 seperti hitungan tangan Week 5 |
| 22 | Menulis loop training + metrik per epoch tanpa menyalin | `train_one_epoch` / `evaluate` |
| 23 | Diagnosis overfit dengan bukti | Val loss 0,53 → 1,09 sementara train acc 100% |
| 24 | Menjelaskan perbedaan kecepatan yang diukur + keterbatasannya | Isi tabel GPU dari Colab |
| 25 | Re-run training/inference dari instruksi | P2 README |

## Kuis (jawab dulu, baru buka)
<details><summary>Kenapa <code>model.eval()</code> penting saat validasi?</summary>
Dropout dimatikan dan BatchNorm memakai statistik tersimpan. Tanpa eval(), prediksi validasi acak (dropout) dan bergantung pada isi batch (BN).
</details>
<details><summary>Loss training turun tapi val loss naik. Apa yang terjadi dan 3 hal yang dicoba?</summary>
Overfitting. Coba: early stopping, dropout/weight decay, augmentasi atau lebih banyak data, model lebih kecil.
</details>
<details><summary>Loss jadi NaN setelah beberapa langkah. Kemungkinan penyebab?</summary>
Learning rate terlalu besar (divergen), input tidak dinormalisasi, log(0) di loss buatan sendiri. Solusi: turunkan lr, gradient clipping, cek data.
</details>
<details><summary>GPU utilization hanya 20%. Apa yang diperiksa pertama?</summary>
Data loading: num_workers, transform per sampel yang lambat, transfer CPU→GPU. Juga batch terlalu kecil.
</details>
<details><summary>Apa isi state_dict dan kenapa kelas model tetap dibutuhkan saat load?</summary>
Dictionary nama parameter → tensor bobot. Tidak berisi kode arsitektur; arsitektur harus dibangun dulu baru diisi bobotnya.
</details>

## Memilih track Phase 5
| | Track A — Computer Vision | Track B — NLP/LLM |
|---|---|---|
| Cocok untuk lowongan | Manufaktur (inspeksi cacat), retail, kesehatan (citra medis), agrikultur | Chatbot/asisten, pencarian dokumen, analisis ulasan/teks, RAG |
| Tool | torchvision, OpenCV, (YOLO) | Hugging Face transformers, sentence-transformers, LLM API/Ollama |
| Di bootcamp ini | `week-27-29-track-a-cv/` (tambahan) | **Track utama**: week 27–30 + capstone |

Cara memutuskan: buka 20 lowongan "Junior ML Engineer" di kota/negara targetmu, hitung berapa yang menyebut NLP/LLM vs computer vision.

## Catatan pribadi (isi sendiri)
- Track yang dipilih dan alasannya:
