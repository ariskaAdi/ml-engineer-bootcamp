# Week 28 (Track B) — Transformer dan Hugging Face

## File dan cara menjalankan
`uv run python weeks/week-28-transformers-huggingface/transformers_finetune.py`
Unduh: DistilBERT (±270 MB) + model SST-2 siap pakai (±270 MB). Di CPU laptop ini fine-tune ±1 jam (sambil menjalankan job lain; tanpa gangguan ±20–30 menit). **Di Google Colab (T4 GPU) ±1–2 menit**: sangat disarankan untuk eksperimen di minggu ini.
Output: `models/week28_distilbert_sst2/` dan `models/week28_comparison.json`.

## Attention secara intuitif
Setiap token membuat tiga vektor: **query** ("aku mencari informasi apa"), **key** ("aku berisi informasi apa"), **value** (isinya).
`Attention = softmax(Q·Kᵀ / √d) · V` → setiap token menjadi rata-rata berbobot dari token lain, dengan bobot = seberapa cocok query-nya dengan key token lain.
Di script, bobot dihitung dengan tangan untuk "the movie was not good" (bobot masih acak karena belum dilatih). Setelah dilatih, "good" bisa belajar memberi perhatian besar ke "not".

**Transformer** = tumpukan blok [multi-head attention → feed-forward], ditambah *positional encoding* supaya urutan kata diketahui. Berbeda dengan BoW, seluruh kalimat dibaca sekaligus dan konteks kiri-kanan diperhitungkan. DistilBERT: 6 layer, 66 juta parameter (versi "disuling" dari BERT yang 2× lebih besar).

## Tokenizer
`"This movie isn't unwatchable!"` → `[CLS] this movie isn ' t un ##watch ##able ! [SEP] [PAD]...`
- **Subword (WordPiece)**: kata langka dipecah menjadi potongan yang dikenal → tidak ada kata out-of-vocabulary.
- Token khusus: `[CLS]` (representasinya dipakai untuk klasifikasi), `[SEP]`, `[PAD]`.
- `attention_mask`: 1 = token asli, 0 = padding (diabaikan).
- Tokenizer **harus sama** dengan yang dipakai saat pretraining. Selalu muat dengan `AutoTokenizer.from_pretrained(nama_model_yang_sama)`.

## Fine-tuning
1. Muat `distilbert-base-uncased` + **kepala klasifikasi baru** (`pre_classifier`, `classifier` = MISSING di load report → diinisialisasi acak). Bobot `vocab_*` = UNEXPECTED (kepala language model yang tidak dipakai). Kedua peringatan ini normal.
2. Loop PyTorch biasa (Week 22): `model(input_ids, attention_mask, labels)` langsung mengembalikan `loss`.
3. Learning rate kecil (3e-5) + linear decay + gradient clipping. lr besar akan merusak pengetahuan pretrained (*catastrophic forgetting*).
4. Hanya 4.000 kalimat (≥ 8 kata, supaya mirip test), 2 epoch, max 64 token.

## Hasil (test = 872 kalimat utuh)
| Model | Test acc | Test F1 | Latensi CPU* | Ukuran | Biaya training |
|---|---|---|---|---|---|
| TF-IDF uni+bigram + LogReg (Week 27, 62k data) | 0,817 | 0,828 | **0,17 ms** | ±10 MB | 19 detik CPU |
| **DistilBERT fine-tune sendiri (4k data, 2 epoch)** | 0,878 | 0,883 | 149 ms | 269 MB | ±1 jam CPU / ±1 menit GPU |
| DistilBERT SST-2 dari Hub (dilatih di 67k data) | **0,911** | — | 70 ms | 269 MB | 0 (sudah jadi) |
*Latensi diukur saat CPU juga dipakai job lain; tanpa gangguan kira-kira setengahnya. Di GPU jauh lebih cepat.

Kasus sulit dari Week 27:
| Kalimat | TF-IDF | DistilBERT fine-tune |
|---|---|---|
| this movie is not bad at all | 0,01 ✗ | **0,88 ✓** |
| i expected it to be bad, but it was wonderful | 0,70 | **0,99 ✓** |
| i expected it to be wonderful, but it was bad | 0,69 ✗ | **0,02 ✓** |
| not the worst film ever | 0,00 ✗ | 0,01 ✗ |
| a masterpiece of boredom | 0,85 ✗ | **0,29 ✓** |
Transformer memahami negasi dan kontras "but" dengan benar; negasi jarak jauh "not the worst" masih gagal.

## Apakah peningkatannya sepadan? (exit criteria)
**Keuntungan:** +6 poin akurasi (0,817 → 0,878) dengan hanya 4k data; jauh lebih baik pada negasi/kontras. Model Hub (+9 poin) tanpa biaya training sama sekali.
**Biaya:** latensi ±870× lebih tinggi di CPU, model 27× lebih besar, butuh GPU untuk training yang wajar, deploy lebih mahal (RAM ≥ 1 GB, cold start lebih lama).

Jawabannya bergantung pada konteks:
- **Sepadan** kalau setiap kesalahan mahal (misalnya moderasi konten, klasifikasi keluhan pelanggan), volume per detik rendah–sedang, dan ada GPU/anggaran. Pilih model Hub yang sudah ada sebelum melatih sendiri.
- **Tidak sepadan** kalau volume sangat besar dengan anggaran CPU kecil, latensi < 5 ms wajib, atau 82% sudah memenuhi kebutuhan bisnis. Di capstone (spam SMS), TF-IDF sudah PR-AUC 0,99, jadi transformer tidak dipakai.
- Jalan tengah: distilasi/kuantisasi model, atau kaskade (TF-IDF untuk kasus mudah, transformer hanya untuk kasus yang ragu).

## Exit criteria — tes diri
- [ ] Jelaskan attention dalam 3 kalimat tanpa rumus.
- [ ] Dengan tabel di atas, argumentasikan pilihan model untuk 2 skenario berbeda.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
