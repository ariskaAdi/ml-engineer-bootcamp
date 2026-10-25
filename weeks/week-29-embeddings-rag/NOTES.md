# Week 29 (Track B) — Embeddings, Semantic Search, dan RAG

## File dan cara menjalankan
`uv run python weeks/week-29-embeddings-rag/embeddings_rag.py`
- Embedding: `intfloat/multilingual-e5-small` (mendukung bahasa Indonesia; unduh ±470 MB sekali).
- Generator: model lokal **Ollama** `qwen2.5:3b` (`ollama pull qwen2.5:3b`; server harus jalan). Kalau Ollama mati, script hanya menampilkan prompt.
- Korpus: semua `weeks/*/NOTES.md` bootcamp ini → kamu bisa bertanya ke catatanmu sendiri.

## Embedding dan cosine similarity
**Embedding** = vektor (di sini 384 angka) yang mewakili *makna* teks. Model dilatih supaya teks bermakna mirip menghasilkan vektor yang arahnya mirip.
`cosine(a, b) = a·b / (|a||b|)`. Kalau vektor dinormalisasi (panjang 1), cosine = dot product biasa → pencarian = satu perkalian matriks `doc_emb @ q`.
- "Model menghafal data latih..." (Indonesia) vs "The model overfits the training set." (Inggris): cosine **0,84**, walaupun tidak ada satu kata pun yang sama. TF-IDF akan memberi 0.
- Di e5, kalimat yang tidak berhubungan pun mendapat cosine ±0,72. **Nilai absolutnya tidak bermakna; yang penting urutannya.** Jangan memakai ambang batas cosine tetap tanpa mengujinya.

## Semantic search
1. **Chunking**: pecah dokumen per heading lalu gabungkan paragraf sampai ±500 karakter (292 chunk).
2. **Index**: embed semua chunk sekali (±1 menit di CPU), simpan matriksnya. (Untuk jutaan chunk: vector database / FAISS.)
3. **Query**: embed pertanyaan, hitung skor ke semua chunk, ambil top-k.

Model e5 dilatih dengan awalan `"query: "` untuk pertanyaan dan `"passage: "` untuk dokumen. Lupa awalan = kualitas turun. Selalu baca *model card* di Hugging Face.

## Evaluasi retrieval (10 pertanyaan dengan jawaban yang diketahui)
| Sistem | Hit@3 | MRR@10 |
|---|---|---|
| paraphrase-multilingual-MiniLM, chunk 1.200 karakter (percobaan pertama) | 0,70 | 0,66 |
| paraphrase-multilingual-MiniLM, chunk 500 karakter | 0,70 | 0,72 |
| Keyword TF-IDF | 0,90 | 0,76 |
| **e5-small, chunk 500 karakter** | **1,00** | **0,95** |
| Hybrid (e5 + TF-IDF, Reciprocal Rank Fusion) | 1,00 | 0,88 |
- **Hit@3** = proporsi pertanyaan yang sumber benarnya muncul di 3 hasil teratas. **MRR** = rata-rata 1/peringkat jawaban benar pertama (1 = selalu di posisi pertama).
- Percobaan pertama gagal karena model MiniLM hanya membaca **128 token** pertama; sisa chunk 1.200 karakter **dipotong diam-diam**. e5-small membaca 512 token. Selalu cek `model.max_seq_length`.
- Embedding yang lemah bisa kalah dari TF-IDF sederhana. **Ukur dulu, jangan berasumsi.**
- *Hybrid search* menggabungkan kekuatan keduanya (kata kunci persis seperti nama fungsi + makna). Di sini sedikit di bawah e5 murni, tapi biasanya lebih tahan di data nyata.

## RAG (Retrieval-Augmented Generation)
```
pertanyaan → [retrieve] top-3 chunk → [augment] masukkan ke prompt → [generate] LLM menjawab dari konteks
```
Prompt meminta: jawab **hanya** dari konteks, sebutkan sumber `[n]`, dan katakan "Tidak ditemukan di catatan" kalau tidak ada.
| Pertanyaan | Hasil |
|---|---|
| Kenapa `duration` dibuang? | Benar, mengutip [2] = catatan Week 17–18 |
| Beda precision dan recall? | Benar tapi umum (model 3B kecil) |
| Harga GPU H100? (tidak ada di catatan) | "Tidak ditemukan di catatan." ✔ |
Percobaan pertama (retrieval buruk) → LLM menjawab "tidak ditemukan" untuk semua pertanyaan. **Kualitas RAG dibatasi oleh kualitas retrieval.**

## Keterbatasan RAG
- **Hallucination**: LLM tetap bisa mengarang atau mencampur dengan pengetahuannya sendiri. Instruksi "jawab hanya dari konteks" mengurangi, tapi tidak menghilangkan. Kutipan sumber memudahkan pengecekan.
- **Kualitas retrieval**: chunk yang salah → jawaban salah/kosong. Chunking, model embedding, dan top-k sangat berpengaruh.
- **Evaluasi sulit**: retrieval bisa diukur (hit@k, MRR). Jawaban perlu set pertanyaan + jawaban acuan, penilaian manusia, atau LLM-as-judge (yang juga bisa salah).
- **Biaya & latensi**: di CPU, qwen2.5:3b butuh 40–75 detik per jawaban. API berbayar: biaya per token, dan konteks panjang = mahal.
- **Privasi**: dokumen dikirim ke LLM. Model lokal (Ollama) menjaga data tetap di komputer.

## Kapan memakai apa? (exit criteria)
| Kebutuhan | Pilihan | Alasan |
|---|---|---|
| Tugas umum, sedikit data, butuh cepat | **Prompt saja** (zero/few-shot ke LLM) | Tanpa training. Coba dulu sebelum yang lain |
| Jawaban harus berdasarkan dokumen yang berubah-ubah atau privat (SOP, catatan, katalog) | **Embeddings + retrieval (RAG)** | Pengetahuan diperbarui dengan menambah dokumen, tanpa melatih ulang; ada kutipan sumber |
| Mencari dokumen mirip / deduplikasi / rekomendasi | **Embeddings saja** | Tanpa LLM; murah dan cepat |
| Tugas tetap dengan label (klasifikasi sentimen/spam/topik), volume besar, latensi rendah | **Fine-tuning** model kecil (atau baseline klasik) | Lebih akurat untuk tugas sempit, jauh lebih murah per prediksi daripada LLM besar |
| Mengubah *gaya/format* output LLM secara konsisten | Fine-tuning LLM (LoRA) | Prompt saja tidak cukup konsisten |
Fine-tuning **tidak** cocok untuk "menambah pengetahuan" yang sering berubah; itu tugas retrieval.

## Exit criteria — tes diri
- [ ] Jelaskan dengan 3 contoh kasus kapan memilih prompt, RAG, atau fine-tuning.
- [ ] Jelaskan kenapa percobaan pertama retrieval gagal dan bagaimana mengetahuinya.

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
