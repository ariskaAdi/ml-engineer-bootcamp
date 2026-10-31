# Week 27 (Track B) — Data Teks dan Baseline NLP Klasik

> **Pilihan track:** Phase 5 meminta memilih satu track. Yang dikerjakan penuh di sini adalah **Track B (NLP)**, karena banyak lowongan ML engineer junior menyebut NLP/LLM/RAG, dan capstone (spam detection) memakainya. Track A (Computer Vision) tersedia sebagai materi tambahan di [week-27-29-track-a-cv/](../week-27-29-track-a-cv/).

## File dan cara menjalankan
`uv run python weeks/week-27-nlp-text-baselines/text_baselines.py` (±1 menit; unduh SST-2 ±5 MB dari Hugging Face).
Data: **SST-2** — kalimat dari review film, label negatif/positif. Label test resmi disembunyikan, jadi split `validation` resmi (872 kalimat) dipakai sebagai test; 5.000 baris dari train dipakai sebagai validation.

## Dari teks mentah ke angka (exit criteria)
1. **Cleaning**: huruf kecil, buang HTML, ganti URL dengan token `URL`, rapikan spasi. Jangan berlebihan: membuang tanda baca `!` atau kata `not` bisa membuang sinyal sentimen.
2. **Tokenisasi**: memecah teks menjadi token. `"This movie is NOT good... !!"` → `['this', 'movie', 'is', 'not', 'good', 'see', '!', '!']`.
3. **Vocabulary**: daftar token unik dari data **train** (13.780 token). Kata yang tidak ada di vocabulary saat inference diabaikan (*out-of-vocabulary*).
4. **Vektorisasi**:
   - **Bag-of-words**: vektor sepanjang vocabulary, isinya jumlah kemunculan setiap kata.
   - **TF-IDF**: hitungan kata × *inverse document frequency*. Kata yang ada di hampir semua kalimat (`the`, IDF 2,2) diberi bobot kecil; kata khas (`masterpiece`, IDF 7,5) diberi bobot besar.
   - Hasilnya adalah matriks *sparse* `(n_kalimat, n_vocabulary)`, hampir semua isinya 0.
5. Model biasa (Logistic Regression) dilatih pada vektor itu.

Semua langkah ada di `make_pipeline(TfidfVectorizer(...), LogisticRegression())`, jadi vocabulary dan IDF hanya dipelajari dari train (tanpa leakage).

## Hasil
| Model | Val acc | **Test acc** | Test F1 | Fit | Inference |
|---|---|---|---|---|---|
| BoW unigram + LogReg | 0,911 | 0,813 | 0,825 | ±9 s | 0,10 ms/kalimat |
| TF-IDF unigram + LogReg | 0,904 | **0,819** | 0,830 | ±7 s | 0,17 ms |
| TF-IDF uni+bigram + LogReg | **0,924** | 0,817 | 0,828 | ±12 s | 0,17 ms |

Baseline: **±82% akurasi test**, sangat cepat (<0,2 ms per kalimat di CPU), dan bobotnya bisa dibaca.

**Kenapa validation 0,92 tapi test 0,82?** Train SST-2 berisi banyak **potongan frasa pendek** (median 7 kata, misalnya `"rushed , slapdash"`), sedangkan test berisi **kalimat utuh** (median 19 kata). Validation yang diambil dari train ikut mudah. Pelajaran: validation harus semirip mungkin dengan data yang akan dihadapi model. Di sini validation tidak memenuhi syarat itu, jadi angka validation terlalu optimis.

## Kenapa urutan kata dan konteks penting
Bag-of-words membuang urutan: `"the film is not good"` dan `"good , not the film"` punya vektor **identik**.
Bigram (`ngram_range=(1, 2)`) mengembalikan sebagian urutan lokal: `"not good"` (bobot −2,44) dan `"not bad"` (+3,71) menjadi fitur sendiri. Tapi masih gagal untuk:
| Kalimat | P(positif) | Masalah |
|---|---|---|
| this movie is not bad at all | 0,01 | `bad` + `not` dihitung terpisah, `not bad at all` tidak tertangkap |
| i expected it to be bad, but it was wonderful | 0,70 | Kata-kata sama persis dengan kalimat di bawah |
| i expected it to be wonderful, but it was bad | 0,69 | ↑ hampir sama → model tidak paham "but" membalik makna |
| not the worst film ever | 0,00 | Negasi jarak jauh |
| a masterpiece of boredom | 0,85 | Butuh memahami kombinasi kata |

Error rate kalimat yang mengandung negasi/kontras (`not, n't, no, never, but, though`): **25,2%** vs **15,5%** tanpa. Transformer (Week 28) membaca seluruh kalimat sekaligus dan memperhitungkan posisi serta hubungan antar kata, sehingga seharusnya lebih baik pada kasus-kasus ini.

## Analogi untuk orang awam
- **Tokenisasi** → memotong kalimat menjadi potongan-potongan kata, seperti memotong roti menjadi irisan sebelum dihitung.
- **Vocabulary** → kamus pribadi model: kata yang tidak ada di kamus diabaikan.
- **Bag-of-words** → memasukkan semua kata ke kantong lalu menghitungnya. Urutan hilang: "anjing menggigit orang" dan "orang menggigit anjing" jadi sama.
- **TF-IDF** → kata yang muncul di semua surat ("yang", "dan") seperti stempel yang ada di setiap dokumen, jadi tidak membantu membedakan. Kata langka seperti "mahakarya" adalah petunjuk yang kuat.
- **Validation tidak mirip test** → berlatih tes membaca dengan potongan kalimat pendek, lalu diuji dengan paragraf panjang. Nilai latihan tampak bagus tapi menipu.

## Exit criteria — tes diri
- [ ] Jelaskan langkah dari kalimat mentah sampai vektor TF-IDF tanpa melihat catatan.
- [ ] Jalankan ulang script dan dapatkan angka yang sama (reproducible: seed + versi library di `uv.lock`).

## Weekly review (isi sendiri)
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:
