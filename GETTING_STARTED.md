# Panduan Belajar dari Nol — Mentor Guide

Halo! Anggap file ini sebagai mentor yang menemanimu langkah demi langkah. Kalau kamu ikuti urutannya, kamu tidak akan kehilangan arah.

**Tiga file utama di repo ini:**

| File | Isinya | Kapan dibuka |
|---|---|---|
| [TODO.md](TODO.md) | **Apa** yang dipelajari setiap minggu, plus exit criteria | Awal dan akhir setiap minggu |
| `GETTING_STARTED.md` (file ini) | **Bagaimana** cara belajarnya, dengan kode yang eksplisit | Setiap kali bingung harus ngapain |
| `weeks/week-XX-.../` | Tempat kamu **praktik**: notebook dan catatan | Setiap sesi belajar |

**Arti simbol di panduan ini:**

| Simbol | Arti |
|---|---|
| 🎯 | Tujuan bagian ini |
| ⌨️ | Ketik sendiri (jangan copy-paste, supaya tanganmu ikut belajar) |
| 🤔 | Tebak dulu hasilnya sebelum menjalankan kode |
| ✅ | Checkpoint: pastikan hasilmu sama sebelum lanjut |
| 💡 | Penjelasan kenapa ini penting |
| 🆘 | Kalau muncul error |

> **Aturan emas:** jangan lanjut ke langkah berikutnya sebelum checkpoint ✅ tercapai. Kalau mentok lebih dari 30 menit, tanya saja (lihat [Bagian 6](#bagian-6--kalau-error-atau-bingung)).

---

## Daftar isi

- [Bagian 1 — Cek kemampuan Python](#bagian-1--cek-kemampuan-python)
- [Bagian 2 — Week 0: setup komputer sampai selesai](#bagian-2--week-0-setup-komputer-sampai-selesai)
- [Bagian 3 — Ritme belajar setiap minggu](#bagian-3--ritme-belajar-setiap-minggu)
- [Bagian 4 — Cara mempelajari satu topik](#bagian-4--cara-mempelajari-satu-topik)
- [Bagian 5 — Kapan boleh mencentang exit criteria](#bagian-5--kapan-boleh-mencentang-exit-criteria)
- [Bagian 6 — Kalau error atau bingung](#bagian-6--kalau-error-atau-bingung)
- [Bagian 7 — Week 1 lengkap: NumPy dan Pandas](#bagian-7--week-1-lengkap-numpy-dan-pandas)
- [Bagian 8 — Template](#bagian-8--template)
- [Bagian 9 — Sumber dataset](#bagian-9--sumber-dataset)
- [Bagian 10 — Package per phase](#bagian-10--package-per-phase)
- [Bagian 11 — Weekly review](#bagian-11--weekly-review)
- [Bagian 12 — Lanjut ke Week 2 dan seterusnya](#bagian-12--lanjut-ke-week-2-dan-seterusnya)

---

## Bagian 1 — Cek kemampuan Python

🎯 Menentukan apakah kamu bisa langsung masuk Week 1, atau perlu memperkuat dasar Python dulu.

Jawab dengan jujur. Bisakah kamu melakukan semua ini **tanpa mencari di Google atau bertanya ke AI**?

- [ ] Memakai variabel, `if/elif/else`, serta loop `for` dan `while`.
- [ ] Memakai list, dictionary, tuple, dan set, dan tahu kapan memakai masing-masing.
- [ ] Menulis function dengan parameter, return value, dan default argument.
- [ ] Membaca dan menulis file teks atau CSV.
- [ ] Menangani error dengan `try/except`.
- [ ] Meng-import module dan memasang package.
- [ ] Menulis class sederhana dengan `__init__` dan method.
- [ ] Memakai list comprehension.
- [ ] Menyelesaikan soal di bawah dalam 15 menit.

**Soal tes:** buat function `count_words(words)` yang menerima list kata dan mengembalikan dictionary berisi setiap kata beserta jumlah kemunculannya.

```python
count_words(["apel", "jeruk", "apel"])
# hasil yang diharapkan: {'apel': 2, 'jeruk': 1}
```

<details>
<summary>Lihat solusi (buka hanya setelah mencoba sendiri)</summary>

```python
def count_words(words):
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1   # .get(w, 0): ambil nilai lama, atau 0 kalau belum ada
    return counts
```

</details>

**Hasilnya:**
- **Semua tercentang:** lanjut ke Bagian 2, lalu Week 1.
- **Ada yang belum:** selesaikan Bagian 2 (setup), lalu kerjakan blok **Python Fundamentals** di [TODO.md](TODO.md) selama 2–4 minggu sebelum Week 1. Jangan dilewati. Dasar Python yang lemah adalah alasan paling umum orang mentok di kursus ML.

---

## Bagian 2 — Week 0: setup komputer sampai selesai

🎯 Di akhir bagian ini kamu punya: Git yang terkonfigurasi, project dengan Python 3.12 terisolasi, notebook yang bisa jalan, dan commit pertama di GitHub.

Semua perintah di bawah untuk Windows dan dijalankan di **PowerShell** (terminal di VS Code juga PowerShell). Jalankan satu per satu, dari folder project (`C:\adi\project\ML-BOOTCAMP`).

### 2.1 Pasang tools

Cek dulu apa yang sudah ada:

```powershell
git --version
uv --version
code --version
```

Pasang yang belum ada:

```powershell
winget install Git.Git
winget install astral-sh.uv
winget install Microsoft.VisualStudioCode
```

Setelah memasang, **tutup lalu buka lagi terminal** supaya perintah barunya dikenali.

Di VS Code, buka panel Extensions (Ctrl+Shift+X) dan pasang: **Python**, **Jupyter**, dan (opsional) **GitLens**.

### 2.2 Konfigurasi Git (sekali saja per komputer)

```powershell
git config --global user.name "Nama Kamu"
git config --global user.email "email@kamu.com"
git config --global init.defaultBranch main
```

✅ Cek dengan `git config --global --list`. Nama dan email-mu harus muncul.

### 2.3 Buat environment Python 3.12 dengan uv

```powershell
uv python install 3.12
uv init --python 3.12
uv add numpy pandas matplotlib seaborn jupyterlab ipykernel
```

💡 **Apa yang terjadi?**
- `uv python install 3.12` memasang Python 3.12 khusus untuk project, tanpa mengganggu Python lain di komputermu.
- `uv init` membuat `pyproject.toml` (daftar package project) dan `.python-version` (versi Python project).
- `uv add` memasang package ke folder `.venv/` dan mencatatnya di `pyproject.toml` serta `uv.lock`.

Dengan uv, kamu **tidak perlu** `requirements.txt`. Kalau suatu saat ada yang membutuhkannya, buat dengan:

```powershell
uv export --format requirements-txt > requirements.txt
```

✅ Cek dengan `uv run python --version`. Hasilnya harus `Python 3.12.x`.

> 💡 Selalu jalankan Python lewat `uv run`, misalnya `uv run python script.py`. Kalau kamu mengetik `python` saja, yang terpakai adalah Python global di komputer, bukan Python project. Kamu juga **tidak perlu** mengaktifkan venv dengan `Activate.ps1`.

### 2.4 Rapikan file bawaan

`uv init` membuat beberapa file contoh. Rapikan:

1. **Hapus `main.py`**, karena tidak dipakai.
2. **Edit `pyproject.toml`**, ganti baris `description`:
   ```toml
   description = "43-week self-study roadmap to become a Junior ML Engineer"
   ```
3. **Isi `README.md`** dengan deskripsi singkat:
   ````markdown
   # ML Engineer Bootcamp

   My 43-week self-study journey to become a Junior Machine Learning Engineer.

   - Roadmap: [TODO.md](TODO.md)
   - How I study: [GETTING_STARTED.md](GETTING_STARTED.md)
   - Weekly practice: [weeks/](weeks/)

   ## Setup
   ```bash
   uv sync
   uv run jupyter lab
   ```
   ````

### 2.5 Buat struktur folder

```text
ML-BOOTCAMP/
├── README.md
├── TODO.md
├── GETTING_STARTED.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── data/                     ← dataset (TIDAK di-commit ke Git)
├── scripts/                  ← script pembantu, misalnya pembuat dataset
├── weeks/
│   ├── week-00-setup/
│   │   ├── hello.ipynb
│   │   └── NOTES.md
│   ├── week-01-numpy-pandas/
│   │   ├── NOTES.md
│   │   ├── 01-numpy-basics.ipynb
│   │   ├── 02-pandas-basics.ipynb
│   │   ├── 03-mini-task-sales.ipynb
│   │   └── 04-experiments.ipynb
│   └── ...
└── src/                      ← kode Python yang dipakai ulang (mulai Phase 3)
```

Buat foldernya dengan:

```powershell
New-Item -ItemType Directory -Force data, scripts, src, weeks\week-00-setup, weeks\week-01-numpy-pandas
```

### 2.6 Buat `.gitignore`

Buat file `.gitignore` di root project dengan isi berikut:

```gitignore
.venv/
__pycache__/
.ipynb_checkpoints/
data/
*.csv
*.pt
*.pth
*.pkl
*.joblib
mlruns/
.env
```

💡 File-file ini tidak boleh masuk Git: `.venv/` terlalu besar dan bisa dibuat ulang dengan `uv sync`, dataset bisa besar dan kadang ada aturan lisensinya, file model (`.pt`, `.pkl`) juga besar, dan `.env` berisi rahasia seperti password atau API key.

### 2.7 Notebook vs file Python biasa

Sebelum tes, pahami dulu dua jenis file yang akan kamu pakai:

| | `.py` (Python biasa) | `.ipynb` (Jupyter Notebook) |
|---|---|---|
| Cara jalan | Seluruh file sekaligus: `uv run python file.py` | Per cell: Shift+Enter |
| Output | Muncul di terminal, lalu hilang | Muncul di bawah cell dan ikut tersimpan |
| Variabel | Hilang setelah program selesai | Tetap ada selama kernel menyala |
| Dipakai untuk | Script, function yang dipakai ulang, API, test | Belajar, EDA, eksperimen, grafik |

Di VS Code kamu **tidak perlu** menjalankan `uv run jupyter lab`. VS Code sudah menyalakan kernel sendiri. Jupyter Lab hanya dipakai kalau kamu ingin bekerja lewat browser.

### 2.8 Tes notebook pertama

1. Di VS Code, buat file `weeks/week-00-setup/hello.ipynb`.
2. Klik **Select Kernel** (pojok kanan atas) → **Python Environments** → pilih **`.venv` (Python 3.12)**.
3. ⌨️ Ketik di cell pertama, lalu jalankan dengan **Shift+Enter**:

```python
import sys
import numpy as np
import pandas as pd
import matplotlib

print("Python    :", sys.version.split()[0])
print("NumPy     :", np.__version__)
print("Pandas    :", pd.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Lokasi    :", sys.executable)
```

✅ **Checkpoint:**
- Versi Python dimulai dengan `3.12`.
- `Lokasi` mengandung `ML-BOOTCAMP\.venv\`. Kalau lokasinya di tempat lain, berarti kernel yang terpilih salah.

4. Tambahkan cell kedua untuk tes grafik:

```python
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.title("Notebook berhasil!")
plt.show()
```

✅ Grafik gelombang sinus muncul di bawah cell.

### 2.9 Masalah setup yang sering muncul

| 🆘 Error | Penyebab | Solusi |
|---|---|---|
| `AssertionError: SRE module mismatch` | Environment variable `PYTHONHOME` mengarah ke Python versi lain | Hapus permanen: `[Environment]::SetEnvironmentVariable('PYTHONHOME', $null, 'User')`. Lalu **tutup VS Code sepenuhnya** (File → Exit) dan buka lagi. Kalau masih error, restart Windows. |
| `Failed to start the Kernel` | Biasanya penyebabnya sama dengan di atas, atau kernel yang terpilih salah | Cek `$env:PYTHONHOME` di terminal (harus kosong), lalu pilih ulang kernel `.venv` |
| `ModuleNotFoundError: No module named 'numpy'` | Kernel notebook bukan `.venv` | Select Kernel → pilih `.venv (Python 3.12)` |
| `Access is denied` saat uninstall Python | File `.exe` sedang dipakai proses lain (misalnya VS Code) | Tutup VS Code, jalankan perintahnya dari PowerShell di Start Menu |
| `uv : The term 'uv' is not recognized` | Terminal dibuka sebelum uv terpasang | Tutup lalu buka lagi terminal (atau VS Code) |
| `running scripts is disabled on this system` | Kebijakan PowerShell memblokir `Activate.ps1` | Tidak perlu aktivasi; pakai `uv run ...` |

### 2.10 Catatan dan commit pertama

Buat `weeks/week-00-setup/NOTES.md`:

```markdown
# Week 00 — Setup

## What I set up
- Git, uv, VS Code, Python 3.12 (project-only via uv)
- Packages: numpy, pandas, matplotlib, seaborn, jupyterlab, ipykernel

## Mistakes I made and how I fixed them
- Error: SRE module mismatch → Cause: PYTHONHOME pointed to Python 3.14 → Fix: removed PYTHONHOME, restarted VS Code

## What I learned
- `uv run` runs code with the project's Python, not the global one.
- `.venv/` is not committed; anyone can recreate it with `uv sync`.
```

Lalu commit:

```powershell
git status                 # lihat file apa saja yang berubah
git add .
git status                 # pastikan .venv/ dan data/ TIDAK ikut
git commit -m "week-00: project setup"
git push
```

💡 Biasakan menjalankan `git status` **sebelum** commit. Kalau `.venv/` atau file `.csv` ikut muncul, berarti `.gitignore`-mu belum benar.

### 2.11 Tes terakhir Week 0

Ini tes untuk membuktikan setup-mu bisa dibuat ulang (*reproducible*):

```powershell
Remove-Item -Recurse -Force .venv
uv sync
uv run python -c "import numpy, pandas; print('OK', numpy.__version__, pandas.__version__)"
```

✅ **Week 0 selesai** kalau perintah terakhir mencetak `OK` beserta versinya. Centang tugas-tugas Week 0 di [TODO.md](TODO.md), lalu commit.

---

## Bagian 3 — Ritme belajar setiap minggu

🎯 Supaya setiap minggu punya pola yang sama dan kamu tidak bingung harus mulai dari mana.

Setiap minggu kira-kira **7–10 jam**, dibagi menjadi **5 sesi @ 1,5–2 jam**:

| Sesi | Kegiatan | Hasil |
|---|---|---|
| **1 — Learn** | Baca tugas minggu ini di TODO.md. Tonton 1 video dan baca 1 sumber per topik. Buat catatan singkat. | `NOTES.md` mulai terisi |
| **2 — Reproduce** | Ketik ulang contoh kode sendiri. Ubah satu hal, tebak hasilnya, baru jalankan. | `01-...ipynb`, `02-...ipynb` |
| **3 — Mini-task** | Kerjakan mini-task minggu ini. | `03-mini-task-...ipynb` |
| **4 — Experiment** | Coba variasi: parameter, cara lain, bandingkan hasilnya. Catat. | `04-experiments.ipynb` dan experiment log |
| **5 — Review** | Tes exit criteria (Bagian 5), lengkapi catatan, centang TODO, commit, push. | Commit `week-XX: ...` |

**Contoh jadwal** (sesuaikan dengan kesibukanmu):

| Sen | Sel | Rab | Kam | Jum | Sab | Min |
|---|---|---|---|---|---|---|
| Sesi 1 | Sesi 2 | libur | Sesi 3 | libur | Sesi 4 | Sesi 5 + weekly review |

💡 Lima sesi 2 jam jauh lebih efektif daripada satu sesi 10 jam di akhir pekan. Otak butuh jeda untuk menyimpan ingatan.

---

## Bagian 4 — Cara mempelajari satu topik

🎯 Pola yang dipakai untuk **setiap** konsep baru.

1. **See** — tonton satu video pendek atau baca satu bagian. Berhenti begitu paham idenya.
2. **Type** — ketik contoh terkecil yang berjalan.
3. **Break** — ubah input, ukuran, atau parameter. Tebak hasilnya dulu, baru jalankan. Tebakan yang salah adalah saat kamu paling banyak belajar.
4. **Explain** — tulis 3–5 kalimat di `NOTES.md` dengan bahasamu sendiri, seolah menjelaskan ke teman.
5. **Rebuild** — besok harinya, buka notebook kosong dan ulangi tanpa melihat catatan.

**Contoh nyata untuk topik "broadcasting":**

```python
# 2. Type — contoh terkecil
import numpy as np
a = np.array([1, 2, 3])
print(a + 10)          # [11 12 13] → angka 10 "disebarkan" ke setiap elemen

# 3. Break — tebak dulu, baru jalankan
b = np.array([[1], [2]])   # 🤔 shape-nya apa? Hasil b + a shape-nya apa?
print(b.shape, (b + a).shape)

c = np.array([1, 2])
# 🤔 apakah a + c berhasil? Kenapa?
```

```markdown
<!-- 4. Explain — tulis di NOTES.md -->
Broadcasting: NumPy otomatis "merenggangkan" array yang dimensinya 1 supaya
ukurannya cocok. Shape dibandingkan dari kanan; dua dimensi cocok kalau sama
atau salah satunya 1. (3,) + (2,) gagal karena 3 ≠ 2 dan tidak ada yang 1.
```

Kalau langkah 5 belum bisa, berarti topiknya **belum** dikuasai. Ulangi dari langkah 2.

---

## Bagian 5 — Kapan boleh mencentang exit criteria

Pakai tiga tes ini:

1. **Tes notebook kosong** — buat ulang hasilnya dari notebook kosong. Yang boleh dibuka hanya dokumentasi resmi, tanpa tutorial dan tanpa AI.
2. **Tes jelaskan dengan suara** — jelaskan konsepnya dengan suara selama 2 menit. Merekam diri sendiri sangat membantu.
3. **Tes "kenapa"** — jawab pertanyaan "kenapa memilih ini, bukan yang lain?" Contoh: "kenapa harga kosong diisi median, bukan rata-rata?"

Lulus ketiganya → centang. Gagal salah satu → ulangi, atau pakai buffer week.

---

## Bagian 6 — Kalau error atau bingung

### Urutan langkah saat error

Kerjakan berurutan, dan beri waktu **20–30 menit** untuk mencoba sendiri sebelum lanjut ke langkah berikutnya:

1. **Baca pesan error dari bawah.** Baris terakhir menyebut jenis error-nya; baris dengan nama file milikmu menunjukkan lokasinya.
2. **Cek isi variabel:** `print(type(x))`, `print(x.shape)`, `print(df.dtypes)`, `df.head()`. Sebagian besar bug ML adalah salah tipe data atau salah ukuran.
3. **Perkecil masalahnya:** coba dengan 5 baris data atau array kecil.
4. **Baca dokumentasi resmi** function yang kamu pakai. Di notebook, ketik `pd.read_csv?` lalu jalankan untuk melihat dokumentasinya.
5. **Cari pesan error-nya** di Google atau Stack Overflow, persis seperti yang tertulis.
6. **Tanya ke AI atau mentor** (lihat format di bawah).
7. **Catat di `NOTES.md`**, di bagian "Mistakes I made". Kamu pasti akan bertemu bug yang sama lagi.

### Error yang sering muncul di Week 1–6

| 🆘 Error | Arti | Biasanya karena |
|---|---|---|
| `NameError: name 'np' is not defined` | Nama belum dikenal | Lupa menjalankan cell `import`, atau kernel baru di-restart |
| `FileNotFoundError` | File tidak ditemukan | Path salah. Notebook berjalan dari **folder notebook itu sendiri**, bukan dari root project |
| `KeyError: 'kolom'` | Kolom tidak ada | Salah ketik nama kolom, ada spasi tersembunyi, atau huruf besar/kecil berbeda. Cek `df.columns` |
| `ValueError: operands could not be broadcast together` | Ukuran array tidak cocok | Cek `.shape` kedua array |
| `TypeError: unsupported operand type(s)` | Operasi pada tipe data yang salah | Misalnya menjumlahkan teks dengan angka. Cek `df.dtypes` |
| `ValueError: Unable to parse string "2 pcs"` | Teks tidak bisa diubah jadi angka | Bersihkan teksnya dulu sebelum `pd.to_numeric` |
| Hasil aneh setelah menjalankan cell acak | Urutan cell tidak berurutan | Restart Kernel → Run All |

### Cara bertanya yang baik (ke mentor atau AI)

Salin template ini supaya jawaban yang kamu dapat tepat sasaran:

```text
Week/sesi : Week 1, Sesi 3 (mini-task sales)
Tujuan    : mengubah kolom quantity jadi angka
Kode      : df["quantity"] = pd.to_numeric(df["quantity"])
Error     : (tempel pesan error LENGKAP)
Sudah coba: cek df["quantity"].unique(), ternyata ada nilai "2 pcs"
Pertanyaan: bagaimana cara membersihkan teks " pcs"? Jelaskan dulu konsepnya, jangan langsung kodenya.
```

**Cara memakai AI yang benar:**
- ✅ "Jelaskan arti error ini dan apa yang harus aku cek. Jangan berikan kode perbaikannya."
- ✅ "Ini penjelasanku tentang broadcasting. Apa yang salah atau kurang?"
- ✅ "Review notebook-ku, tunjukkan bagian yang kurang tepat tanpa menulis ulang."
- ❌ "Kerjakan mini-task ini."

---

## Bagian 7 — Week 1 lengkap: NumPy dan Pandas

🎯 Di akhir Week 1 kamu bisa: memakai array NumPy, membaca dan memanipulasi data dengan Pandas, membersihkan dataset kotor, dan menjelaskan setiap langkah pembersihannya.

### 7.0 Persiapan

Pastikan folder `weeks/week-01-numpy-pandas/` berisi empat notebook dan satu file catatan:

```text
weeks/week-01-numpy-pandas/
├── NOTES.md
├── 01-numpy-basics.ipynb
├── 02-pandas-basics.ipynb
├── 03-mini-task-sales.ipynb
└── 04-experiments.ipynb
```

> Kalau kamu sudah punya `02-mini-task-sales.ipynb` yang masih kosong, ganti namanya menjadi `03-mini-task-sales.ipynb`.

Isi `NOTES.md` dengan kerangka dari [Bagian 8](#bagian-8--template).

---

### Sesi 1 — Learn (±2 jam)

**Baca/tonton:**
1. [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html), sampai bagian "Broadcasting".
2. [Pandas Getting Started tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/), tutorial 1–4.
3. Satu video Pandas dari [Corey Schafer](https://www.youtube.com/@coreyms) atau [Keith Galli](https://www.youtube.com/@KeithGalli).

**Tulis di `NOTES.md`** jawaban dengan bahasamu sendiri:
- Apa beda list Python dengan array NumPy?
- Apa itu `shape` dan `dtype`?
- Apa beda `Series` dengan `DataFrame`?
- Apa beda `.loc` dengan `.iloc`?

Belum bisa menjawab semuanya juga tidak apa-apa. Pertanyaan ini akan terjawab di Sesi 2.

---

### Sesi 2a — NumPy (±1 jam), file `01-numpy-basics.ipynb`

Pilih kernel `.venv`. ⌨️ Ketik setiap cell, dan jawab 🤔 **sebelum** menjalankannya.

**Cell 1 — Import dan array pertama**

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a)
print("shape:", a.shape)   # 🤔 tebak dulu
print("dtype:", a.dtype)
print("ndim :", a.ndim)
```

```text
Output yang diharapkan:
[1 2 3 4 5]
shape: (5,)
dtype: int64
ndim : 1
```

💡 `shape` adalah ukuran array. `(5,)` artinya 1 dimensi berisi 5 elemen. `dtype` adalah tipe data setiap elemen; semua elemen dalam satu array bertipe sama.

**Cell 2 — Array 2 dimensi (matriks)**

```python
m = np.array([[1, 2, 3],
              [4, 5, 6]])
print(m.shape)    # 🤔 (baris, kolom) = ?
print(m[0])       # baris ke-0
print(m[:, 1])    # semua baris, kolom ke-1
print(m[1, 2])    # baris 1, kolom 2
```

```text
Output yang diharapkan:
(2, 3)
[1 2 3]
[2 5]
6
```

💡 Indeks dimulai dari **0**. Tanda `:` artinya "semua".

**Cell 3 — Cara membuat array dengan cepat**

```python
print(np.zeros((2, 3)))
print(np.ones(3))
print(np.arange(0, 10, 2))     # mulai 0, sampai sebelum 10, lompat 2
print(np.linspace(0, 1, 5))    # 5 angka dengan jarak sama dari 0 sampai 1
```

**Cell 4 — Slicing dan boolean mask**

```python
print(a[1:4])      # 🤔 elemen indeks 1, 2, 3
print(a[::-1])     # dibalik
print(a > 2)       # 🤔 hasilnya apa?
print(a[a > 2])    # ambil elemen yang lebih dari 2
```

```text
Output yang diharapkan:
[2 3 4]
[5 4 3 2 1]
[False False  True  True  True]
[3 4 5]
```

💡 `a[a > 2]` disebut **boolean mask**. Konsep yang sama nanti dipakai di Pandas untuk memfilter data, misalnya `df[df["harga"] > 20000]`.

**Cell 5 — Operasi vektor (tanpa loop)**

```python
print(a * 2)
print(a + a)
print(a ** 2)
```

💡 Di NumPy, operasi langsung berlaku ke **semua elemen** tanpa `for`. Ini disebut *vectorization*, dan jauh lebih cepat. Kamu akan membuktikannya di Sesi 4.

**Cell 6 — Agregasi dan `axis`**

```python
print(m.sum())          # semua elemen
print(m.sum(axis=0))    # 🤔 per kolom → hasilnya berapa elemen?
print(m.sum(axis=1))    # per baris
print(m.mean(axis=0))
```

```text
Output yang diharapkan:
21
[5 7 9]
[ 6 15]
[2.5 3.5 4.5]
```

💡 `axis=0` artinya "menghitung **ke bawah**" (hasilnya satu angka per kolom). `axis=1` artinya "menghitung **ke samping**" (satu angka per baris).

**Cell 7 — Broadcasting**

```python
col = np.array([[10], [20], [30]])   # shape (3, 1)
row = np.array([1, 2, 3, 4])         # shape (4,)
hasil = col + row
print(hasil.shape)   # 🤔 tebak dulu!
print(hasil)
```

```text
Output yang diharapkan:
(3, 4)
[[11 12 13 14]
 [21 22 23 24]
 [31 32 33 34]]
```

```python
# 🤔 Apakah ini berhasil? Tebak dulu, baru jalankan.
np.array([1, 2, 3]) + np.array([1, 2, 3, 4])
```

✅ Seharusnya muncul `ValueError: operands could not be broadcast together with shapes (3,) (4,)`. Tulis di `NOTES.md` kenapa yang ini gagal, sedangkan Cell 7 berhasil.

**Cell 8 — Reshape dan transpose**

```python
print(m.reshape(3, 2))   # ubah bentuk, jumlah elemen tetap 6
print(m.T.shape)         # transpose: baris ↔ kolom
```

**Cell 9 — Contoh kasus nyata**

```python
prices = np.array([45000, 25000, 18000, 12000, 85000])
qty = np.array([3, 5, 2, 10, 1])

revenue = prices * qty
print("Revenue per produk:", revenue)
print("Total revenue     :", revenue.sum())
print("Produk > 20rb     :", prices[prices > 20000])
print("Indeks terlaris   :", revenue.argmax())
```

```text
Output yang diharapkan:
Revenue per produk: [135000 125000  36000 120000  85000]
Total revenue     : 501000
Produk > 20rb     : [45000 25000 85000]
Indeks terlaris   : 0
```

**Latihan NumPy** (kerjakan sendiri di cell baru):

```python
x = np.arange(1, 13).reshape(3, 4)
# 1. Tampilkan x dan shape-nya
# 2. Ambil kolom terakhir
# 3. Ambil semua angka genap
# 4. Hitung jumlah per baris
# 5. Kurangi setiap kolom dengan rata-rata kolom itu (pakai broadcasting)
```

<details>
<summary>Lihat solusi</summary>

```python
x = np.arange(1, 13).reshape(3, 4)
print(x, x.shape)            # (3, 4)
print(x[:, -1])              # [ 4  8 12]
print(x[x % 2 == 0])         # [ 2  4  6  8 10 12]
print(x.sum(axis=1))         # [10 26 42]
print(x - x.mean(axis=0))    # setiap kolom: [-4, 0, 4]
```

</details>

---

### Sesi 2b — Pandas dasar (±1 jam), file `02-pandas-basics.ipynb`

**Cell 1 — Membuat DataFrame**

```python
import pandas as pd

df = pd.DataFrame({
    "nama":    ["Andi", "Budi", "Citra", "Dewi", "Eko"],
    "kota":    ["Jakarta", "Bandung", "Jakarta", "Surabaya", "Bandung"],
    "umur":    [25, 32, 28, None, 41],
    "belanja": [150000, 320000, 90000, 210000, 500000],
})
df
```

💡 DataFrame adalah tabel; setiap kolomnya adalah `Series`. Di notebook, menulis `df` di baris terakhir cell akan menampilkannya sebagai tabel yang rapi.

**Cell 2 — Kenali datanya (selalu lakukan ini pertama kali!)**

```python
print(df.shape)       # (jumlah baris, jumlah kolom)
print(df.dtypes)      # tipe data setiap kolom
df.info()             # ringkasan: jumlah data non-kosong dan tipe
df.describe()         # statistik kolom angka
```

🤔 Kolom `umur` ber-dtype `float64`, bukan `int64`. Kenapa? (Petunjuk: lihat nilai `None` milik Dewi.)

**Cell 3 — Memilih kolom dan baris**

```python
df["kota"]                       # satu kolom → Series
df[["nama", "belanja"]]          # beberapa kolom → DataFrame (kurung siku dua)
df.loc[0]                        # baris dengan LABEL indeks 0
df.iloc[1:3]                     # baris berdasarkan POSISI 1 dan 2
df.loc[df["kota"] == "Jakarta", ["nama", "belanja"]]
```

> Jalankan baris-baris di atas satu per satu (satu baris per cell), karena notebook hanya menampilkan hasil baris terakhir.

💡 `.loc` memakai **label** (nama indeks atau kolom); `.iloc` memakai **posisi** (angka urutan). Saat ini keduanya terlihat sama karena indeksnya 0, 1, 2, ...; bedanya akan terlihat setelah data difilter.

**Cell 4 — Filter dengan beberapa kondisi**

```python
df[(df["belanja"] > 100000) & (df["kota"] == "Bandung")]
```

✅ Hasilnya: Budi dan Eko.

💡 Pakai `&` (dan) atau `|` (atau), bukan `and`/`or`, dan **setiap kondisi wajib diberi kurung**.

**Cell 5 — Kolom baru dan nilai kosong**

```python
df["belanja_ribu"] = df["belanja"] / 1000
print(df.isna().sum())                          # jumlah nilai kosong per kolom
df["umur"] = df["umur"].fillna(df["umur"].median())
df
```

✅ Umur Dewi sekarang `30.0` (median dari 25, 28, 32, 41).

**Cell 6 — Groupby, sort, dan value_counts**

```python
print(df.groupby("kota")["belanja"].agg(["sum", "mean", "count"]))
print(df.sort_values("belanja", ascending=False).head(3))
print(df["kota"].value_counts())
```

💡 `groupby` = "kelompokkan per kota, lalu hitung sesuatu untuk tiap kelompok". Ini salah satu operasi yang paling sering kamu pakai.

**Cell 7 — Merge (menggabungkan dua tabel)**

```python
kota_info = pd.DataFrame({
    "kota": ["Jakarta", "Bandung", "Surabaya"],
    "provinsi": ["DKI Jakarta", "Jawa Barat", "Jawa Timur"],
})
df.merge(kota_info, on="kota", how="left")
```

💡 Ini sama dengan `LEFT JOIN` di SQL, yang akan kamu pelajari di Week 2.

**Latihan Pandas** (pakai `df` di atas):
1. Tampilkan orang yang umurnya di atas 27 dan belanjanya di bawah 300.000.
2. Hitung rata-rata belanja per provinsi (merge dulu).
3. Tambahkan kolom `kategori`: `"besar"` kalau belanja ≥ 200.000, selain itu `"kecil"`.

<details>
<summary>Lihat solusi</summary>

```python
# 1
df[(df["umur"] > 27) & (df["belanja"] < 300000)]

# 2
df.merge(kota_info, on="kota", how="left").groupby("provinsi")["belanja"].mean()

# 3
import numpy as np
df["kategori"] = np.where(df["belanja"] >= 200000, "besar", "kecil")
```

</details>

---

### Sesi 3 — Mini-task: membersihkan data penjualan (±2 jam)

🎯 Membersihkan dataset yang "kotor", seperti data asli di dunia kerja, lalu menjawab pertanyaan bisnis.

#### Langkah 1 — Buat dataset latihan

Kita membuat dataset sendiri yang sengaja dikotori. Untuk latihan ini, copy-paste boleh, karena script ini hanya pembuat data, bukan materi utama.

Buat file `scripts/make_sales_data.py`:

```python
"""Membuat dataset penjualan latihan yang sengaja dibuat 'kotor' untuk Week 1."""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # seed: hasil acak selalu sama setiap dijalankan
n = 1000

products = {
    "Kopi Arabika": ("Minuman", 45000),
    "Teh Hijau": ("Minuman", 25000),
    "Roti Gandum": ("Makanan", 18000),
    "Keripik Singkong": ("Makanan", 12000),
    "Madu Hutan": ("Lainnya", 85000),
}

product = rng.choice(list(products), size=n)
dates = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 181, size=n), unit="D")

df = pd.DataFrame({
    "order_id": np.arange(1, n + 1),
    "order_date": dates.strftime("%Y-%m-%d"),
    "product": product,
    "category": [products[p][0] for p in product],
    "quantity": rng.integers(1, 6, size=n),
    "unit_price": [float(products[p][1]) for p in product],
    "city": rng.choice(["Jakarta", "Bandung", "Surabaya", "Yogyakarta"], size=n),
})

# --- Sengaja dikotori, seperti data asli di dunia kerja ---
def pick(k):
    return rng.choice(n, size=k, replace=False)

df.loc[pick(40), "unit_price"] = np.nan                       # harga hilang
idx = pick(30); df.loc[idx, "city"] = df.loc[idx, "city"].str.upper()        # huruf besar semua
idx = pick(30); df.loc[idx, "city"] = "  " + df.loc[idx, "city"] + " "      # spasi berlebih
idx = pick(15); df.loc[idx, "quantity"] = -df.loc[idx, "quantity"]          # angka negatif
df["quantity"] = df["quantity"].astype(str)
idx = pick(10); df.loc[idx, "quantity"] = df.loc[idx, "quantity"] + " pcs"  # angka bercampur teks
df.loc[pick(10), "order_date"] = None                          # tanggal hilang
df = pd.concat([df, df.sample(20, random_state=1)], ignore_index=True)  # baris duplikat

data_dir = Path(__file__).resolve().parents[1] / "data"
data_dir.mkdir(exist_ok=True)
df.to_csv(data_dir / "sales_raw.csv", index=False)
print(f"Tersimpan: {data_dir / 'sales_raw.csv'} ({len(df)} baris)")
```

Jalankan dari root project:

```powershell
uv run python scripts/make_sales_data.py
```

✅ Output: `Tersimpan: ...\data\sales_raw.csv (1020 baris)`

> Setelah Week 1, ulangi mini-task ini dengan dataset asli, misalnya [Online Retail dari UCI](https://archive.ics.uci.edu/dataset/352/online+retail) atau dataset "sales" di Kaggle.

#### Langkah 2 — Buka `03-mini-task-sales.ipynb` dan tulis pertanyaan dulu

**Cell 1 (Markdown).** Ubah tipe cell menjadi Markdown (tekan `M` saat cell terpilih, atau lewat dropdown tipe cell), lalu tulis:

```markdown
# Mini-task Week 1 — Sales Data Cleaning

## Questions
1. Berapa total revenue per bulan?
2. Produk apa yang paling banyak menghasilkan revenue?
3. Kota mana yang revenue-nya paling besar?
4. Berapa persen kontribusi revenue setiap kategori?
```

💡 Selalu tulis pertanyaan **sebelum** menyentuh data. Tanpa pertanyaan, analisis jadi tidak terarah.

#### Langkah 3 — Import dan lokasi data

**Cell 2**

```python
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cari root project (folder yang punya pyproject.toml), supaya path data selalu benar
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
DATA_DIR = ROOT / "data"
print(DATA_DIR)
```

💡 Notebook berjalan dari folder tempat notebook itu berada (`weeks/week-01-numpy-pandas/`), bukan dari root project. Kalau kamu menulis `pd.read_csv("data/sales_raw.csv")`, akan muncul `FileNotFoundError`. Kode di atas mencari root project secara otomatis.

#### Langkah 4 — Kenali datanya

**Cell 3**

```python
df = pd.read_csv(DATA_DIR / "sales_raw.csv")
print(df.shape)
df.head()
```

**Cell 4**

```python
df.info()
```

🤔 Perhatikan kolom `quantity`. Tipenya `str` (teks), padahal seharusnya angka. Kenapa?

**Cell 5 — Daftar masalah**

```python
print("Nilai kosong per kolom:")
print(df.isna().sum())
print("\nBaris duplikat:", df.duplicated().sum())
print("\nIsi unik city:", df["city"].unique())
print("\nIsi unik quantity:", df["quantity"].unique())
```

✅ **Checkpoint:** kamu seharusnya menemukan **6 masalah**:

| # | Masalah | Bukti |
|---|---|---|
| 1 | 20 baris duplikat | `df.duplicated().sum()` = 20 |
| 2 | Nama kota tidak konsisten | `"JAKARTA"`, `"  Jakarta "`, `"Jakarta"` |
| 3 | `quantity` bertipe teks | Ada nilai seperti `"2 pcs"` |
| 4 | `quantity` negatif | Ada nilai seperti `"-3"` |
| 5 | `order_date` kosong dan bertipe teks | 10 nilai kosong |
| 6 | `unit_price` kosong | 41 nilai kosong |

Salin tabel ini (dengan kata-katamu sendiri) ke cell Markdown di notebook.

#### Langkah 5 — Bersihkan, satu masalah per cell

Untuk setiap langkah, tulis **satu kalimat alasan** di cell Markdown sebelum kodenya. Ini syarat exit criteria Week 1: *"Clean a dataset and explain every transformation."*

**Cell 6 — Hapus duplikat**

```python
n_awal = len(df)
df = df.drop_duplicates()
print(f"{n_awal} → {len(df)} baris")
```

✅ `1020 → 1000 baris`

**Cell 7 — Rapikan nama kota**

```python
df["city"] = df["city"].str.strip().str.title()
df["city"].unique()
```

✅ Tinggal 4 kota: Yogyakarta, Surabaya, Bandung, Jakarta.

💡 `.str.strip()` membuang spasi di awal dan akhir teks; `.str.title()` mengubah `"JAKARTA"` menjadi `"Jakarta"`.

**Cell 8 — Ubah quantity jadi angka**

```python
df["quantity"] = df["quantity"].str.replace(" pcs", "", regex=False)
df["quantity"] = pd.to_numeric(df["quantity"])
df["quantity"].dtype
```

✅ `dtype('int64')`

🆘 Kalau kamu langsung menjalankan `pd.to_numeric` tanpa `replace`, akan muncul `ValueError: Unable to parse string "4 pcs"`. Error itu memberi tahu nilai mana yang bermasalah. Coba sengaja, supaya kamu kenal error-nya!

**Cell 9 — Buang quantity yang tidak valid**

```python
print("Quantity <= 0:", (df["quantity"] <= 0).sum())
df = df[df["quantity"] > 0]
print(len(df), "baris")
```

✅ `Quantity <= 0: 15`, lalu `985 baris`

🤔 Kenapa dibuang, bukan diubah jadi positif? (Tidak ada jawaban mutlak. Bisa jadi itu retur barang, bisa juga salah input. Di dunia kerja, tanyakan ke pemilik data. Tulis asumsimu di notebook.)

**Cell 10 — Ubah tanggal jadi tipe datetime**

```python
df["order_date"] = pd.to_datetime(df["order_date"])
print("Tanggal kosong:", df["order_date"].isna().sum())
df = df.dropna(subset=["order_date"])
print(len(df), "baris")
```

✅ `Tanggal kosong: 10`, lalu `975 baris`

💡 Tanggal dalam bentuk teks tidak bisa dikelompokkan per bulan. Setelah menjadi `datetime`, kamu bisa memakai `.dt.month`, `.dt.to_period("M")`, dan lain-lain.

**Cell 11 — Isi harga yang kosong**

```python
df_before_fill = df.copy()   # simpan untuk eksperimen di Sesi 4

median_price = df.groupby("product")["unit_price"].transform("median")
df["unit_price"] = df["unit_price"].fillna(median_price)
df.isna().sum()
```

✅ Semua kolom bernilai `0` (tidak ada lagi yang kosong).

💡 Harga diisi dengan **median harga produk yang sama**, bukan median semua produk. Harga Madu Hutan yang kosong seharusnya 85.000, bukan median gabungan yang hanya 25.000.

**Cell 12 — Hitung revenue dan cek hasil akhir**

```python
df["revenue"] = df["quantity"] * df["unit_price"]
df.info()
```

✅ **Checkpoint akhir pembersihan:** 975 baris tanpa nilai kosong; `quantity` bertipe `int64`, `order_date` bertipe `datetime64`, dan `revenue` bertipe `float64`.

#### Langkah 6 — Jawab pertanyaan bisnis

**Cell 13 — Revenue per bulan**

```python
monthly = df.groupby(df["order_date"].dt.to_period("M"))["revenue"].sum()
print(monthly)

ax = monthly.plot(kind="bar", title="Revenue per Bulan")
ax.set_ylabel("Revenue (Rp)")
plt.tight_layout()
plt.show()
```

✅ Bulan dengan revenue tertinggi: **2026-06** (Rp 21.709.000).

**Cell 14 — Produk terlaris**

```python
top_products = df.groupby("product")["revenue"].sum().sort_values(ascending=False)
top_products.head(5)
```

✅ Peringkat 1: **Madu Hutan** (Rp 48.875.000).

**Cell 15 — Revenue per kota**

```python
df.groupby("city")["revenue"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)
```

✅ Peringkat 1: **Jakarta** (Rp 29.848.000 dari 258 order).

**Cell 16 — Kontribusi per kategori**

```python
(df.groupby("category")["revenue"].sum() / df["revenue"].sum() * 100).round(1)
```

✅ Lainnya ≈ 45,1%, Minuman ≈ 37,3%, Makanan ≈ 17,6%.

**Cell 17 (Markdown) — Kesimpulan.** Tulis 3–5 kalimat dengan kata-katamu sendiri. Contoh awal:

```markdown
## Findings
- Madu Hutan hanya 1 dari 5 produk, tapi menyumbang ±45% revenue karena harganya paling mahal.
- ...

## Assumptions and limitations
- Quantity negatif saya anggap salah input dan saya buang (15 baris).
- Data ini sintetis, jadi pola bulanannya acak dan tidak mencerminkan musim yang nyata.
```

**Cell 18 — Simpan data bersih**

```python
df.to_csv(DATA_DIR / "sales_clean.csv", index=False)
```

> Angka-angka ✅ di atas berasal dari seed `42`. Kalau angkamu berbeda, cek lagi langkah sebelumnya. Kemungkinan ada langkah yang terlewat, atau urutannya berbeda.

---

### Sesi 4 — Eksperimen (±1,5 jam), file `04-experiments.ipynb`

🎯 Membuktikan sendiri dua hal: (1) cara menangani data kosong mempengaruhi hasil, (2) operasi vektor lebih cepat daripada loop.

**Cell 1 — Siapkan data**

Ulangi langkah pembersihan Cell 2–10 dari Sesi 3 sampai sebelum harga diisi, lalu simpan sebagai `df_before_fill`. Atau, lebih rapi: tambahkan baris ini di Sesi 3 tepat setelah Cell 10, lalu jalankan ulang notebook itu:

```python
df.to_csv(DATA_DIR / "sales_before_fill.csv", index=False)
```

Lalu di `04-experiments.ipynb`:

```python
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
DATA_DIR = ROOT / "data"

df_before_fill = pd.read_csv(DATA_DIR / "sales_before_fill.csv", parse_dates=["order_date"])
df_before_fill.isna().sum()
```

**Eksperimen 1 — `dropna()` vs mengisi nilai kosong**

🤔 **Hipotesis:** "Kalau baris berharga kosong dibuang, total revenue akan lebih kecil daripada kalau harganya diisi." Kira-kira berapa persen selisihnya?

```python
# Cara A: buang baris yang harganya kosong
dropped = df_before_fill.dropna(subset=["unit_price"])
total_dropped = (dropped["quantity"] * dropped["unit_price"]).sum()

# Cara B: isi dengan median harga produk yang sama
filled = df_before_fill.copy()
filled["unit_price"] = filled["unit_price"].fillna(
    filled.groupby("product")["unit_price"].transform("median")
)
total_filled = (filled["quantity"] * filled["unit_price"]).sum()

print(f"Baris  : dropna = {len(dropped)}, fill = {len(filled)}")
print(f"Total A: Rp {total_dropped:,.0f}")
print(f"Total B: Rp {total_filled:,.0f}")
print(f"Selisih: {(total_filled - total_dropped) / total_filled:.2%}")
```

✅ Output yang diharapkan:

```text
Baris  : dropna = 937, fill = 975
Total A: Rp 104,183,000
Total B: Rp 108,375,000
Selisih: 3.87%
```

💡 Membuang data berarti **kehilangan transaksi yang benar-benar terjadi**. Di laporan keuangan, selisih 3,87% itu besar. Tidak ada cara yang selalu benar; yang penting kamu tahu dampaknya dan menuliskan pilihanmu.

**Eksperimen 2 — Loop vs vectorized**

```python
q = filled["quantity"].to_numpy()
p = filled["unit_price"].to_numpy()

def revenue_loop(q, p):
    hasil = []
    for i in range(len(q)):
        hasil.append(q[i] * p[i])
    return hasil

# Pastikan hasilnya sama dulu sebelum membandingkan kecepatan
print(np.allclose(revenue_loop(q, p), q * p))   # harus True
```

```python
%timeit revenue_loop(q, p)
```

```python
%timeit q * p
```

✅ Versi vectorized seharusnya **ratusan kali lebih cepat**. Di komputer yang dipakai untuk menguji panduan ini: loop ±1,2 ms, sedangkan vectorized ±7 µs. Angkamu bisa berbeda.

💡 `%timeit` adalah perintah khusus notebook (disebut *magic command*) yang menjalankan kode berkali-kali lalu mengambil rata-ratanya. Perintah ini tidak bisa dipakai di file `.py`.

**Catat di `NOTES.md` → Experiment log:**

```markdown
| # | Hypothesis | What I changed | Result | Conclusion |
|---|---|---|---|---|
| 1 | Dropping rows lowers revenue | dropna vs fill with product median | 104.2M vs 108.4M (−3.87%) | Dropping loses real transactions; fill and document it |
| 2 | Vectorized is faster | Python loop vs `q * p` | ~1.2 ms vs ~7 µs | Avoid loops over rows in NumPy/Pandas |
```

---

### Sesi 5 — Review (±1,5 jam)

**1. Tes notebook kosong (45 menit)**

Buat notebook baru `05-blank-test.ipynb`. **Tanpa melihat** notebook lain, lakukan:
- Baca `sales_raw.csv`.
- Temukan keenam masalahnya.
- Bersihkan semuanya.
- Jawab pertanyaan nomor 2: produk dengan revenue terbesar.

Boleh membuka [dokumentasi Pandas](https://pandas.pydata.org/docs/) saja. Kalau berhasil mendapatkan 975 baris dan Madu Hutan di peringkat pertama, kamu lulus. Hapus notebook ini setelah selesai, atau simpan sebagai bukti latihan.

**2. Tes jelaskan dengan suara (15 menit)**

Jawab pertanyaan ini dengan suara, tanpa membaca catatan:
1. Kenapa array NumPy lebih cepat daripada list Python untuk operasi angka?
2. Apa arti `axis=0` dan `axis=1`?
3. Kenapa `(3, 1) + (4,)` berhasil, tetapi `(3,) + (4,)` gagal?
4. Apa beda `.loc` dan `.iloc`?
5. Kenapa harga kosong diisi median **per produk**, bukan median semua data?
6. Kenapa `quantity` terbaca sebagai teks, dan bagaimana kamu memperbaikinya?
7. Apa risikonya membuang baris yang punya nilai kosong?

**3. Lengkapi `NOTES.md`** sampai semua bagiannya terisi.

**4. Centang di [TODO.md](TODO.md)**, Week 1:
- Review NumPy arrays... → Sesi 2a
- Practice Pandas filtering... → Sesi 2b
- Load a CSV and perform basic data-quality checks → Sesi 3, Langkah 4
- Mini-task: analyze a small sales dataset → Sesi 3
- Exit criteria → lulus tes di atas

**5. Commit dan push**

```powershell
git status
git add .
git commit -m "week-01: numpy pandas basics and sales cleaning mini-task"
git push
```

✅ **Week 1 selesai!** 🎉 Isi juga weekly review ([Bagian 11](#bagian-11--weekly-review)), lalu lanjut ke [Bagian 12](#bagian-12--lanjut-ke-week-2-dan-seterusnya).

---

## Bagian 8 — Template

### `NOTES.md` (satu per minggu)

````markdown
# Week XX — <Topic>

## What I learned (in my own words)
- ...

## Code snippets I want to remember
```python
...
```

## Mistakes I made and how I fixed them
- Error: ...  → Cause: ...  → Fix: ...

## Experiment log
| # | Hypothesis | What I changed | Result | Conclusion |
|---|---|---|---|---|

## Exit criteria check
- [ ] Blank notebook test
- [ ] Explain-out-loud test
- [ ] Why test

## Weekly review
1. Exit criteria passed:
2. Hardest part and why:
3. On schedule?
4. Most important thing next week:

## Questions still open
- ...
````

Catatan boleh ditulis dalam bahasa Indonesia atau Inggris. Menulis dalam bahasa Inggris membantu latihan untuk interview dan portfolio.

### Format commit message

```text
week-01: numpy broadcasting exercises
week-01: sales cleaning mini-task
week-01: notes and exit criteria
```

### Sebelum setiap commit

- [ ] **Restart Kernel → Run All** di setiap notebook berjalan tanpa error.
- [ ] `git status` tidak menampilkan `.venv/`, `data/`, atau file `.csv`.
- [ ] `NOTES.md` sudah diperbarui.

---

## Bagian 9 — Sumber dataset

| Sumber | Cocok untuk |
|---|---|
| `sklearn.datasets` (misalnya `fetch_california_housing`, `load_breast_cancer`) | Latihan cepat tanpa download |
| [Kaggle Datasets](https://www.kaggle.com/datasets) | Mini-task dan portfolio (cek lisensinya) |
| [UCI Machine Learning Repository](https://archive.ics.uci.edu/) | Dataset tabular klasik |
| [OpenML](https://www.openml.org/) | Dataset tabular dengan metadata |
| [Hugging Face Datasets](https://huggingface.co/datasets) | Teks dan gambar (Phase 5) |
| `torchvision.datasets` (misalnya `FashionMNIST`, `CIFAR10`) | Deep learning (Phase 4–5) |

**Saran dataset per minggu:**
- Week 1: dataset sintetis di atas, lalu Online Retail (UCI)
- Week 2: dataset yang sama, dimasukkan ke SQLite
- Week 7: California Housing (`sklearn`)
- Week 8: Telco Customer Churn (Kaggle)
- Week 22: FashionMNIST (`torchvision`)

---

## Bagian 10 — Package per phase

Pasang package hanya saat dibutuhkan:

| Phase | Perintah |
|---|---|
| 1 | `uv add numpy pandas matplotlib seaborn jupyterlab ipykernel` (sudah) |
| 2–3 | `uv add scikit-learn` |
| 3 (Portfolio 1) | `uv add --dev pytest` |
| 4 | `uv add torch torchvision` (CPU di laptop; GPU pakai Google Colab) |
| 5A — CV | `uv add opencv-python` |
| 5B — NLP | `uv add transformers datasets sentence-transformers` |
| 6 | `uv add fastapi uvicorn mlflow` |

Portfolio project dibuat di **repo terpisah** dengan environment sendiri, lalu ditautkan dari README repo ini.

---

## Bagian 11 — Weekly review

Setiap hari Minggu, luangkan 15 menit untuk menjawab di `NOTES.md`:

1. Exit criteria mana yang sudah lulus?
2. Bagian apa yang paling sulit, dan kenapa?
3. Apakah aku sesuai jadwal? Kalau tidak, pakai buffer week atau perlambat tempo?
4. Satu hal terpenting yang harus dikerjakan minggu depan?

Lalu perbarui centang di [TODO.md](TODO.md), commit, dan push.

---

## Bagian 12 — Lanjut ke Week 2 dan seterusnya

Panduan ini membahas Week 0 dan Week 1 secara lengkap. Untuk minggu berikutnya, pola belajarnya sama (Bagian 3–5), hanya materinya yang berbeda.

**Saat siap memulai minggu baru**, minta panduan detailnya ke mentor (Claude):

```text
Buatkan GUIDE.md untuk Week 2 (SQL) dengan format yang sama seperti Week 1 di GETTING_STARTED.md.
```

Panduan itu akan dibuat di folder minggunya, misalnya `weeks/week-02-sql/GUIDE.md`, dan berisi:
- Materi Sesi 1 yang perlu dibaca atau ditonton
- Kode Sesi 2 per cell, dengan 🤔 tebakan dan ✅ output yang diharapkan
- Mini-task Sesi 3 langkah demi langkah, beserta dataset-nya
- Eksperimen Sesi 4
- Pertanyaan review Sesi 5 dan item TODO yang dicentang

**Kalau bingung di tengah jalan**, tanyakan dengan template di [Bagian 6](#bagian-6--kalau-error-atau-bingung). Tidak ada pertanyaan yang terlalu dasar. 💪
