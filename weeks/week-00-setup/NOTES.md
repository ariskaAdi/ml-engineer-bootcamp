# Week 00 — Setup

## What I set up
- Git, uv, VS Code, Python 3.12 (project-only via uv)
- Packages: numpy, pandas, matplotlib, seaborn, jupyterlab, ipykernel

## Mistakes I made and how I fixed them
- Error: SRE module mismatch → Cause: matplotlib has sub modul pyplot → Fix: import matplotlib.pyplot as plt

## What I learned
- `uv run` runs code with the project's Python, not the global one.
- `.venv/` is not committed; anyone can recreate it with `uv sync`.
```

## Analogi untuk orang awam
- **Git** → seperti fitur "riwayat versi" di Google Docs, tapi untuk seluruh folder proyek. Setiap *commit* adalah foto (snapshot) yang bisa kamu buka lagi kapan saja.
- **Branch** → membuat salinan draf untuk mencoba ide baru tanpa merusak versi utama; kalau bagus, digabung (*merge*).
- **Virtual environment (`uv`)** → kotak perkakas khusus untuk satu proyek. Proyek A boleh memakai obeng versi 1, proyek B versi 2, tanpa saling mengganggu.
- **`pyproject.toml` + `uv.lock`** → daftar belanja yang sangat detail (merek dan ukuran persis), sehingga siapa pun bisa membeli barang yang sama persis.
- **`.gitignore`** → daftar barang yang tidak ikut dimasukkan ke koper (data besar, password, file sementara).

