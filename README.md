# ML Engineer Bootcamp

My 43-week self-study journey to become a Junior Machine Learning Engineer.

- Roadmap: [TODO.md](TODO.md)
- How I study: [GETTING_STARTED.md](GETTING_STARTED.md)
- Weekly practice: [weeks/](weeks/) — setiap minggu punya `NOTES.md` (penjelasan + hasil) dan script `.py`

## Setup
```powershell
uv sync                     # semua dependency (numpy … torch, transformers, fastapi, mlflow)
```
Script `.py` memakai penanda `# %%`: buka di VS Code lalu **Run Cell** (seperti notebook), atau jalankan sekaligus dengan `uv run python <file>`. Plot tersimpan di `outputs/`, model di `models/`, data di `data/` (ketiganya tidak di-commit).

## Portfolio
| Proyek | Isi | Hasil utama |
|---|---|---|
| [P1 — Bank Marketing](projects/p1-bank-marketing/) | Tabular ML, data UCI asli, leakage `duration`, pytest | Test PR-AUC 0,454; top-20% menangkap 64% nasabah "yes" |
| [P2 — FashionCNN](projects/p2-fashion-cnn/) | CNN PyTorch dari nol, inference file gambar | Test accuracy 0,929 |
| [P3 — News Topics](projects/p3-news-topics/) | TF-IDF vs embeddings vs DistilBERT fine-tune | TF-IDF full 0,923 > DistilBERT (8k) 0,907 |
| [Capstone — SMS Spam Filter](capstone/) | Python inference + Go gateway + PostgreSQL + Docker + CI | Test PR-AUC 0,992, precision/recall 0,969 |
| [Phase 6 services](services/) | FastAPI + Go REST API + monitoring drift + MLflow untuk model P1 | |

## Peta minggu
| Phase | Week | Folder |
|---|---|---|
| 1 — Python, SQL, Math | 0–6 | `week-00` … `week-06-buffer-review` |
| 2 — Classical ML | 7–13 | `week-07-first-ml-workflow` … `week-13-buffer-review` |
| 3 — Tabular project | 14–19 | `week-14-unsupervised` … `week-19-buffer-review` |
| 4 — PyTorch | 20–26 | `week-20-tensors-gpu` … `week-26-buffer-review` |
| 5 — NLP (Track B) | 27–31 | `week-27-nlp-text-baselines` … `week-31-buffer-review` (+ `week-27-29-track-a-cv` untuk CV) |
| 6 — Deployment & MLOps | 32–38 | `week-32-inference-service` … `week-38-buffer-review` |
| 7 — Capstone & career | 39–42 | `week-39-capstone-problem` … `week-42-portfolio-interview` |

## Kode bersama
[src/mlbootcamp/](src/mlbootcamp/): dataset sintetis (`datasets.py`), path (`paths.py`), pipeline rumah (`features.py`), loop training PyTorch (`torch_utils.py`), runner SQL (`sqlrun.py`), log eksperimen (`experiments.py`).

## Tes
```powershell
uv run pytest projects services/inference-py capstone
cd services/api-go; go test ./...
cd capstone/services/gateway-go; go test ./...
```
CI: [.github/workflows/ci.yml](.github/workflows/ci.yml).
