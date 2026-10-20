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