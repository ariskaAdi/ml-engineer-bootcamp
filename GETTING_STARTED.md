# Getting Started — How to Use This Bootcamp from Zero

This guide explains **exactly how to start and how to study each week**. [TODO.md](TODO.md) says *what* to learn; this file says *how*.

---

## Step 1 — Check your starting point

Answer honestly. Can you do all of these in Python **without searching**?

- [ ] Use variables, `if/elif/else`, `for` and `while` loops.
- [ ] Use lists, dictionaries, tuples, and sets, and know when to use each one.
- [ ] Write functions with parameters, return values, and default arguments.
- [ ] Read and write a text or CSV file.
- [ ] Handle errors with `try/except`.
- [ ] Import modules and install packages.
- [ ] Write a simple class with `__init__` and methods.
- [ ] Use list comprehensions.
- [ ] Solve this in under 15 minutes: *given a list of words, return a dictionary of each word and how many times it appears.*

**All checked →** start at Week 0 in [TODO.md](TODO.md).  
**Any unchecked →** do Week 0, then the **Python Fundamentals** block in [TODO.md](TODO.md) (2–4 weeks) before Week 1. Do not skip this: weak Python is the most common reason people get stuck in ML courses.

---

## Step 2 — Set up your computer (Week 0, day 1)

The commands below are for Windows (PowerShell or Git Bash). Run them one at a time.

### 2.1 Install the tools

Skip any tool you already have (check with `git --version`, `uv --version`, `code --version`).

```powershell
winget install Git.Git
winget install astral-sh.uv
winget install Microsoft.VisualStudioCode
```

Close and reopen the terminal after installing.

In VS Code, install these extensions: **Python**, **Jupyter**, and optionally **GitLens**.

### 2.2 Configure Git (once per computer)

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
```

### 2.3 Create the Python environment with uv

Inside the repository folder:

```bash
uv python install 3.12        # 3.12 has the widest ML library support
uv init --python 3.12         # creates pyproject.toml and .python-version
uv add numpy pandas matplotlib seaborn jupyterlab ipykernel
```

`uv` creates a `.venv/` folder and records dependencies in `pyproject.toml` and `uv.lock`. You do **not** need a separate `requirements.txt` when using uv. If someone needs one, generate it with:

```bash
uv export --format requirements-txt > requirements.txt
```

### 2.4 Create the folder structure

```text
ml-engineer-bootcamp/
├── README.md
├── TODO.md
├── GETTING_STARTED.md
├── pyproject.toml
├── .gitignore
├── data/                    # datasets — NOT committed to Git
├── weeks/
│   ├── week-00-setup/
│   │   └── NOTES.md
│   ├── week-01-numpy-pandas/
│   │   ├── NOTES.md
│   │   ├── 01-numpy-basics.ipynb
│   │   └── 02-mini-task-sales.ipynb
│   └── ...
└── src/                     # reusable Python code, added from Phase 3
```

Create `.gitignore` with:

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

### 2.5 Test that everything works

```bash
uv run jupyter lab
```

Create `weeks/week-00-setup/hello.ipynb`, select the `.venv` kernel, and run:

```python
import numpy as np, pandas as pd
print(np.__version__, pd.__version__)
```

### 2.6 First commit

```bash
git add .
git commit -m "week-00: project setup"
git push
```

✅ Week 0 is done when you can delete `.venv/`, run `uv sync`, and everything works again.

---

## Step 3 — How to study one week

Each week is about **7–10 hours**, split into **5 sessions of about 1.5–2 hours**. Use the same pattern every week:

| Session | What to do | Output |
|---|---|---|
| **1 — Learn** | Read the TODO tasks for the week. Watch 1 video and read 1 resource per topic (see *Free references*). Take short notes. | `NOTES.md` started |
| **2 — Reproduce** | Retype the tutorial examples yourself (do not copy-paste). Change one thing and predict the result before running. | `01-<topic>.ipynb` |
| **3 — Mini-task** | Do the week's mini-task on a **different dataset** from the tutorial. | `02-mini-task.ipynb` |
| **4 — Experiment** | Try variations: different parameters, features, or models. Write down what changed and why. | Experiment log in `NOTES.md` |
| **5 — Review** | Test yourself against the exit criteria (see Step 5), finish notes, update checkboxes in `TODO.md`, commit and push. | Commit `week-XX: ...` |

**Example schedule** (adjust to your life):

| Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|---|---|---|---|---|---|---|
| Session 1 | Session 2 | rest | Session 3 | rest | Session 4 | Session 5 + weekly review |

Consistency matters more than long sessions. Five 2-hour sessions beat one 10-hour weekend.

---

## Step 4 — How to learn one topic (the learning loop)

For **every** new concept (for example, "broadcasting" or "cross-validation"):

1. **See it** — watch one short video or read one section. Stop when you get the idea; do not binge.
2. **Type it** — write the smallest working example yourself.
3. **Break it** — change inputs, shapes, or parameters. Predict the output first, then run. Wrong predictions are where learning happens.
4. **Explain it** — write 3–5 sentences in `NOTES.md` in your own words, as if teaching a friend.
5. **Rebuild it** — the next day, open a blank notebook and do it again without looking.

If you cannot do step 5, you have not learned it yet. Go back to step 2.

---

## Step 5 — How to know you can check off an exit criterion

Use these three tests:

1. **Blank notebook test** — rebuild the result from an empty notebook, using only official documentation (no tutorial, no AI).
2. **Explain-out-loud test** — explain the concept out loud in 2 minutes. Recording yourself helps.
3. **Why test** — answer "why this choice and not another?" (for example, "why F1 and not accuracy here?").

Pass all three → check the box. Fail any → use the buffer week.

---

## Step 6 — Templates

### NOTES.md (one per week)

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
| # | Hypothesis | What I changed | Result (metric) | Conclusion |
|---|---|---|---|---|
| 1 | Scaling helps KNN | Added StandardScaler | F1 0.71 → 0.78 | Yes, KNN is distance-based |

## Exit criteria check
- [ ] Blank notebook test
- [ ] Explain-out-loud test
- [ ] Why test

## Questions still open
- ...
````

### Commit message convention

```text
week-01: numpy broadcasting exercises
week-01: mini-task sales analysis
week-01: notes and exit criteria
```

---

## Step 7 — When you get stuck

Follow this order, and **spend at least 20–30 minutes** before moving to the next step:

1. **Read the full error message**, especially the last line and the line of your code it points to.
2. **Check shapes and types** — `print(x.shape, x.dtype, type(x))`. Most ML bugs are shape or type bugs.
3. **Make it smaller** — reproduce the problem with 5 rows of data or a tiny tensor.
4. **Read the official documentation** for the function you are using.
5. **Search the exact error message** (Stack Overflow, GitHub issues).
6. **Ask AI — the right way:**
   - ✅ "Explain what this error means and what I should check. Don't give me the fixed code."
   - ✅ "Here is my explanation of cross-validation. What is wrong or missing?"
   - ✅ "Review my notebook and point out data leakage, without rewriting it."
   - ❌ "Write the solution for this mini-task."
7. **Write it down** in the "Mistakes I made" section of `NOTES.md`. You will hit the same bug again.

---

## Step 8 — Where to find datasets

| Source | Good for |
|---|---|
| `sklearn.datasets` (for example, `fetch_california_housing`, `load_breast_cancer`) | Quick practice, no download needed |
| [Kaggle Datasets](https://www.kaggle.com/datasets) | Mini-tasks and portfolio projects (check the license) |
| [UCI Machine Learning Repository](https://archive.ics.uci.edu/) | Classic tabular datasets |
| [OpenML](https://www.openml.org/) | Tabular datasets with metadata |
| [Hugging Face Datasets](https://huggingface.co/datasets) | Text and image datasets (Phase 5) |
| `torchvision.datasets` (for example, `FashionMNIST`, `CIFAR10`) | Phase 4–5 deep learning practice |

**Suggested first datasets:**
- Week 1: Online Retail (UCI) or any Kaggle "sales" dataset
- Week 2: the same dataset loaded into SQLite
- Week 7: California Housing (`sklearn`)
- Week 8: Telco Customer Churn (Kaggle)
- Week 22: FashionMNIST (`torchvision`)

---

## Step 9 — Install packages only when a phase needs them

| Phase | Command |
|---|---|
| 1 | `uv add numpy pandas matplotlib seaborn jupyterlab ipykernel` |
| 2–3 | `uv add scikit-learn` |
| 3 (Portfolio 1) | `uv add --dev pytest` |
| 4 | `uv add torch torchvision` (CPU locally; use Google Colab for GPU) |
| 5A — CV | `uv add opencv-python` |
| 5B — NLP | `uv add transformers datasets sentence-transformers` |
| 6 | `uv add fastapi uvicorn mlflow` |

Portfolio projects should live in **separate repositories** with their own environment, linked from this repo's README.

---

## Step 10 — Weekly review (every Sunday, 15 minutes)

Answer these in the week's `NOTES.md`:

1. Which exit criteria did I pass?
2. What was hardest, and why?
3. Am I on schedule? If not, do I use the buffer week or slow down?
4. What is the single most important thing to do next week?

Then update the checkboxes in [TODO.md](TODO.md), commit, and push.

---

## Week 1 — a worked example

This is what Week 1 looks like when you follow the steps above:

| Session | Concrete actions |
|---|---|
| **1** | Read the [NumPy absolute beginners guide](https://numpy.org/doc/stable/user/absolute_beginners.html) and the first 4 [Pandas tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/). Watch one Corey Schafer Pandas video. Start `NOTES.md`. |
| **2** | In `01-numpy-basics.ipynb`: create arrays, check `.shape`, slice them, and try broadcasting (for example, a `(3, 1)` array plus a `(4,)` array — predict the result shape first). In `02-pandas-basics.ipynb`: `read_csv`, `head`, `info`, `describe`, filtering, `groupby`, `merge`. |
| **3** | Download a sales dataset. In `03-mini-task-sales.ipynb`: check missing values, wrong data types, and duplicates; clean them; calculate total sales per month and top 5 products. Write one sentence explaining every cleaning step. |
| **4** | Experiment: compare `dropna()` vs filling missing values — how do the totals change? Rewrite one loop as a vectorized operation and time both with `%timeit`. Log the results. |
| **5** | Blank notebook test: clean the dataset again without looking. Explain each transformation out loud. Check the boxes in `TODO.md`, then run `git commit -m "week-01: numpy pandas and sales mini-task"` and `git push`. |
