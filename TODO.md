# TODO — Self-Study Machine Learning Bootcamp

**Goal:** Build practical foundations for a Junior Machine Learning Engineer role  
**Duration:** 43 weeks (Week 0–42), including 5 flexible buffer weeks — about 38 weeks if every buffer is skipped  
**Study time:** 7–10 hours/week  
**Start date:** ____  
**Target finish date:** ____  
**Budget:** Free learning resources; use free cloud notebooks where possible  
**API/backend preference:** Go for REST APIs; Python for data science, training, and inference experiments

> Progress rule: check off a task only after you can explain the concept and reproduce the result without blindly copying a tutorial. If a phase's exit criteria are not met, use the buffer week to repeat or extend it before moving on.

> Buffer rule: each phase ends with a buffer week. If all exit criteria are already met, skip it (or use it for interview practice) and move on.

## Weekly routine

- [ ] 2 hours — documentation/course and notes
- [ ] 2–3 hours — implement exercises
- [ ] 2 hours — experiments, debugging, and comparison
- [ ] 1–2 hours — review, Git commits, and documentation
- [ ] 30–45 minutes — coding interview practice (Python/DSA, easy–medium), starting from Phase 3

## Core resources (use throughout)

- [StatQuest with Josh Starmer](https://www.youtube.com/@statquest) — statistics and classical ML explained visually.
- [3Blue1Brown](https://www.3blue1brown.com/) — *Essence of Linear Algebra*, *Essence of Calculus*, and *Neural Networks* series.
- [Andrej Karpathy — Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — backpropagation and GPT built from scratch.
- [An Introduction to Statistical Learning with Python (ISLP)](https://www.statlearning.com/) — free textbook for classical ML.
- [DataTalksClub ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) and [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) — free project-based courses with yearly live cohorts.

> Items marked **(paid)** are optional; everything else is free. Links and cohort schedules can change — verify before relying on them.

---

## Week 0 — Setup and Tooling

- [X] Learn Git basics: init, add, commit, branch, merge, `.gitignore`, and writing clear commit messages.
- [X] Create a GitHub account/repository for this bootcamp and push your first commit.
- [ ] Practice terminal basics: navigating directories, files, environment variables, and running scripts.
- [ ] Set up Python with a virtual environment (`venv` or `uv`) and pin dependencies in `requirements.txt` or `pyproject.toml`.
- [ ] Set up Jupyter (local or Google Colab) and VS Code.
- [ ] Create a standard project structure: `src/`, `notebooks/`, `tests/`, `data/` (ignored by Git), `README.md`.

**Exit criteria**
- [ ] Create a new project from scratch, install dependencies in an isolated environment, and push it to GitHub.

**Free references**
- [Pro Git Book](https://git-scm.com/book/en/v2)
- [The Missing Semester of Your CS Education](https://missing.csail.mit.edu/)
- [Python venv Documentation](https://docs.python.org/3/library/venv.html)
- [Corey Schafer (YouTube)](https://www.youtube.com/@coreyms) — Git and Python basics.

---

## Phase 1 — Python, SQL, and Math for Data (Weeks 1–6)

### Week 1: NumPy and Pandas
- [ ] Review NumPy arrays, shapes, indexing, broadcasting, and vectorized operations.
- [ ] Practice Pandas filtering, grouping, merging, missing values, and data types.
- [ ] Load a CSV and perform basic data-quality checks.
- [ ] Mini-task: analyze a small sales or transaction dataset.

**Exit criteria**
- [ ] Clean a dataset and explain every transformation.
- [ ] Calculate descriptive statistics without relying entirely on generated code.

### Week 2: SQL for Data Work
- [ ] Learn `SELECT`, `WHERE`, `ORDER BY`, `GROUP BY`, `HAVING`, and aggregate functions.
- [ ] Practice `INNER JOIN`, `LEFT JOIN`, and subqueries.
- [ ] Learn window functions (`ROW_NUMBER`, `RANK`, `LAG`, running totals).
- [ ] Query a SQLite or PostgreSQL database from Python and load results into Pandas.
- [ ] Mini-task: answer five business questions about the Week 1 dataset using SQL only, then verify with Pandas.

**Exit criteria**
- [ ] Write multi-table queries with joins and aggregations without looking up basic syntax.

### Week 3: Exploratory Data Analysis (EDA)
- [ ] Learn Matplotlib and basic Seaborn.
- [ ] Plot distributions, scatter plots, correlations, and outliers.
- [ ] Write questions before exploring the dataset.
- [ ] Mini-task: create an EDA notebook with at least five useful plots and written interpretations.

**Exit criteria**
- [ ] Explain findings and distinguish correlation from causation.

### Week 4: Statistics and Probability
- [ ] Review mean, median, variance, standard deviation, and distributions.
- [ ] Study sampling, probability, and sampling bias.
- [ ] Learn conditional probability and Bayes' theorem.
- [ ] Understand confidence intervals conceptually.
- [ ] Learn hypothesis testing, p-values, and the basic idea of A/B testing.
- [ ] Mini-task: simulate sampling and distributions using NumPy.

**Exit criteria**
- [ ] Explain why samples vary and why one summary statistic is insufficient.
- [ ] Explain what a p-value does and does not mean.

### Week 5: Linear Algebra and Optimization Intuition
- [ ] Review vectors, matrices, dot products, and transpose.
- [ ] Understand functions, derivatives, gradients, and learning rate intuitively.
- [ ] Understand the chain rule (needed later for backpropagation).
- [ ] Implement simple linear regression with NumPy.
- [ ] Plot loss as parameters change.

**Exit criteria**
- [ ] Explain how gradient descent updates parameters to reduce loss.

### Week 6: Buffer and Review
- [ ] Revisit any unmet exit criteria from Weeks 0–5.
- [ ] Clean up notebooks and commit them with short notes.

**Free references**
- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [NumPy User Guide](https://numpy.org/doc/stable/user/)
- [Pandas Tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/)
- [SQLBolt](https://sqlbolt.com/)
- [Kaggle Learn — Intro to SQL](https://www.kaggle.com/learn/intro-to-sql)
- [Select Star SQL](https://selectstarsql.com/) — interactive SQL book.
- [Python for Data Analysis, 3rd ed. (Wes McKinney)](https://wesmckinney.com/book/) — free online book by the creator of Pandas.
- [Think Stats (Allen Downey)](https://greenteapress.com/wp/think-stats-2e/) — statistics with Python.
- [Mathematics for Machine Learning](https://mml-book.github.io/) — free book; use it as a reference, not cover to cover.
- [Seeing Theory](https://seeing-theory.brown.edu/) — visual introduction to probability and statistics.

**Videos**
- [Corey Schafer](https://www.youtube.com/@coreyms) and [Keith Galli](https://www.youtube.com/@KeithGalli) — Pandas tutorials.
- [freeCodeCamp](https://www.youtube.com/@freecodecamp) — full-length SQL courses.
- [3Blue1Brown — Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) and [Calculus](https://www.3blue1brown.com/topics/calculus) — Weeks 4–5.
- [StatQuest](https://www.youtube.com/@statquest) — statistics fundamentals for Week 4.

---

## Phase 2 — Classical Machine Learning (Weeks 7–13)

### Week 7: First ML Workflow
- [ ] Understand features `X`, target `y`, and train/validation/test splits.
- [ ] Establish a baseline with `DummyRegressor` or `DummyClassifier`.
- [ ] Train Linear Regression.
- [ ] Evaluate with MAE, RMSE, and R².
- [ ] Mini-task: predict house prices.

**Exit criteria**
- [ ] Compare a model with a baseline and explain the metrics.

### Week 8: Classification
- [ ] Learn Logistic Regression, K-Nearest Neighbors, and Decision Trees.
- [ ] Understand confusion matrix, precision, recall, F1, and accuracy.
- [ ] Learn classification thresholds and class imbalance.
- [ ] Mini-task: predict customer churn or another binary target.

**Exit criteria**
- [ ] Choose metrics based on the cost of false positives and false negatives.

### Week 9: Preprocessing, Pipelines, and Feature Engineering
- [ ] Handle missing values with imputers.
- [ ] Learn scaling and one-hot encoding.
- [ ] Create new features (ratios, date parts, binning, interactions) and test whether they help.
- [ ] Use `ColumnTransformer` and `Pipeline`.
- [ ] Rebuild the Week 7 project using a pipeline.

**Exit criteria**
- [ ] Avoid leakage by fitting preprocessing only on training data through the pipeline.

### Week 10: Generalization and Cross-Validation
- [ ] Explain underfitting, overfitting, and generalization.
- [ ] Practice K-fold cross-validation.
- [ ] Understand regularization and model complexity.
- [ ] Review data leakage and appropriate splitting strategies.
- [ ] Learn time-based splits (`TimeSeriesSplit`) and why random splits leak information in time-ordered data.

**Exit criteria**
- [ ] Explain the difference between training and validation scores and choose an appropriate split strategy.

### Week 11: Tree-Based Models
- [ ] Train Decision Tree and Random Forest models.
- [ ] Explore gradient boosting (for example, HistGradientBoosting or a compatible boosting library).
- [ ] Compare with linear/logistic baselines.
- [ ] Inspect feature importance carefully.

**Exit criteria**
- [ ] Select a model based on comparable validation results, not complexity alone.

### Week 12: Hyperparameter Tuning
- [ ] Practice `GridSearchCV` and `RandomizedSearchCV`.
- [ ] Compare tuned models with baselines.
- [ ] Use the test set only for final evaluation.
- [ ] Write a concise experiment report.

**Exit criteria**
- [ ] Produce a reproducible notebook with a baseline, tuned model, final test metrics, and model-selection rationale.

### Week 13: Buffer and Review
- [ ] Revisit any unmet exit criteria from Phase 2.
- [ ] Explain overfitting, leakage, and cross-validation out loud without notes.

**Free references**
- [scikit-learn Getting Started](https://scikit-learn.org/stable/getting_started.html)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Google ML Crash Course Exercises](https://developers.google.com/machine-learning/crash-course/exercises)
- [Kaggle Learn](https://www.kaggle.com/learn)
- [An Introduction to Statistical Learning with Python (ISLP)](https://www.statlearning.com/) — main textbook for this phase.
- *Hands-On Machine Learning* by Aurélien Géron **(paid)** — the [companion notebooks](https://github.com/ageron/handson-ml3) are free.

**Videos and courses**
- [StatQuest](https://www.youtube.com/@statquest) — one video per algorithm or metric before implementing it.
- [Andrew Ng — Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) — free to audit on Coursera.

> Choose one main course style: Andrew Ng is theory-first; fast.ai (Phase 5) is code-first. Do not take both in full at the same time.

---

## Phase 3 — Experimentation and Tabular ML Project (Weeks 14–19)

### Week 14: Unsupervised Learning
- [ ] Learn K-means and clustering evaluation limitations.
- [ ] Understand PCA and dimensionality reduction.
- [ ] Learn why scaling matters for distance-based algorithms.
- [ ] Mini-task: cluster a customer or product dataset.

**Exit criteria**
- [ ] Explain why clusters do not automatically represent meaningful business segments.

### Week 15: Imbalanced Data and Error Analysis
- [ ] Explore class weights and resampling approaches.
- [ ] Compare precision-recall trade-offs.
- [ ] Inspect false positives and false negatives.
- [ ] Run a controlled experiment to address a model weakness.

**Exit criteria**
- [ ] Identify failure cases and propose a testable improvement.

### Week 16: Interpretability and Controlled Experiments
- [ ] Explore permutation importance.
- [ ] Learn the purpose and limitations of SHAP.
- [ ] Change one major experimental factor at a time.
- [ ] Record hypotheses, parameters, metrics, and conclusions.

**Exit criteria**
- [ ] Distinguish predictive association, model explanation, and causation.

### Weeks 17–18: Portfolio Project 1 — Tabular ML

**Week 17 — Data and modeling**
- [ ] Choose a public dataset with an appropriate license.
- [ ] Define the target and evaluation metric.
- [ ] Perform EDA and preprocessing.
- [ ] Build a baseline and compare at least three candidate models.
- [ ] Use cross-validation and a held-out test set.

**Week 18 — Analysis, code quality, and documentation**
- [ ] Conduct error analysis.
- [ ] Move reusable code from notebooks into `src/` modules with type hints.
- [ ] Add basic `pytest` tests for preprocessing functions.
- [ ] Write a README with setup, results, limitations, and reproduction steps.

**Exit criteria**
- [ ] Another person can reproduce the workflow and understand why the final model was selected.
- [ ] Core logic lives in importable Python modules, not only in notebooks.

### Week 19: Buffer and Review
- [ ] Polish Portfolio Project 1 or revisit unmet exit criteria.
- [ ] Start the weekly coding interview practice habit.

**Free references**
- [pytest Documentation](https://docs.pytest.org/)
- [Interpretable Machine Learning (Christoph Molnar)](https://christophm.github.io/interpretable-ml-book/) — free book on permutation importance, SHAP, and their limits (Week 16).
- [DataTalksClub ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) — end-to-end project examples for Portfolio Project 1.

---

## Phase 4 — Deep Learning with PyTorch (Weeks 20–26)

### Week 20: Tensors and GPU Basics
- [ ] Learn tensor shape, dtype, indexing, and broadcasting.
- [ ] Understand CPU vs GPU execution.
- [ ] Use `torch.device` and check CUDA availability.
- [ ] Practice moving tensors and models to the same device.

**Exit criteria**
- [ ] Diagnose a device mismatch and explain why a small workload may not benefit from GPU execution.

### Week 21: Autograd and Neural Networks
- [ ] Understand computational graphs and `requires_grad`.
- [ ] Connect the chain rule from Week 5 to backpropagation.
- [ ] Build models using `nn.Module`.
- [ ] Learn activation functions, loss functions, and optimizers.
- [ ] Train a small classification model.

**Exit criteria**
- [ ] Explain the relationship between prediction, loss, gradients, and parameter updates.

### Week 22: Training Loop
- [ ] Use `Dataset` and `DataLoader`.
- [ ] Implement forward pass, `loss.backward()`, and `optimizer.step()`.
- [ ] Separate training and validation loops.
- [ ] Save and load model state.
- [ ] Practice with FashionMNIST.

**Exit criteria**
- [ ] Write a training loop and record loss and metrics per epoch without copying a complete solution.

### Week 23: Optimization and Regularization
- [ ] Experiment with learning rate and batch size.
- [ ] Compare SGD and Adam.
- [ ] Learn dropout, weight decay, and early stopping.
- [ ] Plot training and validation curves.

**Exit criteria**
- [ ] Diagnose likely overfitting and support the diagnosis with evidence.

### Week 24: GPU Performance and Reproducibility
- [ ] Run an appropriately sized CPU/GPU comparison in a cloud notebook.
- [ ] Measure runtime correctly and repeat measurements where practical.
- [ ] Inspect memory use and data-loading bottlenecks.
- [ ] Record seeds, versions, and experiment settings.
- [ ] Learn when CUDA-specific programming is and is not necessary.

**Exit criteria**
- [ ] Explain the measured performance difference and its limitations.

### Week 25: Portfolio Project 2 — Deep Learning
- [ ] Train a PyTorch model for image or tabular classification.
- [ ] Track training and validation metrics.
- [ ] Save the model and implement inference.
- [ ] Document dataset, configuration, results, and limitations.

**Exit criteria**
- [ ] Re-run training or inference from documented instructions and explain the training loop.

### Week 26: Buffer and Review
- [ ] Revisit any unmet exit criteria from Phase 4.
- [ ] Decide on your Phase 5 track (Computer Vision or NLP).

**Free references**
- [PyTorch Tutorials](https://docs.pytorch.org/tutorials/)
- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [Google Colab](https://colab.research.google.com/)
- [Learn PyTorch for Deep Learning (Daniel Bourke)](https://www.learnpytorch.io/) — free online book; the matching full video course is on freeCodeCamp.
- [Dive into Deep Learning](https://d2l.ai/) — free interactive book with runnable code.

**Videos**
- [3Blue1Brown — Neural Networks](https://www.3blue1brown.com/topics/neural-networks) — intuition for backpropagation (Week 21).
- [Andrej Karpathy — Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — start with micrograd for Weeks 21–22.

---

## Phase 5 — Specialization Track: Choose CV or NLP (Weeks 27–31)

> Pick **one** main track based on the job postings you are targeting. Many current junior ML roles mention NLP, Hugging Face, embeddings, or RAG; computer vision fits roles in manufacturing, retail, health imaging, and similar areas. After the capstone, you can return to the other track.

### Track A — Computer Vision and Transfer Learning

#### Week 27: Image Data and CNN Concepts
- [ ] Understand image tensors, channels, resolution, resizing, and normalization.
- [ ] Learn convolution, filters, pooling, and CNN intuition.
- [ ] Visualize image transformations.

**Exit criteria**
- [ ] Explain how a raw image becomes numerical input to a model.

#### Week 28: TorchVision and Transfer Learning
- [ ] Load datasets and apply image transforms.
- [ ] Use a pretrained model and replace its classification head.
- [ ] Compare frozen features with fine-tuning.
- [ ] Apply data augmentation appropriately.

**Exit criteria**
- [ ] Evaluate on unseen data and describe steps taken to avoid leakage.

#### Week 29: OpenCV and Image Error Analysis
- [ ] Read, resize, and transform images with OpenCV.
- [ ] Keep training and inference preprocessing consistent.
- [ ] Inspect class-wise metrics and misclassified examples.

**Exit criteria**
- [ ] Reproduce the same preprocessing behavior at inference time.

#### Week 30: Portfolio Project 3 — Image Classification
- [ ] Choose a licensed image dataset.
- [ ] Establish a simple baseline.
- [ ] Fine-tune a pretrained model.
- [ ] Report suitable metrics and show correct/incorrect predictions.
- [ ] Document setup and inference.

**Exit criteria**
- [ ] Provide a reproducible project with evaluation and a clear limitations section.

**Free references (Track A)**
- [TorchVision Documentation](https://docs.pytorch.org/vision/stable/index.html)
- [PyTorch Transfer Learning Tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
- [Stanford CS231n](https://cs231n.stanford.edu/) — CNN course notes; lectures are on YouTube.
- [fast.ai — Practical Deep Learning for Coders](https://course.fast.ai/) — code-first course, strong on transfer learning.

**Optional after this track:** explore YOLO and object detection if computer vision becomes your specialization. They are not prerequisites for every ML Engineer role.

### Track B — NLP, Transformers, and LLM Basics

#### Week 27: Text Data and Classical NLP Baselines
- [ ] Learn text cleaning, tokenization, and vocabulary.
- [ ] Build a TF-IDF + Logistic Regression baseline for text classification.
- [ ] Understand why word order and context matter.

**Exit criteria**
- [ ] Explain how raw text becomes numerical input to a model and build a reproducible baseline.

#### Week 28: Transformers and Hugging Face
- [ ] Understand attention and the transformer architecture at an intuitive level.
- [ ] Use Hugging Face `transformers` tokenizers and pretrained models.
- [ ] Fine-tune a small pretrained model (for example, DistilBERT) for text classification.
- [ ] Compare it with the Week 27 baseline.

**Exit criteria**
- [ ] Justify whether the transformer's improvement over the baseline is worth its cost.

#### Week 29: Embeddings, Semantic Search, and RAG Basics
- [ ] Generate sentence embeddings and compute cosine similarity.
- [ ] Build a small semantic search over a document collection.
- [ ] Understand the RAG pattern: retrieve relevant context, then generate an answer.
- [ ] Learn the limitations: hallucination, retrieval quality, evaluation, and cost.

**Exit criteria**
- [ ] Explain when to use fine-tuning, embeddings/retrieval, or a plain prompt.

#### Week 30: Portfolio Project 3 — NLP
- [ ] Choose a licensed text dataset (classification or semantic search).
- [ ] Establish a simple baseline.
- [ ] Fine-tune a pretrained model or build a retrieval pipeline.
- [ ] Report suitable metrics and show good and bad examples.
- [ ] Document setup and inference.

**Exit criteria**
- [ ] Provide a reproducible project with evaluation and a clear limitations section.

**Free references (Track B)**
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Andrej Karpathy — Zero to Hero](https://karpathy.ai/zero-to-hero.html) — the "Let's build GPT" video for Week 28.
- [Stanford CS224N](https://web.stanford.edu/class/cs224n/) — NLP with deep learning; lectures are on YouTube.
- [Speech and Language Processing (Jurafsky & Martin)](https://web.stanford.edu/~jurafsky/slp3/) — free draft textbook; use as a reference.

### Week 31: Buffer and Review
- [ ] Revisit any unmet exit criteria from your chosen track.
- [ ] Optional: skim Week 27 of the other track so you know the basics.

---

## Phase 6 — Deployment and MLOps Basics (Weeks 32–38)

### Week 32: Model Inference and Python Service
- [ ] Separate training code from inference code.
- [ ] Save model artifacts and required preprocessing.
- [ ] Define request/response schemas and input validation.
- [ ] Understand REST API basics.
- [ ] Build a Python inference service (for example, with FastAPI) that loads the model and returns predictions.

**Exit criteria**
- [ ] Send a request directly to the Python service and receive a valid prediction.

### Week 33: Go REST API Integration

**Go integration task**
- [ ] Build a Go REST API using `net/http` or a framework you already know.
- [ ] Accept validated JSON input and return a prediction response.
- [ ] Keep business logic and HTTP handlers separate.
- [ ] Call the Python inference service from Go over internal HTTP/JSON.

**Recommended initial architecture**
- Go API: authentication if needed, business rules, validation, orchestration, and public REST endpoints.
- Python inference service: load the trained model and perform prediction.
- Communication: internal HTTP/JSON between Go and Python.
- Keep the first version simple; do not introduce a message broker unless there is a real need.

**Exit criteria**
- [ ] Send a request to the Go endpoint and receive a valid prediction produced by the Python model.

### Week 34: Testing, Reliability, and CI
- [ ] Validate missing, malformed, and out-of-range inputs.
- [ ] Add unit tests for preprocessing and inference.
- [ ] Test Go handlers and Python inference independently.
- [ ] Handle timeouts and inference failures explicitly.
- [ ] Add structured logs without logging sensitive data.
- [ ] Set up GitHub Actions to run Go and Python tests on every push.

**Exit criteria**
- [ ] Automated tests cover the main success path and important failure cases and run in CI.

### Week 35: Docker and Reproducibility
- [ ] Write a Dockerfile for the Go API.
- [ ] Write a separate Dockerfile for the Python inference service.
- [ ] Configure services using environment variables.
- [ ] Run both services together locally (for example, with Docker Compose) and document startup instructions.
- [ ] Avoid embedding secrets in images or source control.

**Exit criteria**
- [ ] Start the application using documented container commands and verify the end-to-end request flow.

### Week 36: Cloud Deployment and Monitoring Basics
- [ ] Deploy to a free tier (for example, Hugging Face Spaces, Render, or Fly.io) and get a public demo URL.
- [ ] Understand free-tier limits: cold starts, memory, and sleep policies.
- [ ] Log latency, error rate, and prediction distribution.
- [ ] Learn the concepts of data drift and concept drift, and when a model needs retraining.

**Exit criteria**
- [ ] A public endpoint returns predictions, and you can explain which signals would tell you the model is degrading.

### Week 37: Experiment Tracking and Model Lifecycle
- [ ] Track experiment parameters, metrics, and artifacts.
- [ ] Learn the basics of MLflow Tracking.
- [ ] Record model versions and dataset versions where feasible (optional: try DVC).
- [ ] Document how to roll back to a previous model artifact.

**Exit criteria**
- [ ] Identify which experiment produced a deployed model and reproduce its recorded configuration.

### Week 38: Buffer and Review
- [ ] Revisit any unmet exit criteria from Phase 6.

**Free references**
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/) — useful if you need a Python HTTP service.
- [Go Documentation](https://go.dev/doc/)
- [Docker Get Started](https://docs.docker.com/get-started/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking)
- [A Tour of Go](https://go.dev/tour/) and [Go by Example](https://gobyexample.com/) — Go refresher before Week 33.
- [DataTalksClub MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) — experiment tracking, deployment, and monitoring.
- [Made With ML](https://madewithml.com/) — production ML and MLOps practices.
- [Google — Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) — practical ML engineering guidelines.
- *Designing Machine Learning Systems* by Chip Huyen **(paid)** — strongly recommended if you buy one book.

**Videos**
- [TechWorld with Nana](https://www.youtube.com/@TechWorldwithNana) — Docker, Docker Compose, and CI/CD basics.

> Go is a reasonable choice for the surrounding REST API. Python remains the most practical default for model training and the initial inference ecosystem. You do not need to rewrite ML libraries in Go.

---

## Phase 7 — Capstone and Career Preparation (Weeks 39–42)

### Week 39: Define the Problem
- [ ] Choose a practical prediction problem (ideally related to your Phase 5 track).
- [ ] Confirm dataset license and data quality.
- [ ] Define a baseline, target metric, and validation strategy.
- [ ] Write a short experiment plan.

### Week 40: Train and Evaluate
- [ ] Run baseline and candidate experiments.
- [ ] Perform error analysis.
- [ ] Select a model using validation results.
- [ ] Use the test set for final evaluation only.
- [ ] Record limitations and possible improvements.

### Week 41: Integrate and Deploy
- [ ] Package the model for inference.
- [ ] Connect the Python inference service to the Go API.
- [ ] Add tests, CI, and example requests.
- [ ] Deploy and run the end-to-end workflow.
- [ ] Document deployment and configuration.

### Week 42: Portfolio and Interview Practice
- [ ] Clean up the repository and README.
- [ ] Include a clear problem statement, setup instructions, metrics, limitations, and a demo link.
- [ ] Prepare a short technical walkthrough.
- [ ] Practice explaining overfitting, data leakage, cross-validation, model selection, and metric trade-offs.
- [ ] Practice SQL and Python coding questions under time limits.
- [ ] Review the job requirements for junior ML Engineer roles and identify remaining gaps.

**Capstone exit criteria**
- [ ] A reviewer can reproduce the documented inference workflow.
- [ ] Model decisions are supported by experiments.
- [ ] The Go API and Python model service have clear responsibilities.
- [ ] You can explain trade-offs and failure cases without relying on AI-generated explanations.

**Free references**
- [Introduction to Machine Learning Interviews (Chip Huyen)](https://huyenchip.com/ml-interviews-book/) — free book on the ML interview process and questions.
- [NeetCode](https://neetcode.io/) — structured DSA practice with video explanations.
- [StatQuest](https://www.youtube.com/@statquest) — quick concept review before interviews.

---

## Final readiness checklist

- [ ] I can use Git, a terminal, and isolated Python environments confidently.
- [ ] I can query data with SQL, including joins, aggregations, and window functions.
- [ ] I can clean and explore a dataset independently.
- [ ] I can establish a baseline and select appropriate metrics.
- [ ] I understand data leakage, overfitting, and cross-validation.
- [ ] I can train and evaluate classical ML models.
- [ ] I can write a basic PyTorch training loop.
- [ ] I can use a cloud GPU and explain when it helps.
- [ ] I can fine-tune a pretrained model in my chosen track (CV or NLP).
- [ ] I can serve predictions through a Python inference service.
- [ ] I can connect that service to a Go REST API.
- [ ] I can test, document, and reproduce my projects, with tests running in CI.
- [ ] I have deployed at least one project to a public URL.
- [ ] I have at least two polished portfolio projects and one end-to-end deployment.

## Learning rules

1. Prefer official documentation for APIs and use YouTube for visual explanations. For each topic, pick one video and one reading, then practice immediately.
2. Do not learn every library in the same phase; learn a tool when a project requires it.
3. Use AI for explanations, debugging, and review—not as a substitute for implementing and evaluating the model.
4. Keep a record of experiments and explain every metric you report.
5. Free cloud GPU availability and quotas can change. Start with small datasets and CPU where practical; use GPU when the workload justifies it.
6. Buffer weeks are for catching up, not for adding new topics. If you are on track, skip them.
7. This roadmap builds foundations; it does not guarantee job readiness by a fixed date. Assess progress by the exit criteria and project quality.
