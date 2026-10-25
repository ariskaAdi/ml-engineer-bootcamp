# %% [markdown]
# # Week 37 — Experiment Tracking dan Model Lifecycle dengan MLflow
#
# 1. Setiap eksperimen P1 dicatat: parameter, metrik CV, hash data, versi kode (git), model.
# 2. Model terbaik didaftarkan di Model Registry -> versi 1, 2, ...
# 3. Alias "champion" = versi yang di-deploy. Rollback = pindahkan alias ke versi lama.
# 4. Export versi "champion" ke services/inference-py/model (lengkap dengan run_id untuk penelusuran).
#
# UI: uv run mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlflow.db  -> http://localhost:5000

# %%
import hashlib
import json
import os
import shutil
import subprocess
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split  # noqa: E402

from mlbootcamp.paths import OUTPUTS_DIR, ROOT  # noqa: E402

P1_SRC = ROOT / "projects/p1-bank-marketing/src"
sys.path.insert(0, str(P1_SRC))
from bank_marketing.config import RAW_CSV, SEED, TARGET  # noqa: E402
from bank_marketing.data import load_dataset  # noqa: E402
from bank_marketing.features import build_pipeline  # noqa: E402

MLFLOW_DIR = OUTPUTS_DIR / "mlflow"
MLFLOW_DIR.mkdir(exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').as_posix()}")
EXPERIMENT = "bank-marketing"
if mlflow.get_experiment_by_name(EXPERIMENT) is None:
    mlflow.create_experiment(EXPERIMENT, artifact_location=(MLFLOW_DIR / "artifacts").as_uri())
mlflow.set_experiment(EXPERIMENT)
MODEL_NAME = "bank-marketing"
client = MlflowClient()

# %% Data + versi data (hash) + versi kode (git commit)
df = load_dataset()
X, y = df.drop(columns=[TARGET]), df[TARGET]
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
data_sha = hashlib.sha256(RAW_CSV.read_bytes()).hexdigest()
try:
    git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
except (OSError, subprocess.CalledProcessError):
    git_sha = "unknown"
print(f"data sha256={data_sha[:12]}  git={git_sha}  dev={len(X_dev)} test={len(X_test)}")

# %% Jalankan dan catat eksperimen
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
experiments = {
    "logreg_C0.5": (LogisticRegression(max_iter=2000, C=0.5), True),
    "logreg_C0.05": (LogisticRegression(max_iter=2000, C=0.05), True),
    "hgb_depth4": (HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=4,
                                                  min_samples_leaf=50, l2_regularization=1.0, random_state=SEED), False),
}
runs = {}
for run_name, (estimator, scale) in experiments.items():
    pipe = build_pipeline(estimator, scale=scale)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"data_sha256": data_sha, "git_commit": git_sha, "dataset": RAW_CSV.name,
                         "project": "p1-bank-marketing"})
        mlflow.log_params({"model": type(estimator).__name__, "scale": scale, "seed": SEED, "cv_folds": 5,
                           **{k: v for k, v in estimator.get_params().items() if isinstance(v, (int, float, str)) and v is not None}})
        r = cross_validate(pipe, X_dev, y_dev, cv=cv, scoring=["average_precision", "roc_auc"])
        metrics = {"cv_pr_auc": r["test_average_precision"].mean(), "cv_pr_auc_std": r["test_average_precision"].std(),
                   "cv_roc_auc": r["test_roc_auc"].mean()}
        mlflow.log_metrics(metrics)
        oof = cross_val_predict(build_pipeline(estimator, scale=scale), X_dev, y_dev, cv=cv, method="predict_proba")[:, 1]
        threshold = float(np.quantile(oof, 0.8))
        pipe.fit(X_dev, y_dev)
        mlflow.log_metric("threshold_top20", threshold)
        mlflow.sklearn.log_model(
            pipe, name="model",
            signature=infer_signature(X_dev.head(5), pipe.predict_proba(X_dev.head(5))),
            input_example=X_dev.head(2),
            code_paths=[str(P1_SRC / "bank_marketing")],     # kode yang dibutuhkan model ikut tersimpan
            # MLflow 3 menyimpan model sklearn dengan format `skops` (lebih aman daripada pickle): saat load,
            # hanya tipe yang dipercaya yang boleh dibuat. Fungsi buatan kita harus didaftarkan eksplisit.
            # TreePredictor (HistGradientBoosting) juga harus dipercaya eksplisit. Aman di sini karena file
            # model dibuat sendiri. JANGAN memercayai tipe dari file model yang diunduh dari sumber asing.
            skops_trusted_types=["bank_marketing.features.add_features",
                                 "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"],
        )
        runs[run_name] = {"run_id": run.info.run_id, "threshold": threshold, **metrics}
        print(f"{run_name:<14} PR-AUC={metrics['cv_pr_auc']:.4f} +- {metrics['cv_pr_auc_std']:.4f}  run_id={run.info.run_id[:8]}")

# %% Cari eksperimen lewat API (bukan hanya UI)
found = mlflow.search_runs(experiment_names=[EXPERIMENT], order_by=["metrics.cv_pr_auc DESC"], max_results=5)
print(found[["tags.mlflow.runName", "metrics.cv_pr_auc", "metrics.cv_roc_auc", "tags.data_sha256"]].head().to_string())

# %% Registry: daftarkan dua versi, tentukan champion
def register(run_name):
    mv = mlflow.register_model(f"runs:/{runs[run_name]['run_id']}/model", MODEL_NAME)
    client.set_model_version_tag(MODEL_NAME, mv.version, "threshold_top20", str(runs[run_name]["threshold"]))
    return mv.version

v_logreg = register("logreg_C0.5")      # model yang dipilih di P1 (aturan 1-SE)
v_hgb = register("hgb_depth4")          # kandidat lain
client.set_registered_model_alias(MODEL_NAME, "champion", v_logreg)
client.set_registered_model_alias(MODEL_NAME, "challenger", v_hgb)
print(f"champion=v{v_logreg}  challenger=v{v_hgb}")

# %% Muat berdasarkan ALIAS (kode produksi tidak perlu tahu nomor versi)
champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
print("champion PR-AUC di test:", round(float(__import__("sklearn.metrics").metrics.average_precision_score(
    y_test, champion.predict_proba(X_test)[:, 1])), 4))

# %% Export versi champion ke service (model package + metadata yang bisa ditelusuri)
def export_alias(alias: str):
    mv = client.get_model_version_by_alias(MODEL_NAME, alias)
    run = client.get_run(mv.run_id)
    local = mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}/{mv.version}")
    dest = ROOT / "services/inference-py/model"
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "code").mkdir(parents=True)
    import joblib
    joblib.dump(mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{mv.version}"), dest / "model.joblib")
    shutil.copytree(os.path.join(local, "code", "bank_marketing"), dest / "code" / "bank_marketing")
    meta = {"model_name": run.data.tags["mlflow.runName"], "trained_at": mv.creation_timestamp and
            __import__("datetime").datetime.fromtimestamp(mv.creation_timestamp / 1000, __import__("datetime").timezone.utc).isoformat(),
            "threshold": float(mv.tags["threshold_top20"]), "data_sha256": run.data.tags["data_sha256"],
            "mlflow_run_id": mv.run_id, "registry_version": int(mv.version), "alias": alias,
            "test": {"cv_pr_auc": round(run.data.metrics["cv_pr_auc"], 4)}}
    (dest / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"diekspor: {MODEL_NAME} v{mv.version} (@{alias}) run={mv.run_id[:8]} -> {dest}")
    return meta

# %% Skenario: promosikan challenger, lalu ROLLBACK
client.set_registered_model_alias(MODEL_NAME, "champion", v_hgb)          # "deploy" v_hgb
print("setelah promosi, champion =", client.get_model_version_by_alias(MODEL_NAME, "champion").version)
# ... misalnya monitoring menunjukkan masalah -> rollback = arahkan alias kembali ke versi lama
client.set_registered_model_alias(MODEL_NAME, "champion", v_logreg)
print("setelah rollback, champion =", client.get_model_version_by_alias(MODEL_NAME, "champion").version)
meta = export_alias("champion")
print(json.dumps(meta, indent=2))
