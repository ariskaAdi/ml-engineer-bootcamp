"""Fixture: model kecil palsu dengan kontrak yang SAMA (kolom & metadata) seperti model P1.
Jadi tes API tidak butuh artefak P1 dan bisa berjalan di CI."""

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

VALID_CLIENT = {
    "age": 35, "job": "admin.", "marital": "married", "education": "university.degree", "default": "no",
    "housing": "yes", "loan": "no", "contact": "cellular", "month": "may", "day_of_week": "thu",
    "campaign": 1, "pdays": 999, "previous": 0, "poutcome": "nonexistent", "emp_var_rate": -1.8,
    "cons_price_idx": 92.893, "cons_conf_idx": -46.2, "euribor3m": 1.313, "nr_employed": 5099.1,
}
COLUMNS = ["age", "job", "marital", "education", "default", "housing", "loan", "contact", "month",
           "day_of_week", "campaign", "pdays", "previous", "poutcome", "emp.var.rate", "cons.price.idx",
           "cons.conf.idx", "euribor3m", "nr.employed"]


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    rng = np.random.default_rng(0)
    base = {k.replace("emp_var_rate", "emp.var.rate").replace("cons_price_idx", "cons.price.idx")
            .replace("cons_conf_idx", "cons.conf.idx").replace("nr_employed", "nr.employed"): v
            for k, v in VALID_CLIENT.items()}
    X = pd.DataFrame([base] * 200)
    X["poutcome"] = rng.choice(["nonexistent", "success", "failure"], 200)
    X["euribor3m"] = rng.uniform(0.5, 5, 200)
    y = ((X["poutcome"] == "success") | (X["euribor3m"] < 1.0)).astype(int)
    pipe = Pipeline([
        ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["poutcome", "month"]),
                                    ("num", "passthrough", ["euribor3m", "age"])])),
        ("model", LogisticRegression()),
    ]).fit(X[COLUMNS], y)
    joblib.dump(pipe, d / "model.joblib")
    (d / "metadata.json").write_text(json.dumps({
        "model_name": "test-logreg", "trained_at": datetime.now(timezone.utc).isoformat(),
        "threshold": 0.5, "data_sha256": "deadbeef" * 8, "test": {"pr_auc": 0.5}}))
    return d


@pytest.fixture()
def client(model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    from app.main import app
    with TestClient(app) as c:          # 'with' menjalankan lifespan (memuat model)
        yield c


@pytest.fixture()
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "does-not-exist"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    from app.main import app
    with TestClient(app) as c:
        yield c
