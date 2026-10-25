"""Model palsu kecil (TF-IDF + LogReg pada 8 pesan) dengan kontrak yang sama dengan model asli."""

import json
from datetime import datetime, timezone

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

SPAM = ["WINNER! claim your FREE prize now, call 09061701461", "URGENT you have won cash, txt WIN to 80086",
        "Free entry to win a holiday, reply YES", "Congratulations you won a free mobile, call now"]
HAM = ["are we still meeting for lunch today?", "ok see you at home later", "sorry i missed your call",
       "can you pick up some milk on the way"]


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    pipe = make_pipeline(TfidfVectorizer(), LogisticRegression(C=10)).fit(SPAM + HAM, [1] * 4 + [0] * 4)
    joblib.dump(pipe, d / "model.joblib")
    (d / "metadata.json").write_text(json.dumps({
        "model_name": "test", "trained_at": datetime.now(timezone.utc).isoformat(), "threshold": 0.5,
        "data_sha256": "ab" * 32, "test": {}, "target_precision": 0.98}))
    return d


@pytest.fixture()
def client(model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}")
    from app.main import app
    with TestClient(app) as c:
        yield c
