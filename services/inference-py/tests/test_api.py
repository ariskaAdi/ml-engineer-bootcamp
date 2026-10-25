import sqlite3

import pytest

from tests.conftest import VALID_CLIENT


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_reports_model_version(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["model_version"].startswith("test-logreg-")


def test_predict_success(client):
    r = client.post("/v1/predict", json={"instances": [VALID_CLIENT, {**VALID_CLIENT, "poutcome": "success"}]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 2
    for p in body["predictions"]:
        assert 0 <= p["probability"] <= 1
        assert isinstance(p["should_call"], bool)
    assert body["predictions"][1]["probability"] > body["predictions"][0]["probability"]
    assert "X-Request-ID" in r.headers


def test_predict_is_logged_to_database(client, tmp_path):
    client.post("/v1/predict", json={"instances": [VALID_CLIENT]}, headers={"X-Request-ID": "abc123"})
    con = sqlite3.connect(tmp_path / "test.db")
    rows = con.execute("SELECT request_id, probability FROM predictions").fetchall()
    assert rows and rows[-1][0] == "abc123"


@pytest.mark.parametrize("change,reason", [
    ({"age": 5}, "umur di bawah batas"),
    ({"age": 250}, "umur di atas batas"),
    ({"job": "astronaut"}, "kategori tidak dikenal"),
    ({"campaign": 0}, "campaign minimal 1"),
    ({"euribor3m": "tinggi"}, "tipe salah"),
    ({"unexpected_field": 1}, "field tak dikenal"),
])
def test_predict_rejects_invalid_input(client, change, reason):
    r = client.post("/v1/predict", json={"instances": [{**VALID_CLIENT, **change}]})
    assert r.status_code == 422, reason


def test_predict_rejects_missing_field(client):
    bad = {k: v for k, v in VALID_CLIENT.items() if k != "age"}
    assert client.post("/v1/predict", json={"instances": [bad]}).status_code == 422


def test_predict_rejects_empty_and_malformed(client):
    assert client.post("/v1/predict", json={"instances": []}).status_code == 422
    assert client.post("/v1/predict", content=b"{not json", headers={"Content-Type": "application/json"}).status_code == 422


def test_predict_rejects_too_large_batch(client, monkeypatch):
    r = client.post("/v1/predict", json={"instances": [VALID_CLIENT] * 101})
    assert r.status_code == 413


def test_model_not_loaded_returns_503(client_without_model):
    assert client_without_model.get("/health").status_code == 200        # proses tetap hidup
    assert client_without_model.get("/ready").status_code == 503
    assert client_without_model.post("/v1/predict", json={"instances": [VALID_CLIENT]}).status_code == 503


def test_inference_failure_returns_500_without_details(client, monkeypatch):
    from app.main import app

    def boom(_):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(app.state.model, "predict_proba", boom)
    r = client.post("/v1/predict", json={"instances": [VALID_CLIENT]})
    assert r.status_code == 500
    assert "secret" not in r.text
