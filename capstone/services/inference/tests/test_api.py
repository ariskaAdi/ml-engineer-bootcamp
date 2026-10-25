import sqlite3

import pytest


def test_classify_spam_and_ham(client):
    r = client.post("/v1/classify", json={"instances": [{"text": "WINNER claim your FREE prize, call now"},
                                                        {"text": "see you at lunch later"}]})
    assert r.status_code == 200
    spam, ham = r.json()["predictions"]
    assert spam["spam_probability"] > ham["spam_probability"]
    assert spam["is_spam"] and not ham["is_spam"]


def test_raw_text_is_not_stored(client, tmp_path):
    secret = "my bank PIN is 4321, see you at lunch"
    client.post("/v1/classify", json={"instances": [{"text": secret}]})
    con = sqlite3.connect(tmp_path / "t.db")
    row = con.execute("SELECT * FROM sms_predictions").fetchone()
    assert row is not None
    assert not any(isinstance(v, str) and "4321" in v for v in row)     # isi pesan tidak ada di DB
    assert len(secret) in row                                            # tapi panjangnya ada


@pytest.mark.parametrize("payload", [
    {"instances": []},
    {"instances": [{"text": ""}]},
    {"instances": [{"text": "   "}]},
    {"instances": [{"text": "x" * 1001}]},
    {"instances": [{"text": 123}]},
    {"instances": [{"text": "hi", "sender": "x"}]},
    {"messages": [{"text": "hi"}]},
])
def test_invalid_payloads_rejected(client, payload):
    assert client.post("/v1/classify", json=payload).status_code == 422


def test_batch_limit(client):
    assert client.post("/v1/classify", json={"instances": [{"text": "hi"}] * 51}).status_code == 413


def test_whitespace_normalized_like_training(client):
    a = client.post("/v1/classify", json={"instances": [{"text": "free   prize\n\nnow"}]}).json()
    b = client.post("/v1/classify", json={"instances": [{"text": "free prize now"}]}).json()
    assert a["predictions"][0]["spam_probability"] == b["predictions"][0]["spam_probability"]


def test_ready_and_model_info(client):
    assert client.get("/ready").status_code == 200
    assert client.get("/v1/model-info").json()["threshold"] == 0.5
