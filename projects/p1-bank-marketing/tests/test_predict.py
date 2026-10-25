import pytest

from bank_marketing.config import ARTIFACTS_DIR
from bank_marketing.predict import EXAMPLE_CLIENT, predict

needs_model = pytest.mark.skipif(not (ARTIFACTS_DIR / "model.joblib").exists(),
                                 reason="jalankan train.py dulu")


def test_predict_rejects_missing_field():
    bad = {k: v for k, v in EXAMPLE_CLIENT.items() if k != "age"}
    with pytest.raises(ValueError, match="age"):
        predict([bad])


@needs_model
def test_predict_returns_probability_and_decision():
    out = predict([EXAMPLE_CLIENT])[0]
    assert 0 <= out["probability"] <= 1
    assert isinstance(out["should_call"], bool)


@needs_model
def test_previous_success_increases_probability():
    base = predict([EXAMPLE_CLIENT])[0]["probability"]
    success = predict([{**EXAMPLE_CLIENT, "poutcome": "success", "pdays": 6, "previous": 1}])[0]["probability"]
    assert success > base
