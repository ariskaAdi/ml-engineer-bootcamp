"""Training-serving skew guard: normalisasi di service HARUS identik dengan di training."""

import pytest

from app.model import normalize_text as serve_norm
from spam.data import normalize_text as train_norm


@pytest.mark.parametrize("text", ["  hello\tworld \n", "FREE!!! £1000 call 0800", "a  b   c", "ok"])
def test_same_normalization(text):
    assert serve_norm(text) == train_norm(text)
