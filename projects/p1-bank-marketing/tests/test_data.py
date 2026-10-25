import pandas as pd
import pytest

from bank_marketing.data import clean, validate
from bank_marketing.predict import EXAMPLE_CLIENT


def make_raw(n=3, **overrides):
    rows = [{**EXAMPLE_CLIENT, "duration": 100, "y": "no"} for _ in range(n)]
    df = pd.DataFrame(rows)
    for k, v in overrides.items():
        df[k] = v
    return df


def test_clean_maps_target_to_int_and_drops_duplicates():
    raw = make_raw(n=3)                      # 3 baris identik -> 1 baris
    raw.loc[0, "y"] = "yes"
    out = clean(raw)
    assert set(out["y"]) <= {0, 1}
    assert out["y"].dtype.kind == "i"
    assert len(out) == 2                     # baris yes + satu baris no unik


def test_validate_rejects_missing_column():
    with pytest.raises(ValueError, match="kolom hilang"):
        validate(make_raw().drop(columns=["age"]))


def test_validate_rejects_unknown_target():
    with pytest.raises(ValueError, match="target"):
        validate(make_raw(y="maybe"))


def test_validate_rejects_impossible_age():
    with pytest.raises(ValueError, match="umur"):
        validate(make_raw(age=200))
