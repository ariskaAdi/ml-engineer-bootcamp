import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from bank_marketing.features import PDAYS_NEVER, add_features, build_pipeline
from bank_marketing.predict import EXAMPLE_CLIENT


def sample_frame():
    a = {**EXAMPLE_CLIENT, "duration": 300, "pdays": PDAYS_NEVER}
    b = {**EXAMPLE_CLIENT, "duration": 50, "pdays": 6, "poutcome": "success", "previous": 1}
    return pd.DataFrame([a, b])


def test_add_features_removes_leaky_duration():
    assert "duration" not in add_features(sample_frame()).columns


def test_add_features_encodes_pdays_never_contacted():
    out = add_features(sample_frame())
    assert out["was_contacted_before"].tolist() == [0, 1]
    assert out["pdays"].tolist() == [-1, 6]


def test_add_features_does_not_mutate_input():
    df = sample_frame()
    before = df.copy()
    add_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_pipeline_handles_unseen_category():
    train = pd.concat([sample_frame()] * 30, ignore_index=True)
    y = np.tile([0, 1], 30)
    pipe = build_pipeline(LogisticRegression()).fit(train, y)
    new = sample_frame().assign(job="astronaut")          # kategori tidak ada di train
    proba = pipe.predict_proba(new)[:, 1]
    assert proba.shape == (2,)
    assert np.all((proba >= 0) & (proba <= 1))
