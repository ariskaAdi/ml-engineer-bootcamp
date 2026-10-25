import pytest

from news_topics.data import ARTIFACTS, LABELS, clean_text
from news_topics.predict import predict

needs_model = pytest.mark.skipif(not (ARTIFACTS / "tfidf_logreg.joblib").exists(), reason="jalankan train.py dulu")


def test_clean_text_fixes_html_escapes():
    raw = "Oil prices rise #39;again#39; &lt;b&gt;today&lt;/b&gt;  \\ fast"
    assert clean_text(raw) == "Oil prices rise 'again' today fast"


def test_predict_rejects_empty_text():
    with pytest.raises(ValueError):
        predict(["   "])


@needs_model
def test_predict_output_format():
    out = predict(["Stocks fell sharply as the central bank raised interest rates"])[0]
    assert out["label"] in LABELS
    assert abs(sum(out["scores"].values()) - 1) < 1e-3


@needs_model
@pytest.mark.parametrize("text,expected", [
    ("The striker scored twice as the team won the championship final", "Sports"),
    ("NASA launches a new telescope to study distant galaxies", "Sci/Tech"),
    ("Shares of the bank jumped after quarterly profit beat forecasts", "Business"),
    ("Peace talks resumed between the two countries after the ceasefire", "World"),
])
def test_obvious_examples(text, expected):
    assert predict([text])[0]["label"] == expected
