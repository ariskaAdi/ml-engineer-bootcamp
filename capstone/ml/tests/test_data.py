import pandas as pd
import pytest

from spam.data import dedupe, load, normalize_text, split


def test_normalize_keeps_spam_signals():
    assert normalize_text("  FREE!!!\n\n£1000   now ") == "FREE!!! £1000 now"


def test_dedupe_removes_identical_texts():
    df = pd.DataFrame({"text": ["a", "a", "b"], "y": [1, 1, 0]})
    assert dedupe(df)["text"].tolist() == ["a", "b"]


def test_split_has_no_overlap_after_dedupe():
    df = pd.DataFrame({"text": [f"msg {i % 50}" for i in range(200)], "y": [i % 2 for i in range(200)]})
    dev, test = split(dedupe(df))
    assert not set(dev["text"]) & set(test["text"])


def test_load_rejects_unknown_labels(tmp_path):
    p = tmp_path / "bad.tsv"
    p.write_text("ham\thello\nmaybe\tworld\n", encoding="utf-8")
    with pytest.raises(ValueError, match="label"):
        load(p)


def test_load_parses_tab_separated(tmp_path):
    p = tmp_path / "ok.tsv"
    p.write_text('ham\thello "quoted" text\nspam\tWIN  now\n', encoding="utf-8")
    df = load(p)
    assert df["y"].tolist() == [0, 1]
    assert df["text"].tolist() == ['hello "quoted" text', "WIN now"]
