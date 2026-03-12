"""Tests for config-driven validation & preprocessing."""
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "configs"))

from cyberbullying.validation import validate_and_preprocess


def test_validate_and_preprocess_basic():
    df = pd.DataFrame({"text": ["Hello world", "Test 123", "Short"], "type": [0, 1, 0]})
    out = validate_and_preprocess(df, config={})
    assert len(out) == 3
    assert "text" in out.columns


def test_validate_drop_duplicates():
    df = pd.DataFrame({"text": ["Same", "Same", "Other"], "type": [0, 1, 0]})
    out = validate_and_preprocess(df, config={"validation": {"drop_duplicates": True}})
    assert len(out) == 2


def test_validate_min_length():
    df = pd.DataFrame({"text": ["Hi", "Hello world", "X"], "type": [0, 1, 0]})
    out = validate_and_preprocess(df, config={"validation": {"min_text_length": 3}})
    assert len(out) == 1
    assert out["text"].iloc[0] == "Hello world"

