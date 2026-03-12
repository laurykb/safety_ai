"""Tests du preprocessing et feature engineering."""
import pytest

from cyberbullying.feature_engineering import clean_text, count_mentions, count_hashtags


def test_clean_text_removes_mentions(sample_text):
    out = clean_text(sample_text)
    assert "@" not in out
    assert "user" not in out or "user" in sample_text.lower()


def test_clean_text_removes_hashtags(sample_text):
    out = clean_text(sample_text)
    assert "#" not in out


def test_clean_text_lowercase(sample_text):
    out = clean_text(sample_text)
    assert out == out.lower()


def test_count_mentions():
    assert count_mentions("Hello @user and @admin") == 2
    assert count_mentions("No mentions") == 0


def test_count_hashtags():
    assert count_hashtags("#covid #test") == 2
    assert count_hashtags("No hashtags") == 0
