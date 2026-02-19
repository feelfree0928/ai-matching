"""
Unit tests for experience recency decay formula.
"""
import pytest

from etl.experience_scorer import recency_weight

CURRENT_YEAR = 2026


def test_recency_weight_current_year():
    assert recency_weight(CURRENT_YEAR) == 1.0
    assert recency_weight(CURRENT_YEAR + 1) == 1.0


def test_recency_weight_3_years_ago():
    # 1.0 - 3*0.04 = 0.88
    assert abs(recency_weight(CURRENT_YEAR - 3) - 0.88) < 0.01


def test_recency_weight_5_years_ago():
    # 1.0 - 5*0.04 = 0.80
    assert abs(recency_weight(CURRENT_YEAR - 5) - 0.80) < 0.01


def test_recency_weight_10_years_ago():
    # 0.80 * 0.93^5
    w = recency_weight(CURRENT_YEAR - 10)
    assert 0.5 < w < 0.6


def test_recency_weight_15_years_ago():
    w = recency_weight(CURRENT_YEAR - 15)
    assert 0.35 < w < 0.45


def test_recency_weight_20_years_ago():
    w = recency_weight(CURRENT_YEAR - 20)
    assert 0.1 < w < 0.3


def test_recency_weight_25_years_ago():
    w = recency_weight(CURRENT_YEAR - 25)
    assert w < 0.2
