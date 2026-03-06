"""Tests for skills stopwords / clean-skills filtering."""
import pytest

from api.skills_stopwords import (
    SKILLS_STOPWORDS,
    clean_skill_tokens,
    is_clean_skill_token,
    MIN_SKILL_TOKEN_LEN,
)


def test_stopwords_excluded():
    assert not is_clean_skill_token("an")
    assert not is_clean_skill_token("der")
    assert not is_clean_skill_token("im")
    assert not is_clean_skill_token("und")
    assert not is_clean_skill_token("the")
    assert not is_clean_skill_token("and")
    assert not is_clean_skill_token("in")


def test_too_short_excluded():
    assert not is_clean_skill_token("")
    assert not is_clean_skill_token("a")
    assert not is_clean_skill_token("ab")
    assert len("im") == 2 and not is_clean_skill_token("im")
    assert len("an") == 2 and not is_clean_skill_token("an")


def test_clean_skills_pass():
    assert is_clean_skill_token("PMS")
    assert is_clean_skill_token("SAP")
    assert is_clean_skill_token("Excel")
    assert is_clean_skill_token("Buchhaltung")
    assert is_clean_skill_token("check-in")
    assert is_clean_skill_token("night")
    assert is_clean_skill_token("audit")


def test_clean_skill_tokens_filters():
    raw = ["an", "der", "PMS", "im", "SAP", "und", "Excel"]
    got = clean_skill_tokens(raw)
    assert got == ["PMS", "SAP", "Excel"]


def test_clean_skill_tokens_empty():
    assert clean_skill_tokens([]) == []
    assert clean_skill_tokens(["an", "der", "im"]) == []


def test_min_length_constant():
    assert MIN_SKILL_TOKEN_LEN >= 3
