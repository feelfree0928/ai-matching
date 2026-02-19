"""
Unit tests for seniority level inference from job titles.
"""
import pytest

from etl.transformer import infer_seniority


def test_junior():
    assert infer_seniority([{"raw_title": "Junior Accountant"}]) == "junior"
    assert infer_seniority([{"raw_title": "Assistant Controller"}]) == "junior"
    assert infer_seniority([{"raw_title": "Praktikant Buchhaltung"}]) == "junior"


def test_mid():
    assert infer_seniority([{"raw_title": "Sachbearbeiter Finanzen"}]) == "mid"
    assert infer_seniority([{"raw_title": "Specialist Accounting"}]) == "mid"
    assert infer_seniority([{"raw_title": "Fachmann Rechnungswesen"}]) == "mid"


def test_senior():
    assert infer_seniority([{"raw_title": "Senior Real Estate Accountant"}]) == "senior"
    assert infer_seniority([{"raw_title": "Lead Accountant"}]) == "senior"
    assert infer_seniority([{"raw_title": "Fachexperte Finanzen"}]) == "senior"


def test_manager():
    assert infer_seniority([{"raw_title": "Accounting Manager"}]) == "manager"
    assert infer_seniority([{"raw_title": "Leiter Finanzen"}]) == "manager"
    assert infer_seniority([{"raw_title": "Head of Accounting"}]) == "manager"
    assert infer_seniority([{"raw_title": "Teamleiter Buchhaltung"}]) == "manager"


def test_director():
    assert infer_seniority([{"raw_title": "Director of Finance"}]) == "director"
    assert infer_seniority([{"raw_title": "VP Accounting"}]) == "director"
    assert infer_seniority([{"raw_title": "Bereichsleiter Finanzen"}]) == "director"


def test_executive():
    assert infer_seniority([{"raw_title": "CFO"}]) == "executive"
    assert infer_seniority([{"raw_title": "Geschäftsführer"}]) == "executive"
    assert infer_seniority([{"raw_title": "Präsident Verwaltungsrat"}]) == "executive"


def test_most_recent_used():
    # First (most recent) title determines level
    assert infer_seniority([
        {"raw_title": "Senior Accountant"},
        {"raw_title": "Junior Accountant"},
    ]) == "senior"
    assert infer_seniority([
        {"raw_title": "Junior Accountant"},
        {"raw_title": "Senior Accountant"},
    ]) == "junior"


def test_empty_default_mid():
    assert infer_seniority([]) == "mid"
    assert infer_seniority([{"raw_title": ""}]) == "mid"
