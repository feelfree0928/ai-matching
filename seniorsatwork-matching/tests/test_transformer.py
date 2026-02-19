"""
Unit tests for transformer: PHP deserialize, work history parsing.
"""
import pytest

from etl.transformer import parse_work_experiences, parse_languages, strip_html, infer_seniority


def test_strip_html():
    assert strip_html("<p>Hello</p>") == "Hello"
    assert strip_html("<p>a</p><p>b</p>") == "a b"
    assert strip_html("") == ""
    assert strip_html(None) == ""


def test_parse_languages():
    # PHP serialized: 3 entries, lang + degree
    raw = 'a:3:{i:0;a:2:{s:4:"lang";s:6:"German";s:6:"degree";s:13:"Mother tongue";}i:1;a:2:{s:4:"lang";s:6:"French";s:6:"degree";s:6:"Fluent";}i:2;a:2:{s:4:"lang";s:7:"English";s:6:"degree";s:6:"Fluent";}}'
    out = parse_languages(raw)
    assert len(out) == 3
    assert out[0]["lang"] == "German" and out[0]["degree"] == "Mother tongue"
    assert out[1]["lang"] == "French"
    assert out[2]["lang"] == "English"


def test_parse_languages_empty():
    assert parse_languages("") == []
    assert parse_languages(None) == []
    assert parse_languages("not an array") == []


def test_parse_work_experiences_sample():
    # Correctly length-annotated PHP serialized work entry
    raw = 'a:1:{i:0;a:6:{s:28:"job_field_stellenbezeichnung";s:25:"Sachbearbeiterin Finanzen";s:32:"job_field_stellenbezeichnung_von";s:4:"2021";s:32:"job_field_stellenbezeichnung_bis";s:4:"2026";s:24:"job_field_name_des_unter";s:14:"Anwaltskanzlei";s:44:"job_field_most_experience_branches266d8b19f5";a:1:{i:0;s:27:"Beratung / Treuhand / Recht";}s:22:"job_field_beschreibung";s:35:"<p>Fakturierung und Buchhaltung</p>";}}'
    out = parse_work_experiences(raw)
    assert len(out) == 1
    assert out[0]["raw_title"] == "Sachbearbeiterin Finanzen"
    assert out[0]["start_year"] == 2021
    assert out[0]["end_year"] == 2026
    assert out[0]["years_in_role"] == 5
    assert out[0]["company"] == "Anwaltskanzlei"
    assert "Beratung" in out[0]["industry"]
    assert "Fakturierung" in out[0]["description"]


def test_parse_work_experiences_now():
    raw = 'a:1:{i:0;a:6:{s:28:"job_field_stellenbezeichnung";s:10:"Accountant";s:32:"job_field_stellenbezeichnung_von";s:4:"2020";s:32:"job_field_stellenbezeichnung_bis";s:3:"now";s:24:"job_field_name_des_unter";s:4:"ACME";s:44:"job_field_most_experience_branches266d8b19f5";a:1:{i:0;s:11:"Real Estate";}s:22:"job_field_beschreibung";s:0:"";}}'
    out = parse_work_experiences(raw)
    assert len(out) == 1
    assert out[0]["end_year"] == 2026
    assert out[0]["years_in_role"] >= 5


def test_infer_seniority():
    assert infer_seniority([{"raw_title": "Junior Accountant"}]) == "junior"
    assert infer_seniority([{"raw_title": "Senior Real Estate Accountant"}]) == "senior"
    assert infer_seniority([{"raw_title": "Accounting Manager"}]) == "manager"
    assert infer_seniority([{"raw_title": "Sachbearbeiter Finanzen"}]) == "mid"
    assert infer_seniority([]) == "mid"
