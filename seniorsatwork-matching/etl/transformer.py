"""
Transform raw WordPress resume/job data: PHP deserialize, HTML strip, parse work history, infer seniority.
"""
from __future__ import annotations

import re
import warnings
from functools import lru_cache
from typing import Any

import phpserialize
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

# Resume fields sometimes look like URLs; we intentionally parse them as HTML.
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# Seniority levels (order matters for level index 0-5)
SENIORITY_LEVELS = ["junior", "mid", "senior", "manager", "director", "executive"]

SENIORITY_KEYWORDS = {
    "junior": ["junior", "assistant", "trainee", "praktikant", "azubi", "ausbildung"],
    "mid": ["sachbearbeiter", "specialist", "coordinator", "fachmann", "fachfrau", "mitarbeiter"],
    "senior": ["senior", "lead ", "expert", "principal", "fachexperte", "erfahren"],
    "manager": ["manager", "leiter", "head of", "abteilungsleiter", "teamleiter", "team lead"],
    "director": ["director", "vp ", "vice president", "bereichsleiter", "vice president"],
    "executive": ["ceo", "cfo", "cto", "coo", "geschäftsführer", "vorsitzender", "präsident", "president"],
}


def _decode_value(v: Any) -> Any:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, dict):
        return {_decode_value(k): _decode_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_decode_value(x) for x in v]
    return v


def _decode_php_dict(d: dict) -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = k.decode("utf-8", errors="replace") if isinstance(k, bytes) else k
        if isinstance(v, bytes):
            out[key] = v.decode("utf-8", errors="replace")
        elif isinstance(v, dict):
            out[key] = _decode_php_dict(v)
        elif isinstance(v, (list, tuple)):
            out[key] = [
                _decode_php_dict(x) if isinstance(x, dict) else (_decode_value(x) if isinstance(x, bytes) else x)
                for x in v
            ]
        else:
            out[key] = v
    return out


def strip_html(html: str | None) -> str:
    if not html or not html.strip():
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def _safe_int(s: Any, default: int = 0) -> int:
    if s is None:
        return default
    if isinstance(s, int):
        return s
    try:
        return int(str(s).strip())
    except (ValueError, TypeError):
        return default


def _find_industry_key(entry: dict) -> str:
    """Entry keys may be bytes or str; industry is in job_field_most_experience_branches* (PHP array of strings)."""
    for k in entry:
        key = k.decode("utf-8", errors="replace") if isinstance(k, bytes) else k
        if isinstance(key, str) and "job_field_most_experience_branches" in key:
            val = entry[k]
            if isinstance(val, dict):
                # PHP array: {0: b"Industry A", 1: b"Industry B"}
                for ik in sorted(x for x in val if isinstance(x, int)):
                    v = val[ik]
                    if isinstance(v, bytes):
                        return v.decode("utf-8", errors="replace")
                    return str(v)
            if isinstance(val, (list, tuple)) and val:
                first = val[0]
                return first.decode("utf-8", errors="replace") if isinstance(first, bytes) else str(first)
            if isinstance(val, bytes):
                return val.decode("utf-8", errors="replace")
            return str(val) if val else ""
    return ""


def _get_str(entry: dict, key: str, default: str = "") -> str:
    """Get string from entry; keys in PHP may be bytes."""
    for k, v in entry.items():
        k_str = k.decode("utf-8", errors="replace") if isinstance(k, bytes) else k
        if k_str == key:
            if v is None:
                return default
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="replace")
            return str(v).strip()
    return default


def parse_work_experiences(meta_value: str | None) -> list[dict[str, Any]]:
    """
    Parse _noo_resume_field__taetigkeiten PHP serialized array.
    Returns list of dicts with: raw_title, start_year, end_year, years_in_role, company, industry, description.
    """
    if not meta_value or not meta_value.strip().startswith("a:"):
        return []

    try:
        raw = phpserialize.loads(meta_value.encode("utf-8"))
    except Exception:
        return []

    if not isinstance(raw, dict):
        return []

    result = []
    # PHP arrays are stored as dict with int keys 0, 1, 2...
    # Filter and convert keys to ints to avoid bytes/int comparison issues
    int_keys = []
    for k in raw:
        if isinstance(k, int):
            int_keys.append(k)
        elif isinstance(k, bytes):
            try:
                int_keys.append(int(k))
            except (ValueError, TypeError):
                continue
    for i in sorted(int_keys):
        entry = raw[i]
        if not isinstance(entry, dict):
            continue
        decoded = _decode_php_dict(entry) if any(isinstance(k, bytes) for k in entry) else entry
        
        # Safely extract string values, handling bytes
        def _safe_str(val: Any) -> str:
            if val is None:
                return ""
            if isinstance(val, bytes):
                return val.decode("utf-8", errors="replace")
            return str(val)
        
        raw_title = _get_str(entry, "job_field_stellenbezeichnung") or _safe_str(decoded.get("job_field_stellenbezeichnung", ""))
        von_raw = _get_str(entry, "job_field_stellenbezeichnung_von") or _safe_str(decoded.get("job_field_stellenbezeichnung_von", ""))
        bis_raw = _get_str(entry, "job_field_stellenbezeichnung_bis") or _safe_str(decoded.get("job_field_stellenbezeichnung_bis", ""))
        company = _get_str(entry, "job_field_name_des_unter") or _safe_str(decoded.get("job_field_name_des_unter", ""))
        industry_raw = _find_industry_key(entry) or decoded.get("job_field_most_experience_branches266d8b19f5", "")
        if isinstance(industry_raw, list) and industry_raw:
            industry = _safe_str(industry_raw[0])
        else:
            industry = _safe_str(industry_raw)
        desc_html = _get_str(entry, "job_field_beschreibung") or _safe_str(decoded.get("job_field_beschreibung", ""))
        description = strip_html(desc_html)

        # Handle bytes/str conversion for year fields
        von = von_raw if isinstance(von_raw, str) else (von_raw.decode("utf-8", errors="replace") if isinstance(von_raw, bytes) else str(von_raw or ""))
        bis = bis_raw if isinstance(bis_raw, str) else (bis_raw.decode("utf-8", errors="replace") if isinstance(bis_raw, bytes) else str(bis_raw or ""))
        
        start_year = _safe_int(von, 0)
        bis_lower = bis.strip().lower() if bis else ""
        end_year = 2026 if (bis_lower == "now") else _safe_int(bis, 2026)
        if start_year <= 0 or end_year <= 0:
            continue
        years_in_role = max(1, end_year - start_year)

        result.append({
            "raw_title": raw_title or "",
            "start_year": start_year,
            "end_year": end_year,
            "years_in_role": years_in_role,
            "company": company or "",
            "industry": industry if isinstance(industry, str) else "",
            "description": description,
        })
    return result


def parse_languages(meta_value: str | None) -> list[dict[str, str]]:
    """Parse _noo_resume_field_languages_i_speak. Returns list of {lang, degree}."""
    if not meta_value or not meta_value.strip().startswith("a:"):
        return []

    try:
        raw = phpserialize.loads(meta_value.encode("utf-8"))
    except Exception:
        return []

    if not isinstance(raw, dict):
        return []

    result = []
    # Filter and convert keys to ints to avoid bytes/int comparison issues
    int_keys = []
    for k in raw:
        if isinstance(k, int):
            int_keys.append(k)
        elif isinstance(k, bytes):
            try:
                int_keys.append(int(k))
            except (ValueError, TypeError):
                continue
    for i in sorted(int_keys):
        entry = raw[i]
        if not isinstance(entry, dict):
            continue
        decoded = _decode_php_dict(entry) if any(isinstance(k, bytes) for k in entry) else entry
        lang_raw = decoded.get("lang", "") or _get_str(entry, "lang")
        degree_raw = decoded.get("degree", "") or _get_str(entry, "degree")
        lang = lang_raw.decode("utf-8", errors="replace") if isinstance(lang_raw, bytes) else str(lang_raw or "")
        degree = degree_raw.decode("utf-8", errors="replace") if isinstance(degree_raw, bytes) else str(degree_raw or "")
        if lang:
            result.append({"lang": lang, "degree": degree})
    return result


@lru_cache(maxsize=512)
def _match_seniority(title: str) -> str:
    """Cached seniority match. Uses word-boundary regex to avoid substring false-positives."""
    for level in reversed(SENIORITY_LEVELS):
        for kw in SENIORITY_KEYWORDS[level]:
            pattern = r"\b" + re.escape(kw.strip()) + r"\b"
            if re.search(pattern, title):
                return level
    return "mid"


def infer_seniority(work_experiences: list[dict[str, Any]]) -> str:
    """
    Infer seniority from most recent job title (first in list after parsing).
    Returns one of: junior, mid, senior, manager, director, executive.
    """
    if not work_experiences:
        return "mid"
    title = (work_experiences[0].get("raw_title") or "").lower()
    if not title:
        return "mid"
    return _match_seniority(title)


def transform_candidate(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Transform one raw candidate (from extractor) into a normalized candidate document.
    Does NOT add recency_weight/weighted_years (done in experience_scorer) or standardized_title (done in title_standardizer).
    """
    meta = raw.get("meta") or {}
    work_experiences = parse_work_experiences(meta.get("_noo_resume_field__taetigkeiten"))
    languages = parse_languages(meta.get("_noo_resume_field_languages_i_speak"))

    lat = meta.get("_resume_address_lat")
    lon = meta.get("_resume_address_lon")
    try:
        lat_f = float(lat) if lat else None
        lon_f = float(lon) if lon else None
    except (TypeError, ValueError):
        lat_f, lon_f = None, None

    pensum = _safe_int(meta.get("_noo_resume_field_job_field_pensum"), 100)
    pensum_from = _safe_int(meta.get("_noo_resume_field_job_field_pensum_from"), 0)
    work_radius_km = _safe_int(meta.get("_noo_resume_field_job_field_arbeitsradius_km"), 50)
    birth_year = _safe_int(meta.get("_noo_resume_field__jahrgang"), 0)
    retired = str(meta.get("_noo_resume_field_already_retired", "") or "").strip() == "1"
    auf_tragsbasis = str(meta.get("_noo_resume_field_job_field_auftragsbasis", "") or "").strip()
    on_contract_basis = "auftrag" in auf_tragsbasis.lower() or bool(auf_tragsbasis)

    skills_text = strip_html(meta.get("_noo_resume_field_job_field_technische_kenntnisse"))
    education_text = strip_html(meta.get("_noo_resume_field_job_field_diplome"))

    seniority_level = infer_seniority(work_experiences)

    return {
        "post_id": raw["post_id"],
        "post_modified": raw.get("post_modified"),
        "work_experiences": work_experiences,
        "languages": languages,
        "location": {
            "lat": lat_f,
            "lon": lon_f,
            "address": (meta.get("_resume_address") or "").strip(),
        },
        "work_radius_km": work_radius_km if work_radius_km > 0 else 50,
        "pensum_desired": pensum,
        "pensum_from": pensum_from,
        "on_contract_basis": on_contract_basis,
        "skills_text": skills_text,
        "education_text": education_text,
        "birth_year": birth_year if birth_year > 0 else None,
        "retired": retired,
        "seniority_level": seniority_level,
        "job_categories_primary": _parse_category_ids(meta.get("_noo_resume_field_job_category_primary")),
        "job_categories_secondary": _parse_category_ids(meta.get("_noo_resume_field_job_category_secondary")),
    }


def _parse_category_ids(meta_value: Any) -> list[str]:
    """Parse serialized array of category IDs."""
    if not meta_value or not isinstance(meta_value, str) or not meta_value.strip().startswith("a:"):
        return []
    try:
        raw = phpserialize.loads(meta_value.encode("utf-8"))
    except Exception:
        return []
    if not isinstance(raw, dict):
        return []
    # Filter and convert keys to ints to avoid bytes/int comparison issues
    int_keys = []
    for k in raw:
        if isinstance(k, int):
            int_keys.append(k)
        elif isinstance(k, bytes):
            try:
                int_keys.append(int(k))
            except (ValueError, TypeError):
                continue
    return [str(raw[k]) for k in sorted(int_keys)]
