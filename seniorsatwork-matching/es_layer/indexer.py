"""
Bulk index helpers for candidates and job postings.
"""
from __future__ import annotations

import warnings
from typing import Any, Iterator

from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import NotFoundError

from .mappings import (
    CANDIDATES_INDEX,
    JOBS_INDEX,
    CANDIDATES_MAPPING,
    JOBS_MAPPING,
    DENSE_DIMS,
    SENIORITY_TO_INT,
)

# CEFR / WP degree to integer for language_level_max (1-7, 0 = none)
LANGUAGE_DEGREE_TO_INT = {
    "A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6,
    "Mother tongue": 7, "Fluent": 6, "Intermediate": 4,
}


def _language_level_max(languages: list[dict[str, Any]]) -> int:
    """Highest language level across all languages (for ranking)."""
    if not languages:
        return 0
    levels = []
    for lang in languages:
        degree = (lang.get("degree") or "").strip()
        if degree:
            levels.append(LANGUAGE_DEGREE_TO_INT.get(degree, 0))
    return max(levels, default=0)


def get_es_client(url: str | None = None) -> Elasticsearch:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    u = url or os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    kwargs = {"request_timeout": 300}
    user = os.getenv("ELASTICSEARCH_USER")
    password = os.getenv("ELASTICSEARCH_PASSWORD")
    if user and password:
        kwargs["basic_auth"] = (user, password)
    # Allow skipping SSL verification for ES 8.x default self-signed cert (dev only)
    verify = os.getenv("ELASTICSEARCH_VERIFY_CERTS", "true").strip().lower()
    if verify in ("false", "0", "no"):
        kwargs["verify_certs"] = False
        # Suppress TLS warnings when user explicitly disabled verification
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass
        warnings.filterwarnings("ignore", message=".*verify_certs.*")
        warnings.filterwarnings("ignore", message=".*Unverified HTTPS request.*")
    return Elasticsearch(u, **kwargs)


def ensure_indices(es: Elasticsearch) -> None:
    """Create candidates and job_postings indices if they do not exist.
    Uses GET (indices.get) instead of HEAD (indices.exists) to avoid 400 with no body on some ES 8.x setups.
    """
    for index_name, mapping in [
        (CANDIDATES_INDEX, CANDIDATES_MAPPING),
        (JOBS_INDEX, JOBS_MAPPING),
    ]:
        try:
            es.indices.get(index=index_name)
        except NotFoundError:
            es.indices.create(index=index_name, mappings=mapping)


def _sanitize_post_modified(val: Any) -> str | None:
    """Return ES-parseable date string or None. WordPress uses 0000-00-00 00:00:00 for invalid dates."""
    if val is None:
        return None
    if hasattr(val, "isoformat"):  # datetime
        return val.isoformat()
    s = str(val).strip()
    if not s or s.startswith("0000-00-00"):
        return None
    # ES accepts ISO and common formats; pass through valid-looking dates
    return s.replace(" ", "T", 1) if " " in s and len(s) > 10 else s


def _ensure_nonzero_vector(vec: list[float] | None, dims: int = DENSE_DIMS) -> list[float] | None:
    """Return vec if it has non-zero magnitude; else a unit vector so cosine similarity works."""
    if not vec or len(vec) != dims:
        return None
    sq = sum(x * x for x in vec)
    if sq > 0:
        return vec
    # Zero vector: use unit vector along first dimension
    unit = [0.0] * dims
    unit[0] = 1.0
    return unit


def _candidate_doc(c: dict[str, Any]) -> dict[str, Any]:
    """Build Elasticsearch document for one candidate."""
    loc = c.get("location") or {}
    lat, lon = loc.get("lat"), loc.get("lon")
    work_experiences = []
    for exp in c.get("work_experiences") or []:
        work_experiences.append({
            "raw_title": exp.get("raw_title", ""),
            "standardized_title": exp.get("standardized_title", "NONE"),
            "industry": exp.get("industry", ""),
            "start_year": exp.get("start_year"),
            "end_year": exp.get("end_year"),
            "years_in_role": exp.get("years_in_role"),
            "recency_weight": exp.get("recency_weight"),
            "weighted_years": exp.get("weighted_years"),
            "description": (exp.get("description") or "")[:10000],
        })

    doc = {
        "post_id": c["post_id"],
        "post_modified": _sanitize_post_modified(c.get("post_modified")),
        "location": {"lat": lat, "lon": lon} if lat is not None and lon is not None else None,
        "address": (loc.get("address") or "").strip() or None,
        "work_radius_km": c.get("work_radius_km", 50),
        "pensum_desired": c.get("pensum_desired", 100),
        "pensum_from": c.get("pensum_from", 0),
        "available_from": c.get("available_from"),
        "on_contract_basis": c.get("on_contract_basis", False),
        "languages": c.get("languages") or [],
        "seniority_level": c.get("seniority_level", "mid"),
        "seniority_level_int": SENIORITY_TO_INT.get(c.get("seniority_level", "mid"), 1),
        "language_level_max": _language_level_max(c.get("languages") or []),
        "work_experiences": work_experiences,
        "aggregated_title_embedding": _ensure_nonzero_vector(c.get("aggregated_title_embedding")),
        "aggregated_industry_embedding": _ensure_nonzero_vector(c.get("aggregated_industry_embedding")),
        "total_weighted_relevant_years": c.get("total_weighted_relevant_years", 0.0),
        "skills_embedding": _ensure_nonzero_vector(c.get("skills_embedding")),
        "education_embedding": _ensure_nonzero_vector(c.get("education_embedding")),
        "skills_text": (c.get("skills_text") or "")[:32000],
        "education_text": (c.get("education_text") or "")[:32000],
        "birth_year": c.get("birth_year"),
        "retired": c.get("retired", False),
        "job_categories_primary": c.get("job_categories_primary") or [],
        "job_categories_secondary": c.get("job_categories_secondary") or [],
    }
    return doc


def bulk_index_candidates(
    es: Elasticsearch,
    candidates: list[dict[str, Any]],
    chunk_size: int = 50,
    *,
    errors: list[tuple[str, dict]] | None = None,
    request_timeout: int | None = None,
) -> tuple[int, int]:
    """Index candidates using streaming_bulk; return (success_count, error_count).

    Uses streaming_bulk so each chunk's response is processed immediately — no
    silent blocking while waiting for all 26k docs to accumulate.
    If *errors* list is provided, (doc_id, error_info) tuples are appended for
    every failed document.
    """
    def gen() -> Iterator[dict]:
        for c in candidates:
            doc = _candidate_doc(c)
            if doc.get("location") is None:
                continue
            if doc.get("aggregated_title_embedding") is None:
                continue
            yield {
                "_index": CANDIDATES_INDEX,
                "_id": str(c["post_id"]),
                "_source": doc,
            }

    streaming_kw: dict[str, Any] = {
        "chunk_size": chunk_size,
        "raise_on_error": False,
        "raise_on_exception": False,
    }
    if request_timeout is not None:
        streaming_kw["request_timeout"] = request_timeout

    success_count = 0
    fail_count = 0
    for ok, item in helpers.streaming_bulk(es, gen(), **streaming_kw):
        if ok:
            success_count += 1
        else:
            fail_count += 1
            if errors is not None:
                op = item.get("index", item)
                doc_id = op.get("_id", "?")
                err = op.get("error", op)
                errors.append((str(doc_id), err))
    return success_count, fail_count


def bulk_index_jobs(
    es: Elasticsearch,
    jobs: list[dict[str, Any]],
    chunk_size: int = 100,
    *,
    request_timeout: int | None = None,
) -> tuple[int, int]:
    """Index job postings; return (success_count, error_count)."""
    def gen() -> Iterator[dict]:
        for j in jobs:
            doc = _job_doc(j)
            yield {
                "_index": JOBS_INDEX,
                "_id": str(j.get("post_id", "")),
                "_source": doc,
            }
    bulk_kw: dict[str, Any] = {
        "chunk_size": chunk_size,
        "raise_on_error": False,
        "stats_only": True,
    }
    if request_timeout is not None:
        bulk_kw["request_timeout"] = request_timeout
    success, failed = helpers.bulk(es, gen(), **bulk_kw)
    return success, failed


def _job_doc(j: dict[str, Any]) -> dict[str, Any]:
    loc = j.get("location") or {}
    lat, lon = loc.get("lat"), loc.get("lon")
    return {
        "post_id": j.get("post_id"),
        "post_modified": _sanitize_post_modified(j.get("post_modified")),
        "title": j.get("title", ""),
        "standardized_title": j.get("standardized_title", ""),
        "title_embedding": j.get("title_embedding"),
        "industry": j.get("industry", ""),
        "industry_embedding": j.get("industry_embedding"),
        "required_skills_text": j.get("required_skills_text", ""),
        "skills_embedding": j.get("skills_embedding"),
        "required_education_text": j.get("required_education_text", ""),
        "education_embedding": j.get("education_embedding"),
        "expected_seniority_level": j.get("expected_seniority_level", "senior"),
        "expected_seniority_level_int": SENIORITY_TO_INT.get(j.get("expected_seniority_level", "senior"), 2),
        "location": {"lat": lat, "lon": lon} if lat is not None and lon is not None else None,
        "radius_km": j.get("radius_km", 50),
        "pensum_min": j.get("pensum_min", 0),
        "pensum_max": j.get("pensum_max", 100),
        "required_languages": j.get("required_languages") or [],
    }
