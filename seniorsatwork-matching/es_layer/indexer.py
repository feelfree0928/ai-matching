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


def get_es_client(url: str | None = None) -> Elasticsearch:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    u = url or os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    kwargs = {"request_timeout": 60}
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
        warnings.filterwarnings(
            "ignore",
            message=".*verify_certs.*",
            category=warnings.SecurityWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*Unverified HTTPS request.*",
        )
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
        "post_modified": c.get("post_modified"),
        "location": {"lat": lat, "lon": lon} if lat is not None and lon is not None else None,
        "address": (loc.get("address") or "").strip() or None,
        "work_radius_km": c.get("work_radius_km", 50),
        "pensum_desired": c.get("pensum_desired", 100),
        "pensum_from": c.get("pensum_from", 0),
        "on_contract_basis": c.get("on_contract_basis", False),
        "languages": c.get("languages") or [],
        "seniority_level": c.get("seniority_level", "mid"),
        "seniority_level_int": SENIORITY_TO_INT.get(c.get("seniority_level", "mid"), 1),
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
    chunk_size: int = 200,
    *,
    errors: list[tuple[str, dict]] | None = None,
) -> tuple[int, int]:
    """Index candidates; return (success_count, error_count).
    If errors list is provided, append (doc_id, error_info) for each failed item."""
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
    if errors is not None:
        success, err_list = helpers.bulk(
            es, gen(), chunk_size=chunk_size, raise_on_error=False, stats_only=False
        )
        for item in err_list:
            op = item.get("index", item)
            doc_id = op.get("_id", "?")
            err = op.get("error", op)
            errors.append((str(doc_id), err))
        return success, len(err_list)
    success, failed = helpers.bulk(
        es, gen(), chunk_size=chunk_size, raise_on_error=False, stats_only=True
    )
    return success, failed


def bulk_index_jobs(
    es: Elasticsearch,
    jobs: list[dict[str, Any]],
    chunk_size: int = 100,
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
    success, failed = helpers.bulk(
        es, gen(), chunk_size=chunk_size, raise_on_error=False, stats_only=True
    )
    return success, failed


def _job_doc(j: dict[str, Any]) -> dict[str, Any]:
    loc = j.get("location") or {}
    lat, lon = loc.get("lat"), loc.get("lon")
    return {
        "post_id": j.get("post_id"),
        "post_modified": j.get("post_modified"),
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
