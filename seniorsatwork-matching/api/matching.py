"""
Matching: build hard filters, script_score query, run search, format response with score breakdown.
"""
from __future__ import annotations

from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import BadRequestError
from es_layer.indexer import get_es_client
from es_layer.mappings import CANDIDATES_INDEX, SENIORITY_TO_INT
from es_layer.queries import build_hard_filters, build_script_score

from api.config import get_max_results, get_min_score_raw, get_weights
from api.models import CandidateMatch, JobMatchRequest, MatchResponse, ScoreBreakdown
from embeddings.generator import embed_text

# Default zero vector when job has no text for a dimension
def _zero_vec():
    return [0.0] * 1536


def _job_embeddings(req: JobMatchRequest, client=None):
    from openai import OpenAI
    c = client or OpenAI()
    title_vec = embed_text(req.title or " ", c) if req.title else _zero_vec()
    industry_vec = embed_text(req.industry or " ", c) if req.industry else _zero_vec()
    skills_vec = embed_text(req.required_skills or " ", c) if req.required_skills else _zero_vec()
    edu_vec = embed_text(req.required_education or " ", c) if req.required_education else _zero_vec()
    return title_vec, industry_vec, skills_vec, edu_vec


def run_match(
    req: JobMatchRequest,
    es: Elasticsearch | None = None,
    min_score_override: float | None = None,
    max_results_override: int | None = None,
    index: str | None = None,
) -> MatchResponse:
    """
    Run matching: hard filters + script_score, return ranked shortlist with score breakdown.
    index: optional index name (default CANDIDATES_INDEX); used for golden test.
    """
    es = es or get_es_client()
    index_name = index or CANDIDATES_INDEX
    min_score = min_score_override if min_score_override is not None else get_min_score_raw()
    max_results = max_results_override if max_results_override is not None else get_max_results()
    weights = get_weights()

    title_vec, industry_vec, skills_vec, edu_vec = _job_embeddings(req)
    job_seniority_int = SENIORITY_TO_INT.get(req.expected_seniority_level.strip().lower(), 2)

    filters = build_hard_filters(
        location_lat=req.location_lat,
        location_lon=req.location_lon,
        radius_km=req.radius_km,
        pensum_min=req.pensum_min,
        pensum_max=req.pensum_max,
        required_languages=[{"name": l.name, "min_level": l.min_level} for l in req.required_languages],
        required_available_before=req.required_available_before,
    )

    script = build_script_score(
        title_vec=title_vec,
        industry_vec=industry_vec,
        skills_vec=skills_vec,
        edu_vec=edu_vec,
        expected_seniority_int=job_seniority_int,
        weights=weights,
    )

    effective_min_score = req.min_score if req.min_score is not None else min_score

    try:
        resp = es.search(
            index=index_name,
            query={
                "script_score": {
                    "query": {"bool": {"filter": filters}},
                    "script": script["script"],
                }
            },
            min_score=effective_min_score,
            size=max_results,
        )
    except BadRequestError as e:
        detail = str(e)
        try:
            body = getattr(e, "body", None) or getattr(e, "info", None) or {}
            if isinstance(body, dict) and "error" in body:
                err = body["error"]
                root = (err.get("root_cause") or [{}])[0]
                reason = root.get("reason") or err.get("reason") or detail
                script_stack = root.get("script_stack")
                if script_stack:
                    detail = f"{reason} (script: {script_stack})"
                else:
                    detail = reason
        except Exception:
            pass
        return MatchResponse(
            matches=[],
            message=f"Search failed: {detail}",
            total_above_threshold=0,
        )
    except Exception as e:
        return MatchResponse(
            matches=[],
            message=f"Search failed: {e}",
            total_above_threshold=0,
        )

    hits = resp.get("hits", {}).get("hits", [])
    total = resp.get("hits", {}).get("total", {})
    if isinstance(total, dict):
        total_val = total.get("value", 0)
    else:
        total_val = total

    matches = []
    for h in hits:
        src = h.get("_source") or {}
        score_raw = float(h.get("_score", 0))
        # Normalize to 0-100 for display (script score is roughly 0..2)
        total_norm = min(100.0, max(0.0, (score_raw / 2.0) * 100.0))
        # Approximate breakdown by weight (we don't have per-dim from ES)
        w = weights
        breakdown = ScoreBreakdown(
            total=round(total_norm, 1),
            title_score=round(total_norm * w.get("title", 0.38), 1),
            industry_score=round(total_norm * w.get("industry", 0.19), 1),
            experience_score=round(total_norm * w.get("experience", 0.14), 1),
            skills_score=round(total_norm * w.get("skills", 0.1), 1),
            seniority_score=round(total_norm * w.get("seniority", 0.07), 1),
            education_score=round(total_norm * w.get("education", 0.07), 1),
            language_score=round(total_norm * w.get("language", 0.05), 1),
        )
        work_experiences = src.get("work_experiences") or []
        most_relevant = ""
        if work_experiences:
            best = max(work_experiences, key=lambda x: float(x.get("weighted_years", 0) or 0))
            most_relevant = best.get("standardized_title") or best.get("raw_title") or ""
        industries = list({exp.get("industry") for exp in work_experiences if exp.get("industry")})[:5]
        addr = src.get("address") or ""
        matches.append(CandidateMatch(
            post_id=src.get("post_id", 0),
            score=breakdown,
            most_relevant_role=most_relevant,
            total_relevant_years=float(src.get("total_weighted_relevant_years", 0) or 0),
            seniority_level=src.get("seniority_level", ""),
            location=addr,
            pensum_desired=int(src.get("pensum_desired", 100) or 100),
            top_industries=industries,
        ))

    return MatchResponse(
        matches=matches,
        message=None if matches else "No qualified candidates found above threshold.",
        total_above_threshold=total_val,
    )
