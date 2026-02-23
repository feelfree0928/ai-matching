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
from api.models import (
    CandidateLanguage,
    CandidateMatch,
    JobMatchRequest,
    MatchResponse,
    ScoreBreakdown,
    WorkExperienceItem,
)
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


def _build_rank_explanation(
    src: dict[str, Any],
    req: JobMatchRequest,
    rank: int,
    breakdown: ScoreBreakdown,
    weights: dict[str, float],
) -> list[str]:
    """Build rule-based 'Why ranked #N' bullet points from candidate _source and job request."""
    bullets: list[str] = []
    job_title_words = set((req.title or "").lower().split())
    work_experiences = src.get("work_experiences") or []
    industries = list({exp.get("industry") for exp in work_experiences if exp.get("industry")})
    job_industry_words = set((req.industry or "").lower().split())
    cand_seniority = (src.get("seniority_level") or "").strip().lower()
    job_seniority = (req.expected_seniority_level or "senior").strip().lower()

    # Title: job title word overlap with work experience titles
    for exp in work_experiences:
        raw = (exp.get("raw_title") or "").strip()
        if not raw:
            continue
        title_words = set(raw.lower().split())
        overlap = job_title_words & title_words
        if len(overlap) >= 2:
            bullets.append(f"Job title match: {raw}")
            break
    if not any("Job title match" in b for b in bullets) and work_experiences:
        best = max(work_experiences, key=lambda x: float(x.get("weighted_years", 0) or 0))
        best_title = best.get("standardized_title") or best.get("raw_title") or ""
        if best_title:
            bullets.append(f"Most relevant role: {best_title}")

    # Experience
    years = float(src.get("total_weighted_relevant_years", 0) or 0)
    if years > 0:
        bullets.append(f"{years:.0f} years of relevant experience")

    # Industry
    for ind in industries[:3]:
        if ind and job_industry_words and any(w in (ind or "").lower() for w in job_industry_words):
            bullets.append(f"Industry match: {ind}")
            break
    if not any("Industry" in b for b in bullets) and industries:
        bullets.append(f"Industry: {', '.join(industries[:3])}")

    # Seniority
    if cand_seniority == job_seniority:
        bullets.append(f"Ideal seniority level ({job_seniority})")
    elif cand_seniority:
        bullets.append(f"Seniority: {cand_seniority} (expected: {job_seniority})")

    # Skills: keyword overlap
    skills_text = (src.get("skills_text") or "").lower()
    required_skills = (req.required_skills or "").lower()
    if skills_text and required_skills:
        req_tokens = set(w.strip() for w in required_skills.replace(",", " ").split() if len(w.strip()) > 1)
        found = [t for t in req_tokens if t in skills_text][:5]
        if found:
            bullets.append("Skills: " + ", ".join(found))

    # Education
    if breakdown.education_score and breakdown.education_score > (breakdown.total * 0.05):
        bullets.append("Relevant certifications/education")

    return bullets[:8]


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
    max_results = max_results_override if max_results_override is not None else (req.max_results or get_max_results())
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

    source_excludes = [
        "aggregated_title_embedding",
        "aggregated_industry_embedding",
        "skills_embedding",
        "education_embedding",
    ]
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
            source_excludes=source_excludes,
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
    for i, h in enumerate(hits):
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
        work_experiences_raw = src.get("work_experiences") or []
        most_relevant = ""
        if work_experiences_raw:
            best = max(work_experiences_raw, key=lambda x: float(x.get("weighted_years", 0) or 0))
            most_relevant = best.get("standardized_title") or best.get("raw_title") or ""
        industries = list({exp.get("industry") for exp in work_experiences_raw if exp.get("industry")})[:5]
        addr = src.get("address") or ""

        work_experiences = [
            WorkExperienceItem(
                raw_title=exp.get("raw_title", ""),
                standardized_title=exp.get("standardized_title", ""),
                industry=exp.get("industry", ""),
                start_year=exp.get("start_year"),
                end_year=exp.get("end_year"),
                years_in_role=exp.get("years_in_role"),
                weighted_years=exp.get("weighted_years"),
            )
            for exp in work_experiences_raw
        ]
        languages = [
            CandidateLanguage(lang=lg.get("lang", ""), degree=lg.get("degree", ""))
            for lg in (src.get("languages") or [])
        ]
        rank_explanation = _build_rank_explanation(src, req, i + 1, breakdown, weights)

        job_cats_primary = src.get("job_categories_primary")
        job_cats_secondary = src.get("job_categories_secondary")
        if isinstance(job_cats_primary, str):
            job_cats_primary = [job_cats_primary] if job_cats_primary else []
        if isinstance(job_cats_secondary, str):
            job_cats_secondary = [job_cats_secondary] if job_cats_secondary else []
        if not isinstance(job_cats_primary, list):
            job_cats_primary = []
        if not isinstance(job_cats_secondary, list):
            job_cats_secondary = []

        most_exp_industries = src.get("most_experience_industries")
        if isinstance(most_exp_industries, str):
            most_exp_industries = [most_exp_industries] if most_exp_industries else []
        if not isinstance(most_exp_industries, list):
            most_exp_industries = []

        matches.append(CandidateMatch(
            post_id=src.get("post_id", 0),
            score=breakdown,
            rank=i + 1,
            rank_explanation=rank_explanation,
            # identity & contact
            candidate_name=(src.get("candidate_name") or "").strip(),
            phone=(src.get("phone") or "").strip(),
            gender=(src.get("gender") or "").strip(),
            linkedin_url=(src.get("linkedin_url") or "").strip(),
            website_url=(src.get("website_url") or "").strip(),
            cv_file=(src.get("cv_file") or "").strip(),
            # profile text
            short_description=(src.get("short_description") or "").strip(),
            job_expectations=(src.get("job_expectations") or "").strip(),
            highest_degree=(src.get("highest_degree") or "").strip(),
            ai_profile_description=(src.get("ai_profile_description") or "").strip(),
            ai_experience_description=(src.get("ai_experience_description") or "").strip(),
            ai_skills_description=(src.get("ai_skills_description") or "").strip(),
            ai_text_skill_result=(src.get("ai_text_skill_result") or "").strip(),
            # experience & skills
            most_relevant_role=most_relevant,
            total_relevant_years=float(src.get("total_weighted_relevant_years", 0) or 0),
            seniority_level=src.get("seniority_level", ""),
            work_experiences=work_experiences,
            skills_text=(src.get("skills_text") or "").strip(),
            education_text=(src.get("education_text") or "").strip(),
            most_experience_industries=most_exp_industries,
            top_industries=industries,
            # languages
            languages=languages,
            # location
            location=addr,
            zip_code=(src.get("zip_code") or "").strip(),
            work_radius_km=int(src.get("work_radius_km", 50) or 50),
            work_radius_text=(src.get("work_radius_text") or "").strip(),
            # availability & contract
            available_from=src.get("available_from"),
            pensum_desired=int(src.get("pensum_desired", 100) or 100),
            pensum_from=int(src.get("pensum_from", 0) or 0),
            pensum_duration=(src.get("pensum_duration") or "").strip(),
            on_contract_basis=bool(src.get("on_contract_basis", False)),
            voluntary=(src.get("voluntary") or "").strip(),
            # personal
            birth_year=src.get("birth_year"),
            retired=bool(src.get("retired", False)),
            # categories
            job_categories_primary=job_cats_primary,
            job_categories_secondary=job_cats_secondary,
            # profile meta
            profile_status=(src.get("profile_status") or "").strip(),
            registered_at=src.get("registered_at"),
            expires_at=src.get("expires_at"),
            featured=bool(src.get("featured", False)),
            post_date=src.get("post_date"),
        ))

    return MatchResponse(
        matches=matches,
        message=None if matches else "No qualified candidates found above threshold.",
        total_above_threshold=total_val,
    )
