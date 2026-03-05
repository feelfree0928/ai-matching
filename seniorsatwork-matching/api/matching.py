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

from api.config import DEFAULT_WEIGHTS, get_max_results, get_max_raw_score, get_min_score_raw, get_weights
from api.models import (
    CandidateLanguage,
    CandidateMatch,
    JobMatchRequest,
    MatchResponse,
    ScoreBreakdown,
    WorkExperienceItem,
)
from api.score_breakdown import compute_breakdown, has_breakdown_data
from embeddings.generator import embed_text
from api.title_match import normalize_job_title_for_matching, score_title_fit_batch

# Default zero vector when job has no text for a dimension
def _zero_vec():
    return [0.0] * 1536


def _job_embeddings(req: JobMatchRequest, client=None):
    from openai import OpenAI
    c = client or OpenAI()
    # Title-only vector for the title dimension: do NOT mix in industry or skills,
    # so we compare job title to candidate title directly (fixes e.g. Night Receptionist
    # ranking below Job Assistant when the job is for night receptionist).
    # Optional: normalize job title with LLM first (handles "Nacht Rezeptionist/in (m/w/d)" etc.).
    raw_title = (req.title or "").strip() or " "
    normalized_title = normalize_job_title_for_matching(raw_title, c)
    title_text = normalized_title if normalized_title else raw_title
    title_vec = embed_text(title_text, c)
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
    industries = list(dict.fromkeys(exp.get("industry") for exp in work_experiences if exp.get("industry")))
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
        std_title = (best.get("standardized_title") or "").strip()
        raw_title = (best.get("raw_title") or "").strip()
        best_title = raw_title
        if std_title and std_title.upper() != "NONE":
            best_title = std_title
        if best_title:
            bullets.append(f"Most relevant role: {best_title}")

    # Experience
    years = float(src.get("total_weighted_relevant_years", 0) or 0)
    if years > 0:
        bullets.append(f"~{years:.0f} yrs weighted experience")

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
    if breakdown.education_score and breakdown.total and breakdown.education_score > (breakdown.total * 0.05):
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

    # No source_excludes: we need embedding and scalar fields in _source to compute real score breakdown.
    _FALLBACK_THRESHOLDS = [1.30, 1.15]

    def _do_search(min_s: float):
        return es.search(
            index=index_name,
            query={
                "script_score": {
                    "query": {"bool": {"filter": filters}},
                    "script": script["script"],
                }
            },
            min_score=min_s,
            size=max_results,
        )

    try:
        resp = _do_search(effective_min_score)
        # If nothing came back, progressively lower the threshold
        if not resp.get("hits", {}).get("hits"):
            for fallback in _FALLBACK_THRESHOLDS:
                if fallback >= effective_min_score:
                    continue
                resp = _do_search(fallback)
                if resp.get("hits", {}).get("hits"):
                    break
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
    # Order and labels for score_calculation display
    PARAM_ORDER = [
        ("title", "Title"),
        ("industry", "Industry"),
        ("experience", "Experience"),
        ("skills", "Skills"),
        ("seniority", "Seniority"),
        ("education", "Education"),
        ("language", "Language"),
    ]
    for i, h in enumerate(hits):
        src = h.get("_source") or {}
        score_raw = float(h.get("_score", 0))
        w = weights
        max_raw = get_max_raw_score(w)
        # Probability-based score: (raw / max_raw) × 100 so it never exceeds 100.
        total_norm = min(100.0, max(0.0, (score_raw / max_raw) * 100.0)) if max_raw > 0 else 0.0

        experience_detail = None
        if has_breakdown_data(src):
            breakdown_vals, experience_detail = compute_breakdown(
                title_vec,
                industry_vec,
                skills_vec,
                edu_vec,
                job_seniority_int,
                src,
                include_experience_detail=True,
            )
            score_calculation = []
            for key, label in PARAM_ORDER:
                value = breakdown_vals.get(key, 0.0)
                weight = w.get(key, DEFAULT_WEIGHTS.get(key, 0))
                contrib_raw = value * weight
                # Contribution as share of 100: (value×weight / max_raw) × 100 so contributions sum to total_norm
                contribution = round((contrib_raw / max_raw * 100.0), 1) if max_raw else 0.0
                score_calculation.append({
                    "parameter": label,
                    "value": round(value, 3),
                    "weight": weight,
                    "contribution": contribution,
                })
            title_score = round(breakdown_vals.get("title", 0), 3)
            industry_score = round(breakdown_vals.get("industry", 0), 3)
            experience_score = round(breakdown_vals.get("experience", 0), 3)
            skills_score = round(breakdown_vals.get("skills", 0), 3)
            seniority_score = round(breakdown_vals.get("seniority", 0), 3)
            education_score = round(breakdown_vals.get("education", 0), 3)
            language_score = round(breakdown_vals.get("language", 0), 3)
            total_formula = f"total = (raw_score / max_raw) × 100 (probability); max_raw = {round(max_raw, 2)}"
            parts = [f"{c['parameter']}: {c['value']} × {c['weight']} = {c['contribution']}" for c in score_calculation]
            score_display = f"Score {round(total_norm, 1)} ({', '.join(parts)})"
        else:
            score_calculation = []
            title_score = round(total_norm * w.get("title", DEFAULT_WEIGHTS["title"]), 1)
            industry_score = round(total_norm * w.get("industry", DEFAULT_WEIGHTS["industry"]), 1)
            experience_score = round(total_norm * w.get("experience", DEFAULT_WEIGHTS["experience"]), 1)
            skills_score = round(total_norm * w.get("skills", DEFAULT_WEIGHTS["skills"]), 1)
            seniority_score = round(total_norm * w.get("seniority", DEFAULT_WEIGHTS["seniority"]), 1)
            education_score = round(total_norm * w.get("education", DEFAULT_WEIGHTS["education"]), 1)
            language_score = round(total_norm * w.get("language", DEFAULT_WEIGHTS["language"]), 1)
            for key, label in PARAM_ORDER:
                weight = w.get(key, DEFAULT_WEIGHTS.get(key, 0))
                contribution = round((total_norm * weight), 1)
                score_calculation.append({
                    "parameter": label,
                    "value": round(total_norm, 3),
                    "weight": weight,
                    "contribution": contribution,
                })
            total_formula = f"total = (raw_score / max_raw) × 100 (probability, breakdown approximate); max_raw = {round(max_raw, 2)}"
            parts = [f"{c['parameter']}: {c['value']} × {c['weight']} = {c['contribution']}" for c in score_calculation]
            score_display = f"Score {round(total_norm, 1)} ({', '.join(parts)})"

        breakdown = ScoreBreakdown(
            total=round(total_norm, 1),
            raw_score=round(score_raw, 4),
            title_score=title_score,
            industry_score=industry_score,
            experience_score=experience_score,
            skills_score=skills_score,
            seniority_score=seniority_score,
            education_score=education_score,
            language_score=language_score,
            total_formula=total_formula,
            score_calculation=score_calculation,
            score_display=score_display,
            experience_detail=experience_detail,
        )
        work_experiences_raw = src.get("work_experiences") or []
        most_relevant = ""
        if work_experiences_raw:
            best = max(work_experiences_raw, key=lambda x: float(x.get("weighted_years", 0) or 0))
            std_title = (best.get("standardized_title") or "").strip()
            raw_title = (best.get("raw_title") or "").strip()
            most_relevant = std_title if std_title and std_title.upper() != "NONE" else raw_title
        industries = list(dict.fromkeys(exp.get("industry") for exp in work_experiences_raw if exp.get("industry")))[:5]
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

    # LLM title-fit: score how well each candidate's main role matches the job title (no re-embedding).
    # Blends into ranking so e.g. Night Receptionist ranks above Job Assistant for a night receptionist job.
    did_llm_resort = False
    if matches:
        try:
            from openai import OpenAI
            llm_client = OpenAI()
            job_title_for_llm = (req.title or "").strip()
            title_fit_scores = score_title_fit_batch(job_title_for_llm, matches, llm_client)
            if title_fit_scores:
                combined = []
                for m in matches:
                    fit = title_fit_scores.get(m.post_id, 5.0)
                    new_breakdown = m.score.model_copy(update={"llm_title_fit": round(fit, 1)})
                    combined.append(m.model_copy(update={"score": new_breakdown}))
                # Sort by 80% ES-derived total + 20% LLM title fit (0-10 -> 0-100 scale)
                combined.sort(
                    key=lambda m: (
                        -(0.8 * (m.score.total or 0) + 0.2 * ((m.score.llm_title_fit or 5) / 10.0) * 100),
                        0 if (m.most_relevant_role or "").strip().upper() == "NONE" else -1,
                    )
                )
                matches = combined
                did_llm_resort = True
        except Exception:
            pass

    # Tie-break when we did not use LLM resort: prefer candidates with a mapped role (most_relevant_role != NONE)
    if not did_llm_resort:
        matches.sort(
            key=lambda m: (
                -(m.score.total or 0),
                0 if (m.most_relevant_role or "").strip().upper() == "NONE" else -1,
            )
        )
    matches = [m.model_copy(update={"rank": i + 1}) for i, m in enumerate(matches)]

    return MatchResponse(
        matches=matches,
        message=None if matches else "No qualified candidates found above threshold.",
        total_above_threshold=total_val,
    )
