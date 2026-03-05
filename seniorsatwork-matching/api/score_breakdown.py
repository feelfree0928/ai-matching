"""
Compute per-dimension score breakdown in Python (mirrors Painless script in es_layer/queries.py).
Used to show real dimension values (title similarity, experience formula, etc.) in the API
so clients can see why each profile got its total score.
"""
from __future__ import annotations

import math
from typing import Any


def _cosine_sim_plus_one(query_vec: list[float], doc_vec: list[float] | None) -> float:
    """Cosine similarity + 1, range [0, 2]. Returns 1.0 if either vector is missing or empty."""
    if not query_vec or not doc_vec or len(query_vec) != len(doc_vec):
        return 1.0
    try:
        dot = sum(a * b for a, b in zip(query_vec, doc_vec))
        norm_q = math.sqrt(sum(x * x for x in query_vec))
        norm_d = math.sqrt(sum(x * x for x in doc_vec))
        if norm_q <= 0 or norm_d <= 0:
            return 1.0
        cos_sim = dot / (norm_q * norm_d)
        return cos_sim + 1.0
    except (TypeError, ValueError):
        return 1.0


def _get_float(doc: dict[str, Any], key: str) -> float:
    val = doc.get(key)
    if val is None:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _get_str(doc: dict[str, Any], key: str) -> str:
    val = doc.get(key)
    if val is None:
        return ""
    return str(val).strip()


def _get_int(doc: dict[str, Any], key: str, default: int = 0) -> int:
    val = doc.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def compute_breakdown(
    title_vec: list[float],
    industry_vec: list[float],
    skills_vec: list[float],
    edu_vec: list[float],
    job_seniority_int: int,
    doc_source: dict[str, Any],
    *,
    include_experience_detail: bool = True,
) -> tuple[dict[str, float], dict[str, Any] | None]:
    """
    Replicate the Painless script scoring logic to get real per-dimension values.

    Returns:
        (breakdown, experience_detail): breakdown has keys title, industry, experience,
        skills, seniority, education, language with the same values used in the ES script.
        experience_detail is optional dict for UI (primYears, secYears, primRel, etc.).
    """
    # Similarity dimensions (cosine + 1, range [0, 2])
    title_sim = _cosine_sim_plus_one(
        title_vec,
        doc_source.get("aggregated_title_embedding"),
    )
    industry_sim = _cosine_sim_plus_one(
        industry_vec,
        doc_source.get("aggregated_industry_embedding"),
    )
    skills_sim = _cosine_sim_plus_one(
        skills_vec,
        doc_source.get("skills_embedding"),
    )
    edu_sim = _cosine_sim_plus_one(
        edu_vec,
        doc_source.get("education_embedding"),
    )

    # Experience: primary + secondary + total (same constants as Painless)
    prim_years = _get_float(doc_source, "primary_role_weighted_years")
    sec_years = _get_float(doc_source, "secondary_role_weighted_years")
    total_years = _get_float(doc_source, "total_weighted_relevant_years")
    if total_years <= 0 and (prim_years or sec_years):
        total_years = prim_years + sec_years
    prim_title_str = _get_str(doc_source, "primary_role_title")
    prim_title_embedding = doc_source.get("primary_role_title_embedding")

    prim_title_sim = title_sim
    if prim_title_embedding and isinstance(prim_title_embedding, list) and len(prim_title_embedding) == len(title_vec):
        prim_title_sim = _cosine_sim_plus_one(title_vec, prim_title_embedding)

    prim_rel = max(0.2, prim_title_sim - 1.0)
    prim_rel_sq = prim_rel * prim_rel
    years_cap = min(1.0, prim_years / 3.0) if prim_years else 0.0
    none_penalty = 0.10 if prim_title_str == "NONE" else 1.0

    sigmoid_prim = (2.0 / (1.0 + math.exp(-0.25 * prim_years))) if prim_years else 0.0
    exp_primary = sigmoid_prim * prim_rel_sq * years_cap * none_penalty

    agg_rel = max(0.2, title_sim - 1.0)
    agg_rel_sq = agg_rel * agg_rel
    sigmoid_sec = (2.0 / (1.0 + math.exp(-0.20 * sec_years))) if sec_years else 0.0
    exp_secondary = sigmoid_sec * agg_rel_sq * 0.30

    exp_total = 0.0
    if total_years > 0 and agg_rel > 0:
        sigmoid_total = 2.0 / (1.0 + math.exp(-0.15 * total_years))
        exp_total = sigmoid_total * agg_rel * 0.25
    exp_score = exp_primary + exp_secondary + exp_total

    # Seniority: max(0.5, 1 - 0.15*|cand - job|), then *2 in formula
    cand_lvl = _get_int(doc_source, "seniority_level_int", job_seniority_int)
    seniority_fit = max(0.5, 1.0 - 0.15 * abs(cand_lvl - job_seniority_int))
    seniority_value = seniority_fit * 2.0

    # Language
    lang_lvl = _get_float(doc_source, "language_level_max")
    lang_score = 1.0 + lang_lvl / 7.0

    breakdown = {
        "title": title_sim,
        "industry": industry_sim,
        "experience": exp_score,
        "skills": skills_sim,
        "seniority": seniority_value,
        "education": edu_sim,
        "language": lang_score,
    }

    experience_detail = None
    if include_experience_detail:
        experience_detail = {
            "primary_years": round(prim_years, 2),
            "secondary_years": round(sec_years, 2),
            "total_years": round(total_years, 2),
            "primary_relevance": round(prim_rel, 3),
            "exp_primary": round(exp_primary, 4),
            "exp_secondary": round(exp_secondary, 4),
            "exp_total": round(exp_total, 4),
            "none_penalty": none_penalty,
        }

    return breakdown, experience_detail


def has_breakdown_data(doc_source: dict[str, Any]) -> bool:
    """Return True if doc has enough fields to compute a real breakdown (at least title vector)."""
    return bool(doc_source.get("aggregated_title_embedding"))
