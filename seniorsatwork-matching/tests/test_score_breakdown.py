"""
Unit tests for score breakdown (experience formula: primary + secondary + total).
"""
import pytest

from api.score_breakdown import compute_breakdown

# Dense dims for dummy vectors (match ES embedding size)
DIMS = 1536


def _dummy_vec(value: float = 1.0) -> list[float]:
    return [value] * DIMS


def test_more_total_weighted_experience_scores_higher():
    """
    With same title relevance, a candidate with 20 years total weighted experience
    should get a higher experience dimension score than one with 10 years,
    so that twice as much experience does not score lower.
    """
    job_seniority_int = 2
    title_vec = _dummy_vec(1.0)
    industry_vec = _dummy_vec(0.0)
    skills_vec = _dummy_vec(0.0)
    edu_vec = _dummy_vec(0.0)
    # Same title relevance: no embedding so title_sim = 1.0, prim_rel = agg_rel = 0.2
    doc_10 = {
        "primary_role_weighted_years": 10.0,
        "secondary_role_weighted_years": 0.0,
        "total_weighted_relevant_years": 10.0,
        "primary_role_title": "Accountant",
        "aggregated_title_embedding": None,
    }
    doc_20 = {
        "primary_role_weighted_years": 10.0,
        "secondary_role_weighted_years": 10.0,
        "total_weighted_relevant_years": 20.0,
        "primary_role_title": "Accountant",
        "aggregated_title_embedding": None,
    }
    breakdown_10, _ = compute_breakdown(
        title_vec, industry_vec, skills_vec, edu_vec, job_seniority_int, doc_10
    )
    breakdown_20, _ = compute_breakdown(
        title_vec, industry_vec, skills_vec, edu_vec, job_seniority_int, doc_20
    )
    assert breakdown_20["experience"] > breakdown_10["experience"], (
        "Candidate with 20 years total weighted experience should score higher on experience "
        "than candidate with 10 years (same title relevance)."
    )
