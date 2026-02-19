"""
Experience recency decay: compute recency_weight and weighted_years per role, aggregate industry text.
"""
from __future__ import annotations

from typing import Any

CURRENT_YEAR = 2026


def recency_weight(end_year: int) -> float:
    """
    Weight by how recent the experience is. Current/ongoing = 1.0; decays over time.
    - 0–5 years ago: linear decay to 0.80
    - 5–15 years ago: ~7%/yr decay to ~0.40
    - 15+ years ago: steep decay toward zero
    """
    years_ago = CURRENT_YEAR - end_year
    if years_ago <= 0:
        return 1.0
    if years_ago <= 5:
        return 1.0 - (years_ago * 0.04)
    if years_ago <= 15:
        return 0.80 * (0.93 ** (years_ago - 5))
    return 0.40 * (0.85 ** (years_ago - 15))


def apply_experience_scoring(candidate: dict[str, Any]) -> dict[str, Any]:
    """
    Mutate candidate['work_experiences'] in place: add recency_weight and weighted_years per entry.
    Also set candidate['total_weighted_relevant_years'] and candidate['aggregated_industry_parts']
    (list of (industry, weight) for building industry embedding text).
    """
    experiences = candidate.get("work_experiences") or []
    total_weighted = 0.0
    industry_parts: list[tuple[str, float]] = []

    for exp in experiences:
        end_year = int(exp.get("end_year", CURRENT_YEAR))
        years_in_role = max(1, int(exp.get("years_in_role", 1)))
        rw = recency_weight(end_year)
        weighted_years = years_in_role * rw
        exp["recency_weight"] = rw
        exp["weighted_years"] = weighted_years
        total_weighted += weighted_years
        industry = (exp.get("industry") or "").strip()
        if industry:
            industry_parts.append((industry, weighted_years))

    candidate["total_weighted_relevant_years"] = total_weighted
    candidate["aggregated_industry_parts"] = industry_parts
    return candidate
