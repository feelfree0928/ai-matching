"""
Experience recency decay: compute recency_weight and weighted_years per role, aggregate industry text.
"""
from __future__ import annotations

import datetime
from typing import Any

CURRENT_YEAR: int = datetime.date.today().year

RECENCY_FLOOR = 0.38


def recency_weight(end_year: int) -> float:
    """
    Weight by how recent the experience is. Current/ongoing = 1.0; decays over time.
    - 0–5 years ago: linear decay to 0.80
    - 5–15 years ago: ~7%/yr decay to ~0.40
    - 15+ years ago: decay with a floor (RECENCY_FLOOR) so long careers are not over-penalized.
    """
    years_ago = CURRENT_YEAR - end_year
    if years_ago <= 0:
        return 1.0
    if years_ago <= 5:
        return 1.0 - (years_ago * 0.04)
    if years_ago <= 15:
        return 0.80 * (0.93 ** (years_ago - 5))
    return max(RECENCY_FLOOR, 0.40 * (0.85 ** (years_ago - 15)))


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
        years_in_role = float(exp.get("years_in_role", 1) or 1)
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

    # Primary role: prefer an ongoing role (end_year == CURRENT_YEAR) so career changers
    # who recently entered a new field are ranked by their current expertise, not an old
    # long role that dominated weighted_years.
    best_exp = None
    best_w = -1.0
    ongoing_exp = None
    ongoing_w = -1.0

    for exp in experiences:
        w = float(exp.get("weighted_years", 0) or 0)
        if int(exp.get("end_year", 0) or 0) >= CURRENT_YEAR:
            if w > ongoing_w:
                ongoing_w = w
                ongoing_exp = exp
        if w > best_w:
            best_w = w
            best_exp = exp

    # Use ongoing role as primary if it exists; fallback to highest weighted_years
    chosen = ongoing_exp if ongoing_exp is not None else best_exp
    chosen_w = ongoing_w if ongoing_exp is not None else best_w

    # primary_role_title uses raw_title directly (no LLM standardization)
    candidate["primary_role_title"] = (
        (chosen.get("raw_title") or "").strip()
        if chosen else ""
    )
    candidate["primary_role_weighted_years"] = chosen_w if chosen else 0.0
    # Secondary years = everything except the primary role
    candidate["secondary_role_weighted_years"] = max(0.0, total_weighted - chosen_w) if chosen else total_weighted

    return candidate
