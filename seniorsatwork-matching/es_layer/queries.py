"""
Reusable Elasticsearch query templates for matching.
"""
from __future__ import annotations

from typing import Any

from .mappings import CANDIDATES_INDEX, DENSE_DIMS


def build_hard_filters(
    location_lat: float,
    location_lon: float,
    radius_km: int,
    pensum_min: int,
    pensum_max: int,
    required_languages: list[dict[str, str]],
) -> list[dict]:
    """Build filter context list for geo, pensum, languages."""
    filters = [
        {"exists": {"field": "location"}},
        {
            "geo_distance": {
                "distance": f"{radius_km}km",
                "location": {"lat": location_lat, "lon": location_lon},
            }
        },
        {"range": {"pensum_desired": {"gte": pensum_min}}},
        {"range": {"pensum_from": {"lte": pensum_max}}},
    ]
    for lang_req in required_languages or []:
        name = lang_req.get("name") or lang_req.get("lang", "")
        min_level = (lang_req.get("min_level") or "B2").upper()
        degrees = _acceptable_degrees(min_level)
        if name and degrees:
            filters.append({
                "nested": {
                    "path": "languages",
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"languages.lang": name}},
                                {"terms": {"languages.degree": degrees}},
                            ]
                        }
                    },
                }
            })
    return filters


def _acceptable_degrees(min_level: str) -> list[str]:
    """Map CEFR / 'native' to WP degree labels."""
    if min_level in ("C1", "C2", "NATIVE", "NATIVE "):
        return ["Mother tongue", "Fluent"]
    return ["Mother tongue", "Fluent", "Intermediate"]


def build_knn(
    query_vector: list[float],
    filters: list[dict],
    k: int = 100,
    num_candidates: int = 1000,
) -> dict:
    return {
        "field": "aggregated_title_embedding",
        "query_vector": query_vector,
        "k": k,
        "num_candidates": num_candidates,
        "filter": {"bool": {"filter": filters}},
    }


def build_script_score(
    title_vec: list[float],
    industry_vec: list[float],
    skills_vec: list[float],
    edu_vec: list[float],
    expected_seniority_int: int,
    weights: dict[str, float],
) -> dict:
    """Build script_score for 6-dimension scoring."""
    w_t = weights.get("title", 0.40)
    w_i = weights.get("industry", 0.20)
    w_e = weights.get("experience", 0.15)
    w_s = weights.get("skills", 0.10)
    w_sen = weights.get("seniority", 0.08)
    w_edu = weights.get("education", 0.07)
    return {
        "script": {
            "source": """
                double titleSim = doc['aggregated_title_embedding'].size() == 0 ? 1.0 : cosineSimilarity(params.titleVec, 'aggregated_title_embedding') + 1.0;
                double industrySim = doc['aggregated_industry_embedding'].size() == 0 ? 1.0 : cosineSimilarity(params.industryVec, 'aggregated_industry_embedding') + 1.0;
                double skillsSim = doc['skills_embedding'].size() == 0 ? 1.0 : cosineSimilarity(params.skillsVec, 'skills_embedding') + 1.0;
                double eduSim = doc['education_embedding'].size() == 0 ? 1.0 : cosineSimilarity(params.eduVec, 'education_embedding') + 1.0;
                double years = doc['total_weighted_relevant_years'].size() > 0 ? doc['total_weighted_relevant_years'].value : 0.0;
                double expScore = 2.0 / (1.0 + Math.exp(-0.2 * years));
                def candLvl = doc['seniority_level_int'].size() > 0 ? doc['seniority_level_int'].value : params.jobLvl;
                def jobLvl = params.jobLvl;
                double seniorityFit = Math.max(0.5, 1.0 - 0.15 * Math.abs(candLvl - jobLvl));
                return (params.wT * titleSim) + (params.wI * industrySim) + (params.wE * expScore)
                     + (params.wS * skillsSim) + (params.wSen * seniorityFit * 2.0) + (params.wEdu * eduSim);
            """,
            "params": {
                "titleVec": title_vec,
                "industryVec": industry_vec,
                "skillsVec": skills_vec,
                "eduVec": edu_vec,
                "jobLvl": expected_seniority_int,
                "wT": w_t,
                "wI": w_i,
                "wE": w_e,
                "wS": w_s,
                "wSen": w_sen,
                "wEdu": w_edu,
            },
        }
    }
