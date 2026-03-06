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
    required_available_before: str | None = None,
) -> list[dict]:
    """Build filter context list for geo, pensum, languages, availability."""
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
    if required_available_before:
        filters.append({
            "bool": {
                "should": [
                    {"bool": {"must_not": {"exists": {"field": "available_from"}}}},
                    {"range": {"available_from": {"lte": required_available_before}}},
                ],
                "minimum_should_match": 1,
            }
        })
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
    job_category_labels: list[str] | None = None,
) -> dict:
    """Build script_score for 8-dimension scoring (title, industry, experience, skills, seniority, education, language, category).

    Experience uses a two-part model (Fix D):
      - Primary role (highest weighted_years single role): scored against its OWN title embedding.
        Only the primary role's cosine similarity to the job title gates its experience credit.
      - Secondary roles (all other experience combined): gated by the blended aggregated title
        similarity AND an additional 0.3 discount, so career breadth adds a small signal but
        can never override a primary-role title mismatch.

    The titleRel floor is 0.2 (not 0.5) so unrelated candidates receive ~4% of experience credit
    (titleRelSq = 0.04) instead of 25%.
    """
    w_t = weights.get("title", 0.15)
    w_i = weights.get("industry", 0.12)
    w_e = weights.get("experience", 0.25)
    w_s = weights.get("skills", 0.25)
    w_sen = weights.get("seniority", 0.06)
    w_edu = weights.get("education", 0.05)
    w_lang = weights.get("language", 0.02)
    w_cat = weights.get("category", 0.10)
    job_cats = job_category_labels or []
    return {
        "script": {
            "source": """
                double titleSim = doc['primary_role_title_embedding'].size() > 0
                    ? cosineSimilarity(params.titleVec, 'primary_role_title_embedding') + 1.0
                    : doc['aggregated_title_embedding'].size() == 0 ? 1.0
                      : cosineSimilarity(params.titleVec, 'aggregated_title_embedding') + 1.0;
                double industrySim = doc['aggregated_industry_embedding'].size() == 0 ? 1.0
                    : cosineSimilarity(params.industryVec, 'aggregated_industry_embedding') + 1.0;
                double skillsSim = doc['skills_embedding'].size() == 0 ? 1.0
                    : cosineSimilarity(params.skillsVec, 'skills_embedding') + 1.0;
                double eduSim = doc['education_embedding'].size() == 0 ? 1.0
                    : cosineSimilarity(params.eduVec, 'education_embedding') + 1.0;

                double primYears = doc['primary_role_weighted_years'].size() > 0
                    ? doc['primary_role_weighted_years'].value : 0.0;
                double secYears = doc['secondary_role_weighted_years'].size() > 0
                    ? doc['secondary_role_weighted_years'].value : 0.0;
                double totalYears = doc['total_weighted_relevant_years'].size() > 0
                    ? doc['total_weighted_relevant_years'].value : (primYears + secYears);

                double primTitleSim = doc['primary_role_title_embedding'].size() == 0 ? titleSim
                    : cosineSimilarity(params.titleVec, 'primary_role_title_embedding') + 1.0;
                double primRel    = Math.max(0.2, primTitleSim - 1.0);
                double primRelSq  = primRel * primRel;
                double yearsCap   = Math.min(1.0, primYears / 3.0);
                def primTitle = doc['primary_role_title'].size() > 0 ? doc['primary_role_title'].value : '';
                double nonePenalty = (primTitle.toString() == 'NONE') ? 0.10 : 1.0;
                double expPrimary = (2.0 / (1.0 + Math.exp(-0.25 * primYears))) * primRelSq * yearsCap * nonePenalty;

                double aggRel    = Math.max(0.2, titleSim - 1.0);
                double aggRelSq  = aggRel * aggRel;
                double expSecondary = (2.0 / (1.0 + Math.exp(-0.20 * secYears))) * aggRelSq * 0.30;

                double expTotal = (totalYears > 0 && aggRel > 0)
                    ? (2.0 / (1.0 + Math.exp(-0.15 * totalYears))) * aggRel * 0.25 : 0.0;
                double expScore = expPrimary + expSecondary + expTotal;

                def candLvl = doc['seniority_level_int'].size() > 0 ? doc['seniority_level_int'].value : params.jobLvl;
                def jobLvl = params.jobLvl;
                double seniorityFit = Math.max(0.5, 1.0 - 0.15 * Math.abs(candLvl - jobLvl));
                double langLvl = doc['language_level_max'].size() > 0 ? doc['language_level_max'].value : 0.0;
                double langScore = 1.0 + langLvl / 7.0;

                double catScore = 1.0;
                if (params.jobCategoryLabels != null && params.jobCategoryLabels.length > 0) {
                    boolean hasMatch = false;
                    for (int j = 0; j < params.jobCategoryLabels.length; j++) {
                        String jc = params.jobCategoryLabels[j];
                        for (int i = 0; i < doc['job_category_labels'].size(); i++) {
                            if (doc['job_category_labels'][i].toString().equals(jc)) {
                                hasMatch = true;
                                break;
                            }
                        }
                        if (hasMatch) break;
                    }
                    catScore = hasMatch ? 2.0 : 0.5;
                }

                return (params.wT * titleSim) + (params.wI * industrySim) + (params.wE * expScore)
                     + (params.wS * skillsSim) + (params.wSen * seniorityFit * 2.0)
                     + (params.wEdu * eduSim) + (params.wLang * langScore) + (params.wCat * catScore);
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
                "wLang": w_lang,
                "wCat": w_cat,
                "jobCategoryLabels": job_cats,
            },
        }
    }

