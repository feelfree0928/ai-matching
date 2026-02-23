"""
Golden dataset test: RE-ACC-001 job vs 10 candidates. Asserts C04 > C01 > C09 and C07 excluded.
Requires Elasticsearch and OpenAI API key; skipped if unavailable.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

GOLDEN_DIR = os.path.join(os.path.dirname(__file__))
JOB_PATH = os.path.join(GOLDEN_DIR, "RE-ACC-001_job.json")
CANDIDATES_PATH = os.path.join(GOLDEN_DIR, "RE-ACC-001_candidates.json")
TEST_INDEX = "candidates_golden_test"

# Expected ranking (post_id): 1st C04=10004, 2nd C01=10001, 3rd C09=10009; C07=10007 must be excluded
EXPECTED_ORDER_POST_IDS = [10004, 10001, 10009, 10002, 10006, 10003, 10008, 10005, 10010]
EXCLUDED_POST_ID = 10007  # C07, score 41


@pytest.fixture(scope="module")
def es_client():
    try:
        from es_layer.indexer import get_es_client
        es = get_es_client()
        es.info()
        return es
    except Exception:
        pytest.skip("Elasticsearch not available")


@pytest.fixture(scope="module")
def golden_index(es_client):
    from es_layer.mappings import CANDIDATES_MAPPING
    if es_client.indices.exists(index=TEST_INDEX):
        es_client.indices.delete(index=TEST_INDEX)
    es_client.indices.create(index=TEST_INDEX, mappings=CANDIDATES_MAPPING)
    yield TEST_INDEX
    if es_client.indices.exists(index=TEST_INDEX):
        es_client.indices.delete(index=TEST_INDEX)


def _load_candidates_with_embeddings():
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    from openai import OpenAI
    from embeddings.generator import add_embeddings_to_candidate
    from etl.experience_scorer import apply_experience_scoring
    client = OpenAI()
    candidates = []
    for c in raw:
        apply_experience_scoring(c)
        work_experiences = c.get("work_experiences", [])
        for exp in work_experiences:
            exp.setdefault("standardized_title", exp.get("raw_title", ""))
        cand = {
            "post_id": c["post_id"],
            "work_experiences": work_experiences,
            "skills_text": c.get("skills_text", ""),
            "education_text": c.get("education_text", ""),
            "languages": c.get("languages", []),
            "location": c.get("location", {}),
            "work_radius_km": c.get("work_radius_km", 50),
            "pensum_desired": c.get("pensum_desired", 100),
            "pensum_from": c.get("pensum_from", 0),
            "seniority_level": c.get("seniority_level", "mid"),
            "total_weighted_relevant_years": c.get("total_weighted_relevant_years", 0),
            "aggregated_industry_parts": [tuple(x) if isinstance(x, (list, tuple)) else x for x in c.get("aggregated_industry_parts", [])],
            "available_from": c.get("available_from"),
            "retired": c.get("retired", False),
            "on_contract_basis": c.get("on_contract_basis", False),
            "job_categories_primary": [],
            "job_categories_secondary": [],
        }
        add_embeddings_to_candidate(cand, client)
        candidates.append(cand)
    return candidates


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_golden_ranking(es_client, golden_index):
    """Index 10 golden candidates, run match for RE-ACC-001 job, assert full ranking order and C07 excluded."""
    from api.matching import run_match
    from api.models import JobMatchRequest
    from es_layer.indexer import _candidate_doc

    candidates = _load_candidates_with_embeddings()
    for c in candidates:
        doc = _candidate_doc(c)
        es_client.index(index=golden_index, id=str(c["post_id"]), document=doc)
    es_client.indices.refresh(index=golden_index)

    with open(JOB_PATH, "r", encoding="utf-8") as f:
        job_data = json.load(f)
    req = JobMatchRequest(
        title=job_data["title"],
        description=job_data.get("description"),
        required_skills=job_data.get("required_skills"),
        required_education=job_data.get("required_education"),
        industry=job_data.get("industry"),
        expected_seniority_level=job_data.get("expected_seniority_level", "senior"),
        location_lat=job_data["location_lat"],
        location_lon=job_data["location_lon"],
        radius_km=job_data.get("radius_km", 50),
        pensum_min=job_data.get("pensum_min", 0),
        pensum_max=job_data.get("pensum_max", 100),
        required_languages=[{"name": l["name"], "min_level": l.get("min_level", "B2")} for l in job_data.get("required_languages", [])],
        max_results=job_data.get("max_results", 20),
        min_score=1.0,
    )

    response = run_match(req, es=es_client, index=golden_index, min_score_override=1.0, max_results_override=15)

    post_ids = [m.post_id for m in response.matches]
    assert post_ids[0] == 10004, f"Expected C04 (10004) first, got {post_ids}"
    assert post_ids[1] == 10001, f"Expected C01 (10001) second, got {post_ids}"
    assert post_ids[2] == 10009, f"Expected C09 (10009) third, got {post_ids}"
    assert EXCLUDED_POST_ID not in post_ids, f"C07 (10007) should be excluded, got {post_ids}"
    expected_in_result = [pid for pid in EXPECTED_ORDER_POST_IDS if pid in post_ids]
    assert expected_in_result == EXPECTED_ORDER_POST_IDS[:len(expected_in_result)], (
        f"Expected candidates in order {EXPECTED_ORDER_POST_IDS[:len(expected_in_result)]}, got {expected_in_result}"
    )
    assert len(response.matches) <= 9


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_golden_score_separation(es_client, golden_index):
    """C04 should score significantly higher than C07 (score gap >= 15 points out of 100)."""
    from api.matching import run_match
    from api.models import JobMatchRequest
    from es_layer.indexer import _candidate_doc

    candidates = _load_candidates_with_embeddings()
    for c in candidates:
        doc = _candidate_doc(c)
        es_client.index(index=golden_index, id=str(c["post_id"]), document=doc)
    es_client.indices.refresh(index=golden_index)

    with open(JOB_PATH, "r", encoding="utf-8") as f:
        job_data = json.load(f)
    req = JobMatchRequest(
        title=job_data["title"],
        description=job_data.get("description"),
        required_skills=job_data.get("required_skills"),
        required_education=job_data.get("required_education"),
        industry=job_data.get("industry"),
        expected_seniority_level=job_data.get("expected_seniority_level", "senior"),
        location_lat=job_data["location_lat"],
        location_lon=job_data["location_lon"],
        radius_km=job_data.get("radius_km", 50),
        pensum_min=job_data.get("pensum_min", 0),
        pensum_max=job_data.get("pensum_max", 100),
        required_languages=[{"name": l["name"], "min_level": l.get("min_level", "B2")} for l in job_data.get("required_languages", [])],
        max_results=20,
        min_score=1.0,
    )
    response = run_match(req, es=es_client, index=golden_index, min_score_override=0.0, max_results_override=20)

    scores = {m.post_id: m.score.total for m in response.matches}
    assert 10004 in scores, "C04 should appear in results"
    assert scores[10004] > 70.0, f"C04 should score above 70/100, got {scores.get(10004)}"
    assert 10007 not in scores or scores.get(10007, 0) < 50.0, (
        f"C07 should score below 50/100 or be excluded, got {scores.get(10007)}"
    )
    if 10004 in scores and 10007 in scores:
        assert scores[10004] - scores[10007] >= 15.0, (
            f"Score gap C04 vs C07 should be >= 15 pts, got {scores[10004] - scores[10007]}"
        )
