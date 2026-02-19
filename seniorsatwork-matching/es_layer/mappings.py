"""
Elasticsearch index mappings for candidates and job_postings.
"""
DENSE_DIMS = 1536

CANDIDATES_INDEX = "candidates"
JOBS_INDEX = "job_postings"

CANDIDATES_MAPPING = {
    "dynamic": "false",
    "properties": {
        "post_id": {"type": "long"},
        "post_modified": {"type": "date"},
        "location": {"type": "geo_point"},
        "address": {"type": "keyword"},
        "work_radius_km": {"type": "integer"},
        "pensum_desired": {"type": "integer"},
        "pensum_from": {"type": "integer"},
        "on_contract_basis": {"type": "boolean"},
        "languages": {
            "type": "nested",
            "properties": {
                "lang": {"type": "keyword"},
                "degree": {"type": "keyword"},
            },
        },
        "seniority_level": {"type": "keyword"},
        "seniority_level_int": {"type": "integer"},
        "work_experiences": {
            "type": "nested",
            "properties": {
                "raw_title": {"type": "text"},
                "standardized_title": {"type": "keyword"},
                "industry": {"type": "keyword"},
                "start_year": {"type": "integer"},
                "end_year": {"type": "integer"},
                "years_in_role": {"type": "integer"},
                "recency_weight": {"type": "float"},
                "weighted_years": {"type": "float"},
                "description": {"type": "text"},
            },
        },
        "aggregated_title_embedding": {
            "type": "dense_vector",
            "dims": DENSE_DIMS,
            "index": True,
            "similarity": "cosine",
        },
        "aggregated_industry_embedding": {
            "type": "dense_vector",
            "dims": DENSE_DIMS,
            "index": True,
            "similarity": "cosine",
        },
        "total_weighted_relevant_years": {"type": "float"},
        "skills_embedding": {
            "type": "dense_vector",
            "dims": DENSE_DIMS,
            "index": True,
            "similarity": "cosine",
        },
        "education_embedding": {
            "type": "dense_vector",
            "dims": DENSE_DIMS,
            "index": True,
            "similarity": "cosine",
        },
        "skills_text": {"type": "text"},
        "education_text": {"type": "text"},
        "birth_year": {"type": "integer"},
        "retired": {"type": "boolean"},
        "job_categories_primary": {"type": "keyword"},
        "job_categories_secondary": {"type": "keyword"},
    },
}

JOBS_MAPPING = {
    "dynamic": "false",
    "properties": {
        "post_id": {"type": "long"},
        "post_modified": {"type": "date"},
        "title": {"type": "text"},
        "standardized_title": {"type": "keyword"},
        "title_embedding": {
            "type": "dense_vector",
            "dims": DENSE_DIMS,
            "index": True,
            "similarity": "cosine",
        },
        "industry": {"type": "keyword"},
        "industry_embedding": {
            "type": "dense_vector",
            "dims": DENSE_DIMS,
            "index": True,
            "similarity": "cosine",
        },
        "required_skills_text": {"type": "text"},
        "skills_embedding": {
            "type": "dense_vector",
            "dims": DENSE_DIMS,
            "index": True,
            "similarity": "cosine",
        },
        "required_education_text": {"type": "text"},
        "education_embedding": {
            "type": "dense_vector",
            "dims": DENSE_DIMS,
            "index": True,
            "similarity": "cosine",
        },
        "expected_seniority_level": {"type": "keyword"},
        "expected_seniority_level_int": {"type": "integer"},
        "location": {"type": "geo_point"},
        "radius_km": {"type": "integer"},
        "pensum_min": {"type": "integer"},
        "pensum_max": {"type": "integer"},
        "required_languages": {
            "type": "nested",
            "properties": {
                "name": {"type": "keyword"},
                "min_level": {"type": "keyword"},
            },
        },
    },
}

SENIORITY_TO_INT = {
    "junior": 0,
    "mid": 1,
    "senior": 2,
    "manager": 3,
    "director": 4,
    "executive": 5,
}
