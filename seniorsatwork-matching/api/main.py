"""
FastAPI matching service: match, config, health, sync endpoints.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from api import config as app_config
from api.matching import run_match
from api.models import JobMatchRequest, MatchResponse
from es_layer.indexer import get_es_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # shutdown if needed
    pass


app = FastAPI(title="Job Matching API", version="1.0", lifespan=lifespan)


@app.post("/api/match", response_model=MatchResponse)
def post_match(req: JobMatchRequest) -> MatchResponse:
    """Match candidates for a job; returns ranked shortlist with score breakdown."""
    return run_match(req)


@app.get("/api/jobs/{post_id}/matches", response_model=MatchResponse)
def get_job_matches(post_id: int) -> MatchResponse:
    """Get matches for an already-indexed job by post_id. Fetches job from ES and runs match."""
    es = get_es_client()
    from es_layer.mappings import JOBS_INDEX
    try:
        doc = es.get(index=JOBS_INDEX, id=str(post_id))
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")
    src = doc.get("_source") or {}
    req = JobMatchRequest(
        post_id=post_id,
        title=src.get("title", ""),
        industry=src.get("industry"),
        required_skills=src.get("required_skills_text"),
        required_education=src.get("required_education_text"),
        expected_seniority_level=src.get("expected_seniority_level", "senior"),
        location_lat=src.get("location", {}).get("lat", 47.37),
        location_lon=src.get("location", {}).get("lon", 8.54),
        radius_km=src.get("radius_km", 50),
        pensum_min=src.get("pensum_min", 0),
        pensum_max=src.get("pensum_max", 100),
        required_languages=[],
    )
    return run_match(req)


@app.post("/api/index/candidates/sync")
def post_sync_candidates() -> dict[str, Any]:
    """Trigger delta sync of candidates from WordPress."""
    import subprocess
    import sys
    import os
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "incremental_sync.py")
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        return {
            "ok": result.returncode == 0,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": "Sync timed out after 600s"}
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": str(e)}


@app.post("/api/index/jobs/sync")
def post_sync_jobs() -> dict[str, Any]:
    """Sync job postings from WordPress (placeholder – implement same pattern as candidates)."""
    return {"ok": True, "message": "Job sync not yet implemented; use initial_load or custom script."}


@app.get("/api/health")
def get_health() -> dict[str, Any]:
    """Elasticsearch and DB connectivity."""
    out = {"elasticsearch": "unknown", "database": "unknown"}
    try:
        es = get_es_client()
        info = es.info()
        out["elasticsearch"] = "ok" if info else "error"
    except Exception as e:
        out["elasticsearch"] = f"error: {e}"
    try:
        from etl.extractor import _get_connection
        conn = _get_connection()
        conn.ping()
        conn.close()
        out["database"] = "ok"
    except Exception as e:
        out["database"] = f"error: {e}"
    return out


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    """Current scoring weights, threshold, max_results."""
    cfg = app_config.load_config()
    return {
        "scoring_weights": cfg.get("scoring_weights", app_config.DEFAULT_WEIGHTS),
        "min_score_raw": cfg.get("min_score_raw", app_config.DEFAULT_MIN_SCORE_RAW),
        "max_results": cfg.get("max_results", app_config.DEFAULT_MAX_RESULTS),
    }


class ConfigUpdate(BaseModel):
    scoring_weights: dict[str, float] | None = None
    min_score_raw: float | None = None
    max_results: int | None = None


@app.patch("/api/config")
def patch_config(updates: ConfigUpdate) -> dict[str, Any]:
    """Update weights/threshold; persisted to config.json."""
    u = {}
    if updates.scoring_weights is not None:
        u["scoring_weights"] = updates.scoring_weights
    if updates.min_score_raw is not None:
        u["min_score_raw"] = updates.min_score_raw
    if updates.max_results is not None:
        u["max_results"] = updates.max_results
    if not u:
        return get_config()
    cfg = app_config.update_config(u)
    return {
        "scoring_weights": cfg.get("scoring_weights"),
        "min_score_raw": cfg.get("min_score_raw"),
        "max_results": cfg.get("max_results"),
    }


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "Job Matching API", "docs": "/docs"}
