"""
Pydantic request/response models for the matching API.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class LanguageRequirement(BaseModel):
    name: str = Field(..., description="e.g. German")
    min_level: str = Field(default="B2", description="B2, C1, native")


class JobMatchRequest(BaseModel):
    post_id: Optional[int] = None
    title: str
    description: Optional[str] = None
    required_skills: Optional[str] = None
    required_education: Optional[str] = None
    industry: Optional[str] = None
    expected_seniority_level: str = Field(default="senior", description="junior|mid|senior|manager|director|executive")
    location_lat: float
    location_lon: float
    radius_km: int = 50
    pensum_min: int = 0
    pensum_max: int = 100
    required_languages: list[LanguageRequirement] = Field(default_factory=list)
    required_available_before: Optional[str] = Field(default=None, description="ISO date (yyyy-MM-dd); only candidates available on or before this date pass")
    max_results: Optional[int] = None
    min_score: Optional[float] = None


class ScoreBreakdown(BaseModel):
    total: float = Field(..., description="0-100 normalized")
    raw_score: float = Field(default=0.0, description="Elasticsearch script score (typical ~1.0-1.45)")
    title_score: float = 0.0
    industry_score: float = 0.0
    experience_score: float = 0.0
    skills_score: float = 0.0
    seniority_score: float = 0.0
    education_score: float = 0.0
    language_score: float = 0.0
    # How total is computed: raw_score = Σ(value×weight); total = (raw_score/1.4)×100
    total_formula: str = Field(
        default="",
        description="Short explanation: raw_score = Σ(value × weight); total = (raw_score / 1.4) × 100",
    )
    # Per-dimension: parameter, value, weight, and contribution = (value×weight)/raw_score×total
    score_calculation: list[dict[str, Any]] = Field(
        default_factory=list,
        description='e.g. {"parameter": "Title", "value": 0.8, "weight": 0.15, "contribution": 13.3}',
    )
    # Ready-to-display string: "Score 78.6 (Title: 1.24 × 0.15 = 13.3, Experience: ...)"
    score_display: str = Field(
        default="",
        description="Formatted for UI: Score X (Title: v × w = c, ...)",
    )
    # How experience was calculated (primary_years, secondary_years, exp_primary, exp_secondary, etc.)
    experience_detail: Optional[dict[str, Any]] = Field(
        default=None,
        description="Experience sub-components for tooltips: primary_years, secondary_years, primary_relevance, exp_primary, exp_secondary, none_penalty",
    )


class WorkExperienceItem(BaseModel):
    raw_title: str = ""
    standardized_title: str = ""
    industry: str = ""
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    years_in_role: Optional[float] = None
    weighted_years: Optional[float] = None


class CandidateLanguage(BaseModel):
    lang: str = ""
    degree: str = ""


class CandidateMatch(BaseModel):
    post_id: int
    score: ScoreBreakdown
    rank: int = 0
    rank_explanation: list[str] = Field(default_factory=list, description="Why ranked #N bullet points")

    # ── Identity & contact ───────────────────────────────
    candidate_name: str = ""
    phone: str = ""
    gender: str = ""
    linkedin_url: str = ""
    website_url: str = ""
    cv_file: str = ""

    # ── Summary & expectations ───────────────────────────
    short_description: str = ""
    job_expectations: str = ""
    highest_degree: str = ""

    # ── AI-generated descriptions (when available) ───────
    ai_profile_description: str = ""
    ai_experience_description: str = ""
    ai_skills_description: str = ""
    ai_text_skill_result: str = ""

    # ── Experience & skills ──────────────────────────────
    most_relevant_role: str = ""
    total_relevant_years: float = 0.0
    seniority_level: str = ""
    work_experiences: list[WorkExperienceItem] = Field(default_factory=list)
    skills_text: str = ""
    education_text: str = ""
    most_experience_industries: list[str] = Field(default_factory=list)
    top_industries: list[str] = Field(default_factory=list)

    # ── Languages ────────────────────────────────────────
    languages: list[CandidateLanguage] = Field(default_factory=list)

    # ── Location ─────────────────────────────────────────
    location: str = ""
    zip_code: str = ""
    work_radius_km: int = 50
    work_radius_text: str = ""

    # ── Availability & contract ──────────────────────────
    available_from: Optional[str] = None
    pensum_desired: int = 100
    pensum_from: int = 0
    pensum_duration: str = ""
    on_contract_basis: bool = False
    voluntary: str = ""

    # ── Personal ─────────────────────────────────────────
    birth_year: Optional[int] = None
    retired: bool = False

    # ── Categories ───────────────────────────────────────
    job_categories_primary: list[str] = Field(default_factory=list)
    job_categories_secondary: list[str] = Field(default_factory=list)

    # ── Profile meta ─────────────────────────────────────
    profile_status: str = ""
    registered_at: Optional[str] = None
    expires_at: Optional[str] = None
    featured: bool = False
    post_date: Optional[str] = None


class MatchResponse(BaseModel):
    matches: list[CandidateMatch] = Field(default_factory=list)
    message: Optional[str] = None
    total_above_threshold: int = 0
