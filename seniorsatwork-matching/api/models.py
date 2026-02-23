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
    max_results: int = 20
    min_score: Optional[float] = None


class ScoreBreakdown(BaseModel):
    total: float = Field(..., description="0-100 normalized")
    title_score: float = 0.0
    industry_score: float = 0.0
    experience_score: float = 0.0
    skills_score: float = 0.0
    seniority_score: float = 0.0
    education_score: float = 0.0
    language_score: float = 0.0


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
    most_relevant_role: str = ""
    total_relevant_years: float = 0.0
    seniority_level: str = ""
    location: str = ""
    pensum_desired: int = 100
    top_industries: list[str] = Field(default_factory=list)
    rank: int = 0
    work_experiences: list[WorkExperienceItem] = Field(default_factory=list)
    skills_text: str = ""
    education_text: str = ""
    languages: list[CandidateLanguage] = Field(default_factory=list)
    birth_year: Optional[int] = None
    available_from: Optional[str] = None
    pensum_from: int = 0
    on_contract_basis: bool = False
    retired: bool = False
    job_categories_primary: list[str] = Field(default_factory=list)
    job_categories_secondary: list[str] = Field(default_factory=list)
    rank_explanation: list[str] = Field(default_factory=list, description="Why ranked #N bullet points")


class MatchResponse(BaseModel):
    matches: list[CandidateMatch] = Field(default_factory=list)
    message: Optional[str] = None
    total_above_threshold: int = 0
