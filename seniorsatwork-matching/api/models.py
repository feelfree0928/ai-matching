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


class CandidateMatch(BaseModel):
    post_id: int
    score: ScoreBreakdown
    most_relevant_role: str = ""
    total_relevant_years: float = 0.0
    seniority_level: str = ""
    location: str = ""
    pensum_desired: int = 100
    top_industries: list[str] = Field(default_factory=list)


class MatchResponse(BaseModel):
    matches: list[CandidateMatch] = Field(default_factory=list)
    message: Optional[str] = None
    total_above_threshold: int = 0
