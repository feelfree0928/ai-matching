"""
LLM re-ranking: after ES matching, re-rank candidates and drop irrelevant ones using an LLM.
Always applied (no config toggle). Uses a seasoned-recruiter persona for logical ordering.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from api.models import CandidateMatch, JobMatchRequest

RERANK_MODEL = "gpt-4o-mini"
MAX_CANDIDATES_FOR_RERANK = 50
CANDIDATE_SUMMARY_CHARS = 400
JOB_DESC_CHARS = 600
JOB_SKILLS_CHARS = 350


def _job_summary(req: "JobMatchRequest") -> str:
    parts = [
        f"Title: {req.title or ''}",
        f"Industry: {req.industry or 'Not specified'}",
        f"Expected seniority: {req.expected_seniority_level or 'Not specified'}",
    ]
    if req.description:
        parts.append(f"Description: {(req.description or '')[:JOB_DESC_CHARS]}")
    if req.required_skills:
        parts.append(f"Required skills: {(req.required_skills or '')[:JOB_SKILLS_CHARS]}")
    if req.required_education:
        parts.append(f"Required education: {(req.required_education or '')[:250]}")
    return "\n".join(parts)


def _candidate_summary(m: "CandidateMatch") -> str:
    exp_lines = []
    for e in (m.work_experiences or [])[:8]:
        title = e.standardized_title or e.raw_title or "—"
        ind = f" · {e.industry}" if e.industry else ""
        yrs = f" ({e.years_in_role}y)" if e.years_in_role else ""
        exp_lines.append(f"  - {title}{yrs}{ind}")
    exp_block = "\n".join(exp_lines) if exp_lines else "  (none)"
    skills = (m.skills_text or "")[:200].replace("\n", " ")
    industries = ", ".join((m.top_industries or m.most_experience_industries or [])[:3])
    cats = ", ".join((m.job_category_labels or [])[:3])
    profile = (m.short_description or m.ai_profile_description or "")[:150].replace("\n", " ")
    return (
        f"[post_id={m.post_id}]\n"
        f"Best role: {m.most_relevant_role or '—'} | {m.total_relevant_years:.0f} yrs relevant | Seniority: {m.seniority_level or '—'}\n"
        f"Industries: {industries or '—'}\n"
        f"Job function: {cats or '—'}\n"
        f"Career: {exp_block}\n"
        f"Skills: {skills or '—'}\n"
        f"Profile: {profile or '—'}"
    )


def _parse_rerank_response(text: str) -> list[int] | None:
    """Extract ordered post_ids from LLM JSON. Expects {"post_ids": [1, 2, ...]} or [1, 2, ...]."""
    if not text or not text.strip():
        return None
    text = text.strip()
    # Strip markdown code block if present
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [int(x) for x in data if isinstance(x, (int, float))]
        if isinstance(data, dict):
            ids = data.get("post_ids") or data.get("post_ids_order") or data.get("order")
            if ids is None and "candidates" in data:
                ids = [c.get("post_id") for c in data["candidates"] if isinstance(c, dict)]
            if isinstance(ids, list):
                return [int(x) for x in ids if x is not None and isinstance(x, (int, float))]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def rerank_candidates(
    req: "JobMatchRequest",
    matches: list["CandidateMatch"],
    client: OpenAI | None = None,
) -> list["CandidateMatch"]:
    """
    Re-rank candidates using an LLM as a seasoned recruiter. Drops candidates who are
    completely irrelevant. Returns a new list in recruiter-logical order (best first).
    On failure or empty LLM response, returns original list unchanged.
    """
    if not matches:
        return matches
    # Cap to avoid token overflow and latency
    to_rerank = matches[:MAX_CANDIDATES_FOR_RERANK]
    job_text = _job_summary(req)
    candidates_text = "\n\n".join(_candidate_summary(m) for m in to_rerank)

    system_prompt = """You are a seasoned recruiter with decades of experience. Your task is to re-rank candidates for a job posting.

Rules:
- Output ONLY a JSON object with a single key "post_ids" whose value is an array of integers (the candidate post_id values), in order of fit: best match first.
- Include ONLY candidates who are relevant to the job. Omit anyone who is clearly irrelevant (wrong field, no transferable experience, or fundamentally mismatched).
- Rank by logical fit: role relevance, industry alignment, experience level, and skills match. Prefer candidates who clearly meet the job's core requirements.
- Do not add commentary. Output valid JSON only."""

    user_prompt = f"""Job posting:\n{job_text}\n\nCandidates (each block starts with [post_id=...]):\n{candidates_text}\n\nReturn a JSON object with key "post_ids": an array of post_id values in order of best to worst fit. Include only relevant candidates. Example: {{"post_ids": [123, 456, 789]}}"""

    try:
        c = client or OpenAI()
        resp = c.chat.completions.create(
            model=RERANK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        content = (resp.choices[0].message.content or "").strip()
        ordered_ids = _parse_rerank_response(content)
    except Exception:
        ordered_ids = None

    if not ordered_ids:
        return matches

    by_id = {m.post_id: m for m in to_rerank}
    reordered = []
    for pid in ordered_ids:
        if pid in by_id:
            reordered.append(by_id[pid])
    # Only return candidates the LLM deemed relevant (in LLM order). Do not append omitted ones.
    if len(matches) > len(to_rerank):
        reordered.extend(matches[len(to_rerank):])
    return reordered
