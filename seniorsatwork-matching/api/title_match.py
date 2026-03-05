"""
Title matching without re-embedding: LLM job-title normalization and LLM title-fit scoring.

- Normalize job title at query time (e.g. "Nacht Rezeptionist/in (m/w/d)" -> "Nacht Rezeptionist")
  so the title vector is comparable to candidate titles; does not depend on standardized_titles.txt.
- Score each shortlisted candidate's title fit to the job title via one batch LLM call,
  then blend into ranking so exact/semantic title matches (e.g. Night Receptionist) rank above
  generic ones (e.g. Job Assistant) when the job is for night receptionist.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from api.models import CandidateMatch

MODEL = "gpt-4o-mini"


def normalize_job_title_for_matching(job_title: str, client: OpenAI | None = None) -> str | None:
    """
    Normalize a job title for embedding/matching: remove parentheticals (m/w/d), slashes,
    and return a single clean phrase in the same language. No dependency on standardized list.
    Returns None on failure so caller can fall back to raw title.
    """
    if not (job_title or "").strip():
        return None
    c = client or OpenAI()
    prompt = f"""Normalize this job title for semantic matching. Return a single short phrase: the core job title only.
- Same language as the input. Remove gender variants like /in, (m/w/d), (f/m). Remove extra punctuation.
- No explanation. One line only.

Job title: {job_title.strip()}"""
    try:
        resp = c.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=80,
        )
        out = (resp.choices[0].message.content or "").strip()
        out = re.sub(r"\s+", " ", out).strip()
        return out if out else None
    except Exception:
        return None


def score_title_fit_batch(
    job_title: str,
    matches: list["CandidateMatch"],
    client: OpenAI | None = None,
) -> dict[int, float]:
    """
    For each candidate, score 0-10 how well their main role title fits the job title.
    Returns dict post_id -> score (0-10). Missing entries get 5.0 (neutral).
    Does not depend on standardized_titles.txt; LLM understands e.g. Night Receptionist vs Job Assistant.
    """
    if not job_title or not matches:
        return {}
    c = client or OpenAI()
    # Build compact candidate list: post_id and best role title(s)
    lines = []
    for m in matches:
        role = (m.most_relevant_role or "").strip()
        if not role or role.upper() == "NONE":
            roles = []
            for e in (m.work_experiences or [])[:3]:
                r = (e.standardized_title or e.raw_title or "").strip()
                if r and r.upper() != "NONE":
                    roles.append(r)
            role = ", ".join(roles) if roles else "—"
        lines.append(f"  {m.post_id}: {role}")
    block = "\n".join(lines)

    prompt = f"""Job title: {job_title.strip()}

Candidates (post_id: main role title):
{block}

Score each candidate 0-10 on how well their role title matches the job title. 10 = same or equivalent role (e.g. Night Receptionist for "Nacht Rezeptionist"), 0 = completely unrelated (e.g. Job Assistant for a night receptionist job). Use decimals if needed.
Reply with a JSON object only: {{ "post_id": score, ... }}. One entry per post_id."""

    try:
        resp = c.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        data = json.loads(text)
        if not isinstance(data, dict):
            return {}
        result = {}
        for pid, val in data.items():
            try:
                pid_int = int(pid)
                score = float(val) if val is not None else 5.0
                result[pid_int] = max(0.0, min(10.0, score))
            except (TypeError, ValueError):
                continue
        return result
    except Exception:
        return {}
