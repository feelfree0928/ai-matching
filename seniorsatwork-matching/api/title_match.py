"""
Title matching utilities:
- normalize_and_resolve_categories(): single LLM call that both normalizes the job title
  and resolves matching category labels from the known WP taxonomy. Used at query time.
- normalize_job_title_for_matching(): standalone title normalization (kept for direct use).
"""
from __future__ import annotations

import json
import re

from openai import OpenAI

MODEL = "gpt-4o-mini"


def normalize_and_resolve_categories(
    job_title: str,
    known_category_labels: list[str],
    client: OpenAI | None = None,
) -> tuple[str, list[str]]:
    """
    Single LLM call: normalize job title AND resolve matching category labels.

    Returns:
        (normalized_title, matched_category_labels)
        normalized_title: clean job title for embedding (same language, no m/w/d etc.)
        matched_category_labels: subset of known_category_labels that best match this job.
        On failure returns (raw job_title, []).
    """
    if not (job_title or "").strip():
        return job_title, []

    c = client or OpenAI()
    cats_block = "\n".join(f"  - {lbl}" for lbl in (known_category_labels or [])[:300])

    prompt = f"""You are processing a job posting title for a matching system.

Job title: {job_title.strip()}

Task 1 – Normalize the title:
  Return a clean, short job title: same language as input, remove gender variants (/in, m/w/d, f/m/d),
  remove extra punctuation, keep the core role name only.

Task 2 – Match categories:
  From the list below, select the labels that best describe this job's function.
  Return 0–3 labels that clearly match. Return empty array if none fit well.

Available category labels:
{cats_block}

Reply with JSON only (no markdown):
{{"normalized_title": "...", "category_labels": ["label1", "label2"]}}"""

    try:
        resp = c.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        data = json.loads(text)
        normalized = str(data.get("normalized_title") or "").strip() or job_title
        raw_labels = data.get("category_labels") or []
        known_set = set(known_category_labels or [])
        matched = [lbl for lbl in raw_labels if isinstance(lbl, str) and lbl in known_set]
        return normalized, matched
    except Exception:
        return job_title, []


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
