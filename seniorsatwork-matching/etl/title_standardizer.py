"""
Map raw job titles to standardized titles via GPT-4o-mini; cache results in SQLite.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from openai import OpenAI

# Default path for SQLite cache
DEFAULT_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "title_mappings.db")
BATCH_SIZE = 20


def _ensure_cache_dir():
    Path(DEFAULT_CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)


def load_standardized_titles(path: str | None = None) -> list[str]:
    """Load canonical titles from file (one per line; skip empty and # lines)."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "standardized_titles.txt")
    titles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                titles.append(line)
    return titles


def get_cached_mapping(raw_title: str, cache_path: str = DEFAULT_CACHE_PATH) -> str | None:
    """Return standardized title if cached, else None."""
    _ensure_cache_dir()
    if not os.path.exists(cache_path):
        return None
    with sqlite3.connect(cache_path) as conn:
        cur = conn.execute(
            "SELECT std_title FROM title_mappings WHERE raw_title = ?",
            (raw_title.strip(),),
        )
        row = cur.fetchone()
        return row[0] if row else None


def set_cached_mapping(raw_title: str, std_title: str, cache_path: str = DEFAULT_CACHE_PATH) -> None:
    _ensure_cache_dir()
    with sqlite3.connect(cache_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS title_mappings (
                raw_title TEXT PRIMARY KEY,
                std_title TEXT NOT NULL,
                mapped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO title_mappings (raw_title, std_title) VALUES (?, ?)",
            (raw_title.strip(), std_title.strip()),
        )
        conn.commit()


def _init_db(cache_path: str) -> None:
    _ensure_cache_dir()
    with sqlite3.connect(cache_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS title_mappings (
                raw_title TEXT PRIMARY KEY,
                std_title TEXT NOT NULL,
                mapped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def map_titles_batch(
    raw_titles: list[str],
    standardized_titles: list[str],
    client: OpenAI | None = None,
    cache_path: str = DEFAULT_CACHE_PATH,
) -> dict[str, str]:
    """
    Map a batch of raw titles to standardized titles using GPT-4o-mini.
    Uses cache; only unmapped titles are sent to the API.
    Returns dict raw_title -> standardized_title (or 'NONE').
    """
    _init_db(cache_path)
    result = {}
    to_map = []
    for t in raw_titles:
        t = t.strip()
        if not t:
            continue
        cached = get_cached_mapping(t, cache_path)
        if cached is not None:
            result[t] = cached
        else:
            to_map.append(t)

    if not to_map:
        return result

    titles_block = "\n".join(standardized_titles[:2000])  # limit token size
    for i in range(0, len(to_map), BATCH_SIZE):
        batch = to_map[i : i + BATCH_SIZE]
        titles_json = json.dumps(batch, ensure_ascii=False)
        prompt = f"""You are mapping job titles to a canonical list.
For each raw title below, return the single closest standardized title from the provided list.
If no reasonable match (similarity < 60%), return 'NONE'.
Respond as a JSON object: {{ "raw_title1": "standardized_or_NONE", "raw_title2": "..." }}.
Use the exact raw title strings as keys.

Raw titles: {titles_json}

Standardized titles list:
{titles_block}
"""

        client = client or OpenAI()
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = (resp.choices[0].message.content or "").strip()
            # Strip markdown code block if present
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            mapping = json.loads(content)
        except Exception:
            mapping = {t: "NONE" for t in batch}

        for raw, std in mapping.items():
            std = (std or "NONE").strip()
            result[raw] = std
            set_cached_mapping(raw, std, cache_path)

    return result


def apply_standardized_titles(
    candidate: dict[str, Any],
    standardized_titles: list[str],
    client: OpenAI | None = None,
    cache_path: str = DEFAULT_CACHE_PATH,
) -> dict[str, Any]:
    """
    For each work experience, set standardized_title (from cache or API).
    Mutates candidate['work_experiences'] in place.
    """
    experiences = candidate.get("work_experiences") or []
    if not experiences:
        return candidate

    raw_titles = list({exp.get("raw_title", "").strip() for exp in experiences if exp.get("raw_title")})
    mapping = map_titles_batch(raw_titles, standardized_titles, client=client, cache_path=cache_path)

    for exp in experiences:
        raw = (exp.get("raw_title") or "").strip()
        exp["standardized_title"] = mapping.get(raw, "NONE") if raw else "NONE"

    return candidate
