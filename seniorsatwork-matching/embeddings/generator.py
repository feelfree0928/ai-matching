"""
Generate embeddings via OpenAI text-embedding-3-small; cache in SQLite; compute weighted aggregates.
"""
from __future__ import annotations

from typing import Any

from openai import OpenAI

from .cache import DEFAULT_CACHE_PATH, get_cached_embedding, set_cached_embedding

EMBEDDING_MODEL = "text-embedding-3-small"
DIMS = 1536
BATCH_SIZE = 100


def _embed_batch(texts: list[str], client: OpenAI, cache_path: str) -> list[list[float]]:
    """Embed a batch; use cache where possible, call API for rest."""
    results = [None] * len(texts)
    to_call = []
    indices = []
    for i, t in enumerate(texts):
        if not (t and t.strip()):
            results[i] = [0.0] * DIMS
            continue
        cached = get_cached_embedding(t.strip(), cache_path)
        if cached is not None:
            results[i] = cached
        else:
            to_call.append(t.strip())
            indices.append(i)

    if to_call:
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=to_call)
        by_idx = {e.index: e.embedding for e in resp.data}
        for j, idx in enumerate(indices):
            vec = by_idx.get(j, [0.0] * DIMS)
            results[idx] = vec
            set_cached_embedding(to_call[j], vec, cache_path)

    return results


def weighted_mean_embedding(
    items: list[tuple[str, float]],
    client: OpenAI,
    cache_path: str = DEFAULT_CACHE_PATH,
) -> list[float] | None:
    """
    items = [(text, weight), ...]. Returns weighted mean of embeddings, or zero vector if empty.
    """
    if not items:
        return None
    texts = [x[0] for x in items]
    weights = [max(0.0, float(x[1])) for x in items]
    total_w = sum(weights)
    if total_w <= 0:
        return None
    vecs = _embed_batch(texts, client, cache_path)
    out = [0.0] * DIMS
    for v, w in zip(vecs, weights):
        if v is None:
            continue
        for i in range(DIMS):
            out[i] += v[i] * w
    for i in range(DIMS):
        out[i] /= total_w
    return out


def embed_text(
    text: str,
    client: OpenAI | None = None,
    cache_path: str = DEFAULT_CACHE_PATH,
) -> list[float]:
    """Single text to embedding; use cache."""
    if not text or not text.strip():
        return [0.0] * DIMS
    cached = get_cached_embedding(text.strip(), cache_path)
    if cached is not None:
        return cached
    c = client or OpenAI()
    vecs = _embed_batch([text.strip()], c, cache_path)
    return vecs[0] if vecs else [0.0] * DIMS


def add_embeddings_to_candidate(
    candidate: dict[str, Any],
    client: OpenAI,
    cache_path: str = DEFAULT_CACHE_PATH,
) -> dict[str, Any]:
    """
    Compute and set aggregated_title_embedding, aggregated_industry_embedding,
    primary_role_title_embedding, skills_embedding, education_embedding on candidate.
    """
    experiences = candidate.get("work_experiences") or []
    # Title: weighted by recency_weight * years_in_role using raw_title directly.
    # No standardization step: text-embedding-3-small handles German natively and
    # standardization was causing NONE penalties for titles not in the canonical list.
    title_items = []
    for exp in experiences:
        t = (exp.get("raw_title") or "").strip()
        if t:
            w = float(exp.get("recency_weight", 1.0)) * float(exp.get("years_in_role", 1))
            title_items.append((t, w))
    candidate["aggregated_title_embedding"] = weighted_mean_embedding(
        title_items, client, cache_path
    ) if title_items else None
    # Fallback: no work history at all – use skills or generic so candidate can still be indexed
    if candidate.get("aggregated_title_embedding") is None:
        pass  # handled below

    # Primary role title embedding: embed only the single highest-weighted role title.
    # Stored separately so Painless can compute an isolated per-role cosine similarity
    # instead of relying on the blended aggregated vector.
    primary_role_title = (candidate.get("primary_role_title") or "").strip()
    if primary_role_title and primary_role_title != "NONE":
        candidate["primary_role_title_embedding"] = embed_text(primary_role_title, client, cache_path)
    elif title_items:
        # Fall back to the highest-weight title from the title_items list
        best = max(title_items, key=lambda x: x[1])
        candidate["primary_role_title_embedding"] = embed_text(best[0], client, cache_path)
    else:
        candidate["primary_role_title_embedding"] = None

    if candidate.get("aggregated_title_embedding") is None:
        fallback_text = (candidate.get("skills_text") or "").strip()[:500] or "Professional"
        candidate["aggregated_title_embedding"] = embed_text(fallback_text, client, cache_path)

    # Industry: weighted same way
    industry_parts = candidate.get("aggregated_industry_parts") or []
    if industry_parts:
        candidate["aggregated_industry_embedding"] = weighted_mean_embedding(
            industry_parts, client, cache_path
        )
    else:
        industry_items = []
        for exp in experiences:
            ind = (exp.get("industry") or "").strip()
            if ind:
                w = float(exp.get("weighted_years", 1.0))
                industry_items.append((ind, w))
        candidate["aggregated_industry_embedding"] = weighted_mean_embedding(
            industry_items, client, cache_path
        ) if industry_items else None

    # Skills
    skills_text = (candidate.get("skills_text") or "").strip()
    if skills_text:
        candidate["skills_embedding"] = embed_text(skills_text, client, cache_path)
    else:
        candidate["skills_embedding"] = None

    # Education
    education_text = (candidate.get("education_text") or "").strip()
    if education_text:
        candidate["education_embedding"] = embed_text(education_text, client, cache_path)
    else:
        candidate["education_embedding"] = None

    # Do NOT pre-fill missing embeddings with zero vectors.
    # The ES indexer converts zero vectors to a unit vector [1, 0, 0...], which means
    # the size()==0 neutral guard in the Painless script never fires and cosine similarity
    # is computed against an arbitrary axis — corrupting industry/skills/education scoring.
    # Leaving as None means the field is omitted from the ES document, size()==0 returns true,
    # and the Painless script correctly returns 1.0 (neutral) for candidates with no data.

    return candidate
