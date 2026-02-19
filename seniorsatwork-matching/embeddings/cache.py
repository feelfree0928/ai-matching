"""
SQLite cache for embeddings keyed by SHA-256 of input text. Avoids re-calling OpenAI when text unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import numpy as np

DEFAULT_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings.db")
EMBEDDING_MODEL = "text-embedding-3-small"


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def get_cached_embedding(text: str, cache_path: str = DEFAULT_CACHE_PATH) -> list[float] | None:
    """Return cached vector if present, else None."""
    if not text or not text.strip():
        return None
    _ensure_dir(cache_path)
    if not os.path.exists(cache_path):
        return None
    key = _text_hash(text.strip())
    with sqlite3.connect(cache_path) as conn:
        cur = conn.execute(
            "SELECT vector FROM embeddings WHERE text_hash = ? AND model = ?",
            (key, EMBEDDING_MODEL),
        )
        row = cur.fetchone()
        if not row:
            return None
        vec = np.frombuffer(row[0], dtype=np.float32).tolist()
        return vec


def set_cached_embedding(
    text: str,
    vector: list[float],
    cache_path: str = DEFAULT_CACHE_PATH,
) -> None:
    _ensure_dir(cache_path)
    with sqlite3.connect(cache_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                vector BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (text_hash, model)
            )
            """
        )
        arr = np.array(vector, dtype=np.float32)
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (text_hash, model, vector) VALUES (?, ?, ?)",
            (_text_hash(text.strip()), EMBEDDING_MODEL, arr.tobytes()),
        )
        conn.commit()
