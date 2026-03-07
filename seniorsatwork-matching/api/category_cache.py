"""
In-process cache for WordPress term labels (job category names).
Loaded once per server process from the DB; avoids per-request DB calls.
"""
from __future__ import annotations

_TERM_LABELS_CACHE: list[str] | None = None


def get_known_category_labels() -> list[str]:
    """Return sorted list of known job category label strings, loading from DB on first call."""
    global _TERM_LABELS_CACHE
    if _TERM_LABELS_CACHE is None:
        try:
            from etl.extractor import fetch_term_labels_standalone
            labels = fetch_term_labels_standalone()
            _TERM_LABELS_CACHE = sorted(set(labels.values()))
        except Exception:
            _TERM_LABELS_CACHE = []
    return _TERM_LABELS_CACHE
