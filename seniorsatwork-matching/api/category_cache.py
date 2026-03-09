"""
In-process cache for WordPress term labels (job category names).
Loaded once per server process from the DB; avoids per-request DB calls.
Falls back to ES aggregation when DB is unavailable.
"""
from __future__ import annotations

_TERM_LABELS_CACHE: list[str] | None = None
_TERM_LABELS_MAP: dict[str, str] | None = None  # id -> label


def _fetch_categories_from_es() -> list[str]:
    """Extract unique job_category_labels from indexed candidates (fallback when DB unavailable)."""
    try:
        from es_layer.indexer import get_es_client
        from es_layer.mappings import CANDIDATES_INDEX
        es = get_es_client()
        resp = es.search(
            index=CANDIDATES_INDEX,
            size=0,
            aggs={
                "categories": {
                    "terms": {
                        "field": "job_category_labels",
                        "size": 500,
                        "min_doc_count": 1,
                    }
                }
            },
        )
        buckets = resp.get("aggregations", {}).get("categories", {}).get("buckets", [])
        return sorted(set(b.get("key", "") for b in buckets if b.get("key")))
    except Exception:
        return []


def get_known_category_labels() -> list[str]:
    """Return sorted list of known job category label strings. Tries DB first, then ES fallback."""
    global _TERM_LABELS_CACHE
    if _TERM_LABELS_CACHE is None:
        try:
            from etl.extractor import fetch_term_labels_standalone
            labels = fetch_term_labels_standalone()
            _TERM_LABELS_CACHE = sorted(set(labels.values()))
        except Exception:
            pass
        if not _TERM_LABELS_CACHE:
            _TERM_LABELS_CACHE = _fetch_categories_from_es()
    return _TERM_LABELS_CACHE


def get_term_id_to_label() -> dict[str, str]:
    """Return {term_id: label} for resolving IDs to labels and vice versa."""
    global _TERM_LABELS_MAP
    if _TERM_LABELS_MAP is None:
        try:
            from etl.extractor import fetch_term_labels_standalone
            _TERM_LABELS_MAP = fetch_term_labels_standalone()
        except Exception:
            _TERM_LABELS_MAP = {}
    return _TERM_LABELS_MAP or {}


def resolve_to_category_ids(values: list[str]) -> list[str]:
    """Resolve category values to IDs for strict filtering. Labels are mapped to IDs; IDs pass through."""
    if not values:
        return []
    id_to_label = get_term_id_to_label()
    label_to_id = {v: k for k, v in id_to_label.items()}
    ids = []
    for v in values:
        v = str(v).strip()
        if not v:
            continue
        if v in label_to_id:
            ids.append(label_to_id[v])
        else:
            ids.append(v)  # already an ID
    return list(dict.fromkeys(ids))
