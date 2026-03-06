"""
MySQL extractor: pull candidates (resumes) and job postings from WordPress/MariaDB.
"""
import os
from typing import Any

import pymysql
from dotenv import load_dotenv

load_dotenv()

# Meta keys we need for candidate profiles (resumes)
RESUME_META_KEYS = [
    # ── Location ─────────────────────────────────────────
    "_resume_address",
    "_resume_address_lat",
    "_resume_address_lon",
    "_noo_resume_field_job_field_zip",
    "_noo_resume_field_job_field_arbeitsradius_km",
    "_noo_resume_field_job_field_arbeitsradius",
    # ── Work history & experience ─────────────────────────
    "_noo_resume_field__taetigkeiten",
    "_noo_resume_field_job_field_most_experience_branches",
    # ── Skills, education, expectations ──────────────────
    "_noo_resume_field_job_field_technische_kenntnisse",
    "_noo_resume_field_job_field_diplome",
    "_highest_degree",
    "_job_expectations",
    # ── AI-generated text (audio transcriptions) ─────────
    "_noo_resume_field_job_field_audio_describe_result",
    "_noo_resume_field_job_field_audio_experience_result",
    "_noo_resume_field_job_field_audio_skill_result",
    "_noo_resume_field_job_field_text_skill_result",
    # ── Languages ─────────────────────────────────────────
    "_noo_resume_field_languages_i_speak",
    # ── Availability & contract terms ────────────────────
    "_noo_resume_field_job_field_pensum",
    "_noo_resume_field_job_field_pensum_from",
    "_noo_resume_field_job_field_pensum_duration",
    "_noo_resume_field_job_field_available_from",
    "_noo_resume_field_job_field_auftragsbasis",
    "_noo_resume_field_job_field_freiwillig",
    # ── Categories ───────────────────────────────────────
    "_noo_resume_field_job_category_primary",
    "_noo_resume_field_job_category_secondary",
    "_job_category",
    # ── Personal info ────────────────────────────────────
    "_noo_resume_field__jahrgang",
    "_noo_resume_field__sex",
    "_noo_resume_field__status",
    "_noo_resume_field__registration",
    "_noo_resume_field_already_retired",
    # ── Contact & online presence ─────────────────────────
    "_noo_resume_field__phone",
    "linkedin",
    "_noo_resume_field_linkedin",
    "website",
    "_noo_resume_field_cvfile",
    # ── Profile meta ─────────────────────────────────────
    "_viewable",
    "_featured",
    "_expires",
    "user_short_description",
]

# Post types in WordPress (verify against live DB)
POST_TYPE_RESUME = "noo_resume"  # Noo Job Board uses 'noo_resume', not 'resume'
POST_TYPE_JOB = "noo_job"  # Noo Job Board uses 'noo_job', not 'job_listing'


def _get_connection() -> pymysql.Connection:
    return pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASS", ""),
        database=os.getenv("DB_NAME", ""),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def _meta_keys_placeholder(keys: list[str]) -> str:
    """Build SQL placeholder list for meta_key IN (...)."""
    return ", ".join("%s" for _ in keys)


def fetch_term_labels(conn: pymysql.Connection, taxonomy: str = "job_category") -> dict[str, str]:
    """
    Fetch WordPress taxonomy term labels (term_id -> name).
    Used to map category IDs to human-readable labels for job_category_labels.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT t.term_id, t.name
            FROM wp_terms t
            JOIN wp_term_taxonomy tt ON t.term_id = tt.term_id
            WHERE tt.taxonomy = %s
            """,
            (taxonomy,),
        )
        rows = cur.fetchall()
    return {str(r["term_id"]): (r["name"] or "").strip() for r in rows}


def fetch_term_labels_standalone(taxonomy: str = "job_category") -> dict[str, str]:
    """
    Fetch term labels for pipeline use. Opens and closes its own connection.
    Call once at pipeline startup.
    """
    conn = _get_connection()
    try:
        return fetch_term_labels(conn, taxonomy)
    finally:
        conn.close()


def fetch_job_categories_by_post_ids(conn: pymysql.Connection, post_ids: list[int], taxonomy: str = "job_category") -> dict[int, list[str]]:
    """
    Fetch job category labels per post_id via wp_term_relationships.
    Returns {post_id: [label1, label2, ...]}.
    """
    if not post_ids:
        return {}
    placeholders = ", ".join("%s" for _ in post_ids)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT tr.object_id AS post_id, t.name
            FROM wp_term_relationships tr
            JOIN wp_term_taxonomy tt ON tr.term_taxonomy_id = tt.term_taxonomy_id
            JOIN wp_terms t ON tt.term_id = t.term_id
            WHERE tr.object_id IN ({placeholders})
            AND tt.taxonomy = %s
            """,
            post_ids + [taxonomy],
        )
        rows = cur.fetchall()
    result: dict[int, list[str]] = {}
    for row in rows:
        pid = row["post_id"]
        name = (row["name"] or "").strip()
        if pid not in result:
            result[pid] = []
        if name:
            result[pid].append(name)
    return result


def fetch_job_categories_standalone(post_ids: list[int], taxonomy: str = "job_category") -> dict[int, list[str]]:
    """Fetch job categories for given post IDs. Opens and closes its own connection."""
    conn = _get_connection()
    try:
        return fetch_job_categories_by_post_ids(conn, post_ids, taxonomy)
    finally:
        conn.close()


def extract_candidates(
    post_type: str = POST_TYPE_RESUME,
    post_status: str = "publish",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Extract resumes from wp_posts and their meta from wp_postmeta.
    Returns list of dicts: { "post_id", "post_content", "post_excerpt", "post_modified", "meta": { meta_key: meta_value } }.
    
    Args:
        post_type: WordPress post type (default: POST_TYPE_RESUME)
        post_status: WordPress post status (default: "publish")
        limit: Optional limit on number of candidates to extract (for testing). None = extract all.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            if limit is not None:
                # When limiting, only get candidates that have location (lat/lon) so they can be indexed
                query = """
                    SELECT DISTINCT p.ID AS post_id, p.post_title, p.post_content,
                                    p.post_excerpt, p.post_modified, p.post_date
                    FROM wp_posts p
                    INNER JOIN wp_postmeta pm_lat ON p.ID = pm_lat.post_id
                        AND pm_lat.meta_key = '_resume_address_lat'
                        AND pm_lat.meta_value IS NOT NULL AND TRIM(pm_lat.meta_value) != ''
                    INNER JOIN wp_postmeta pm_lon ON p.ID = pm_lon.post_id
                        AND pm_lon.meta_key = '_resume_address_lon'
                        AND pm_lon.meta_value IS NOT NULL AND TRIM(pm_lon.meta_value) != ''
                    WHERE p.post_type = %s AND p.post_status = %s
                    ORDER BY p.ID
                    LIMIT %s
                """
                params = [post_type, post_status, limit]
            else:
                query = """
                    SELECT ID AS post_id, post_title, post_content, post_excerpt,
                           post_modified, post_date
                    FROM wp_posts
                    WHERE post_type = %s AND post_status = %s
                    ORDER BY ID
                """
                params = [post_type, post_status]

            cur.execute(query, params)
            rows = cur.fetchall()

        if not rows:
            return []

        post_ids = [r["post_id"] for r in rows]
        meta_by_post: dict[int, dict[str, Any]] = {r["post_id"]: {} for r in rows}

        with conn.cursor() as cur:
            placeholders = ", ".join("%s" for _ in post_ids)
            cur.execute(
                f"""
                SELECT post_id, meta_key, meta_value
                FROM wp_postmeta
                WHERE post_id IN ({placeholders})
                AND meta_key IN ({_meta_keys_placeholder(RESUME_META_KEYS)})
                """,
                post_ids + RESUME_META_KEYS,
            )
            for row in cur.fetchall():
                pid = row["post_id"]
                if pid in meta_by_post:
                    meta_by_post[pid][row["meta_key"]] = row["meta_value"]

        result = []
        for r in rows:
            result.append({
                "post_id": r["post_id"],
                "post_title": r.get("post_title") or "",
                "post_content": r["post_content"] or "",
                "post_excerpt": r["post_excerpt"] or "",
                "post_modified": r["post_modified"],
                "post_date": r.get("post_date"),
                "meta": meta_by_post.get(r["post_id"], {}),
            })
        return result
    finally:
        conn.close()


def extract_job_postings(
    post_type: str = POST_TYPE_JOB,
    post_status: str = "publish",
) -> list[dict[str, Any]]:
    """
    Extract all published job postings from wp_posts.
    Returns list of dicts with post_id, post_content, post_excerpt, post_modified, and meta.
    Job postings may use different meta keys; we fetch all meta for those post_ids.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ID AS post_id, post_content, post_excerpt, post_title, post_modified
                FROM wp_posts
                WHERE post_type = %s AND post_status = %s
                ORDER BY ID
                """,
                (post_type, post_status),
            )
            rows = cur.fetchall()

        if not rows:
            return []

        post_ids = [r["post_id"] for r in rows]
        meta_by_post: dict[int, dict[str, Any]] = {r["post_id"]: {} for r in rows}

        with conn.cursor() as cur:
            placeholders = ", ".join("%s" for _ in post_ids)
            cur.execute(
                f"""
                SELECT post_id, meta_key, meta_value
                FROM wp_postmeta
                WHERE post_id IN ({placeholders})
                """,
                post_ids,
            )
            for row in cur.fetchall():
                pid = row["post_id"]
                if pid in meta_by_post:
                    meta_by_post[pid][row["meta_key"]] = row["meta_value"]

        result = []
        for r in rows:
            result.append({
                "post_id": r["post_id"],
                "post_title": r.get("post_title") or "",
                "post_content": r["post_content"] or "",
                "post_excerpt": r["post_excerpt"] or "",
                "post_modified": r["post_modified"],
                "meta": meta_by_post.get(r["post_id"], {}),
            })
        return result
    finally:
        conn.close()
