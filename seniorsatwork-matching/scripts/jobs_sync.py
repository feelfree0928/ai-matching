"""
Delta sync for job postings: extract jobs modified since last sync and re-index them.
Run via API endpoint POST /api/index/jobs/sync or directly as a script.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

JOB_SYNC_STATE_PATH = os.getenv(
    "JOB_SYNC_STATE_PATH",
    os.path.join(os.path.dirname(__file__), "..", "job_sync_state.json"),
)


def get_last_synced() -> str | None:
    if not os.path.exists(JOB_SYNC_STATE_PATH):
        return None
    try:
        with open(JOB_SYNC_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("last_synced_at")
    except Exception:
        return None


def set_last_synced(ts: str) -> None:
    os.makedirs(os.path.dirname(JOB_SYNC_STATE_PATH) or ".", exist_ok=True)
    with open(JOB_SYNC_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump({"last_synced_at": ts}, f, indent=2)


def fetch_modified_jobs(since: str | None) -> list[dict]:
    """Fetch job postings with post_modified > since. If since is None, fetch all."""
    import pymysql

    conn = pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASS", ""),
        database=os.getenv("DB_NAME", ""),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        with conn.cursor() as cur:
            if since:
                cur.execute(
                    """
                    SELECT ID AS post_id, post_title, post_content, post_excerpt, post_modified
                    FROM wp_posts
                    WHERE post_type = 'noo_job' AND post_status = 'publish'
                    AND post_modified > %s
                    ORDER BY ID
                    """,
                    (since,),
                )
            else:
                cur.execute(
                    """
                    SELECT ID AS post_id, post_title, post_content, post_excerpt, post_modified
                    FROM wp_posts
                    WHERE post_type = 'noo_job' AND post_status = 'publish'
                    ORDER BY ID
                    """
                )
            rows = cur.fetchall()

        if not rows:
            return []

        post_ids = [r["post_id"] for r in rows]
        meta_by_post: dict[int, dict] = {r["post_id"]: {} for r in rows}

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


def main() -> None:
    from etl.transformer import transform_job
    from es_layer.indexer import bulk_index_jobs, get_es_client

    since = get_last_synced()
    sync_start = datetime.now(timezone.utc).isoformat()

    if since:
        print(f"Job sync: incremental since {since}")
    else:
        print("Job sync: full (no previous sync state found)")

    raw_jobs = fetch_modified_jobs(since)
    print(f"Fetched {len(raw_jobs)} job postings to process.")

    if not raw_jobs:
        print("Nothing to sync.")
        set_last_synced(sync_start)
        return

    transformed = []
    for rj in raw_jobs:
        try:
            j = transform_job(rj)
            transformed.append(j)
        except Exception as e:
            print(f"Skip job post_id={rj.get('post_id')}: {e}")

    if not transformed:
        print("No jobs could be transformed.")
        set_last_synced(sync_start)
        return

    es = get_es_client()
    ok, failed = bulk_index_jobs(es, transformed)
    print(f"Jobs indexed: {ok} ok, {failed} failed.")
    set_last_synced(sync_start)


if __name__ == "__main__":
    main()
