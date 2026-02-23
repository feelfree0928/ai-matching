"""
Delta sync: only re-process candidates modified since last_synced_at.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

SYNC_STATE_PATH = os.getenv("SYNC_STATE_PATH", os.path.join(os.path.dirname(__file__), "..", "sync_state.json"))


def get_last_synced() -> str | None:
    if not os.path.exists(SYNC_STATE_PATH):
        return None
    try:
        with open(SYNC_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("last_synced_at")
    except Exception:
        return None


def set_last_synced(ts: str) -> None:
    os.makedirs(os.path.dirname(SYNC_STATE_PATH) or ".", exist_ok=True)
    with open(SYNC_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump({"last_synced_at": ts}, f, indent=2)


def fetch_modified_candidates(since: str | None) -> list[dict]:
    """Fetch raw candidates with post_modified > since. If since is None, fetch all."""
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
                    SELECT p.ID AS post_id, p.post_content, p.post_excerpt, p.post_modified
                    FROM wp_posts p
                    WHERE p.post_type = 'resume' AND p.post_status = 'publish'
                    AND p.post_modified > %s
                    ORDER BY p.ID
                    """,
                    (since,),
                )
            else:
                cur.execute(
                    """
                    SELECT p.ID AS post_id, p.post_content, p.post_excerpt, p.post_modified
                    FROM wp_posts p
                    WHERE p.post_type = 'resume' AND p.post_status = 'publish'
                    ORDER BY p.ID
                    """
                )
            rows = cur.fetchall()
        if not rows:
            return []
        post_ids = [r["post_id"] for r in rows]
        from etl.extractor import RESUME_META_KEYS
        placeholders = ", ".join("%s" for _ in post_ids)
        meta_placeholders = ", ".join("%s" for _ in RESUME_META_KEYS)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT post_id, meta_key, meta_value
                FROM wp_postmeta
                WHERE post_id IN ({placeholders})
                AND meta_key IN ({meta_placeholders})
                """,
                post_ids + RESUME_META_KEYS,
            )
            meta_rows = cur.fetchall()
        meta_by_post = {r["post_id"]: {} for r in rows}
        for row in meta_rows:
            meta_by_post[row["post_id"]][row["meta_key"]] = row["meta_value"]
        result = []
        for r in rows:
            result.append({
                "post_id": r["post_id"],
                "post_content": r.get("post_content") or "",
                "post_excerpt": r.get("post_excerpt") or "",
                "post_modified": r["post_modified"],
                "meta": meta_by_post.get(r["post_id"], {}),
            })
        return result
    finally:
        conn.close()


def main() -> None:
    last = get_last_synced()
    print(f"Last synced: {last or 'never'}")

    raw_list = fetch_modified_candidates(last)
    print(f"Found {len(raw_list)} modified candidates.")

    if not raw_list:
        print("Nothing to sync.")
        return

    titles_path = os.path.join(os.path.dirname(__file__), "..", "standardized_titles.txt")
    from etl.title_standardizer import load_standardized_titles, apply_standardized_titles
    from etl.transformer import transform_candidate
    from etl.experience_scorer import apply_experience_scoring
    from embeddings.generator import add_embeddings_to_candidate
    from es_layer.indexer import bulk_index_candidates, get_es_client
    from openai import OpenAI
    from tqdm import tqdm

    standardized_titles = load_standardized_titles(titles_path)
    client = OpenAI()
    es = get_es_client()

    processed = []
    for raw in tqdm(raw_list, desc="Transform + standardize + embed"):
        try:
            c = transform_candidate(raw)
            apply_experience_scoring(c)
            apply_standardized_titles(c, standardized_titles, client=client)
            add_embeddings_to_candidate(c, client)
            if c.get("location", {}).get("lat") is not None and c.get("location", {}).get("lon") is not None:
                if c.get("aggregated_title_embedding") is not None:
                    processed.append(c)
        except Exception as e:
            tqdm.write(f"Skip post_id={raw.get('post_id')}: {e}")

    if processed:
        success, failed = bulk_index_candidates(es, processed, chunk_size=200)
        print(f"Indexed: {success} ok, {failed} failed.")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    set_last_synced(now)
    print(f"Updated last_synced_at to {now}.")


if __name__ == "__main__":
    main()
