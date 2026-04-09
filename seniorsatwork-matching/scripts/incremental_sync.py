"""
Delta sync: only re-process candidates modified since last_synced_at.

Smart incremental logic:
 - Recovers watermark from Elasticsearch if the on-disk state file is lost.
 - Processes modified rows in batches (cursor-based) to bound memory and allow
   crash-safe progress (watermark advances after each successfully indexed batch).
 - Detects recently unpublished/deleted posts and removes them from ES.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

from etl.extractor import POST_TYPE_RESUME, RESUME_META_KEYS

load_dotenv()

SYNC_STATE_PATH = os.getenv(
    "SYNC_STATE_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "sync_state.json"),
)
BATCH_SIZE = int(os.getenv("SYNC_BATCH_SIZE", "500"))


# ── Watermark helpers ────────────────────────────────────────────────────────

def _watermark_str_from_rows(rows: list[dict]) -> str | None:
    """Return max(post_modified) from rows as 'YYYY-MM-DD HH:MM:SS', or None."""
    best: datetime | None = None
    for r in rows:
        pm = r.get("post_modified")
        if pm is None:
            continue
        if isinstance(pm, datetime):
            cand = pm
        elif isinstance(pm, str):
            s = pm.strip().replace("Z", " ").replace("T", " ")
            cand = None
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
                try:
                    cand = datetime.strptime(s[:26], fmt)
                    break
                except ValueError:
                    continue
            if cand is None:
                continue
        else:
            continue
        if best is None or cand > best:
            best = cand
    if best is None:
        return None
    if best.tzinfo is not None:
        best = best.astimezone(timezone.utc).replace(tzinfo=None)
    return best.strftime("%Y-%m-%d %H:%M:%S")


def _read_watermark_file() -> str | None:
    if not os.path.exists(SYNC_STATE_PATH):
        return None
    try:
        with open(SYNC_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("last_synced_at")
    except Exception:
        return None


def _save_watermark(ts: str) -> None:
    os.makedirs(os.path.dirname(SYNC_STATE_PATH) or ".", exist_ok=True)
    with open(SYNC_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump({"last_synced_at": ts}, f, indent=2)


def _resolve_watermark(es) -> str:
    """Determine the sync starting point.

    Priority: on-disk file > ES max(post_modified) > epoch (full load).
    """
    from es_layer.indexer import get_max_post_modified
    from es_layer.mappings import CANDIDATES_INDEX

    wm = _read_watermark_file()
    if wm:
        print(f"Watermark from file: {wm}")
        return wm

    print("No watermark file found — recovering from Elasticsearch...")
    es_wm = get_max_post_modified(es, CANDIDATES_INDEX)
    if es_wm:
        print(f"Recovered watermark from ES: {es_wm}")
        _save_watermark(es_wm)
        return es_wm

    print("ES index empty — starting from epoch (full load, batched).")
    return "1970-01-01 00:00:00"


# ── Database helpers ─────────────────────────────────────────────────────────

def _get_connection():
    import pymysql
    return pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASS", ""),
        database=os.getenv("DB_NAME", ""),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def _meta_keys_placeholder() -> str:
    return ", ".join("%s" for _ in RESUME_META_KEYS)


def fetch_candidate_batch(conn, since: str, limit: int) -> list[dict]:
    """Fetch the next batch of published candidates with post_modified > since."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.ID AS post_id, p.post_title, p.post_content,
                   p.post_excerpt, p.post_modified, p.post_date
            FROM wp_posts p
            WHERE p.post_type = %s AND p.post_status = 'publish'
              AND p.post_modified > %s
            ORDER BY p.post_modified ASC
            LIMIT %s
            """,
            (POST_TYPE_RESUME, since, limit),
        )
        rows = cur.fetchall()

    if not rows:
        return []

    post_ids = [r["post_id"] for r in rows]
    placeholders = ", ".join("%s" for _ in post_ids)
    meta_ph = _meta_keys_placeholder()

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT post_id, meta_key, meta_value
            FROM wp_postmeta
            WHERE post_id IN ({placeholders})
              AND meta_key IN ({meta_ph})
            """,
            post_ids + RESUME_META_KEYS,
        )
        meta_rows = cur.fetchall()

    meta_by_post: dict[int, dict] = {r["post_id"]: {} for r in rows}
    for row in meta_rows:
        meta_by_post[row["post_id"]][row["meta_key"]] = row["meta_value"]

    return [
        {
            "post_id": r["post_id"],
            "post_title": r.get("post_title") or "",
            "post_content": r.get("post_content") or "",
            "post_excerpt": r.get("post_excerpt") or "",
            "post_modified": r["post_modified"],
            "post_date": r.get("post_date"),
            "meta": meta_by_post.get(r["post_id"], {}),
        }
        for r in rows
    ]


def fetch_recently_unpublished(conn, since: str) -> list[int]:
    """Find posts that were modified since `since` but are no longer published."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ID FROM wp_posts
            WHERE post_type = %s
              AND post_modified > %s
              AND post_status != 'publish'
            """,
            (POST_TYPE_RESUME, since),
        )
        return [r["ID"] for r in cur.fetchall()]


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    from etl.extractor import fetch_term_labels_standalone
    from etl.transformer import transform_candidate
    from etl.experience_scorer import apply_experience_scoring
    from embeddings.generator import add_embeddings_to_candidate
    from es_layer.indexer import (
        bulk_delete_by_ids,
        bulk_index_candidates,
        ensure_indices,
        get_es_client,
    )
    from es_layer.mappings import CANDIDATES_INDEX
    from openai import OpenAI

    es = get_es_client()
    ensure_indices(es)

    original_watermark = _resolve_watermark(es)
    watermark = original_watermark

    term_labels = fetch_term_labels_standalone()
    client = OpenAI()
    conn = _get_connection()

    total_success = 0
    total_failed = 0
    total_skipped = 0
    batch_num = 0

    try:
        while True:
            batch_num += 1
            raw_batch = fetch_candidate_batch(conn, watermark, BATCH_SIZE)
            if not raw_batch:
                break

            print(f"Batch {batch_num}: processing {len(raw_batch)} candidates (watermark > {watermark})...")

            processed = []
            for raw in raw_batch:
                try:
                    c = transform_candidate(raw, term_labels=term_labels)
                    apply_experience_scoring(c)
                    add_embeddings_to_candidate(c, client)
                    has_location = (
                        c.get("location", {}).get("lat") is not None
                        and c.get("location", {}).get("lon") is not None
                    )
                    if has_location and c.get("aggregated_title_embedding") is not None:
                        processed.append(c)
                    else:
                        total_skipped += 1
                except Exception as e:
                    print(f"  Skip post_id={raw.get('post_id')}: {e}")
                    total_skipped += 1

            if processed:
                ok, fail = bulk_index_candidates(es, processed, chunk_size=200)
                total_success += ok
                total_failed += fail
                print(f"  Indexed: {ok} ok, {fail} failed.")

            batch_wm = _watermark_str_from_rows(raw_batch)
            if batch_wm:
                watermark = batch_wm
                _save_watermark(watermark)

            if len(raw_batch) < BATCH_SIZE:
                break

        # ── Deletion detection ───────────────────────────────────────────
        print("Checking for recently unpublished/deleted candidates...")
        stale_ids = fetch_recently_unpublished(conn, original_watermark)
        if stale_ids:
            str_ids = [str(sid) for sid in stale_ids]
            deleted, absent, del_err = bulk_delete_by_ids(es, CANDIDATES_INDEX, str_ids)
            print(
                f"Stale unpublished posts: removed {deleted} from ES, "
                f"{absent} were already absent (never indexed or already deleted)"
                + (f", {del_err} delete error(s)" if del_err else "")
                + ".",
            )
        else:
            print("No stale candidates to remove.")
    finally:
        conn.close()

    print(f"\nSync complete. Indexed: {total_success}, failed: {total_failed}, skipped: {total_skipped}.")
    if watermark != original_watermark:
        print(f"Watermark advanced: {original_watermark} -> {watermark}")
    else:
        print(f"Watermark unchanged: {watermark}")


if __name__ == "__main__":
    main()
