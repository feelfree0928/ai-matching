"""
One-time full ETL: extract candidates from WordPress → transform → title standardize → embeddings → index to Elasticsearch.
"""
from __future__ import annotations

import argparse
import os
import sys

# Project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from es_layer.indexer import bulk_index_candidates, bulk_index_jobs, ensure_indices, get_es_client
from etl.experience_scorer import apply_experience_scoring
from etl.extractor import extract_candidates, extract_job_postings
from etl.title_standardizer import apply_standardized_titles, load_standardized_titles
from etl.transformer import transform_candidate, transform_job
from embeddings.generator import add_embeddings_to_candidate
from openai import OpenAI
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Load candidates into Elasticsearch")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of candidates to process (for testing). Default: process all."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Shortcut: process only 100 candidates (equivalent to --limit 100)"
    )
    args = parser.parse_args()
    
    # Determine limit: --test flag takes precedence, then --limit, then env var, then None (all)
    limit = None
    if args.test:
        limit = 100
    elif args.limit is not None:
        limit = args.limit
    else:
        env_limit = os.getenv("INITIAL_LOAD_LIMIT")
        if env_limit:
            try:
                limit = int(env_limit)
            except ValueError:
                limit = None
    
    titles_path = os.path.join(os.path.dirname(__file__), "..", "standardized_titles.txt")
    standardized_titles = load_standardized_titles(titles_path)
    if not standardized_titles:
        print("Warning: no standardized titles found; title standardization will map to NONE.")

    if limit:
        print(f"TEST MODE: Processing only {limit} candidates (use without --limit for full load)")
    else:
        print("FULL LOAD MODE: Processing all candidates")
    
    print("Extracting candidates from WordPress/MariaDB...")
    raw_candidates = extract_candidates(limit=limit)
    print(f"Extracted {len(raw_candidates)} candidates.")

    if not raw_candidates:
        print("No candidates to process. Exiting.")
        return

    client = OpenAI()
    es = get_es_client()
    try:
        ensure_indices(es)
    except Exception as e:
        err_type = type(e).__name__
        print(f"Elasticsearch connection failed ({err_type}): {e}")
        print()
        print("Check:")
        print("  1. Elasticsearch is running (e.g. sudo systemctl status elasticsearch)")
        print("  2. ELASTICSEARCH_URL in .env is correct (e.g. http://localhost:9200)")
        print("  3. From this host: curl " + os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))
        print("  4. If using Elasticsearch 8.x with security, use https and set ELASTICSEARCH_USER / ELASTICSEARCH_PASSWORD")
        sys.exit(1)

    processed = []
    skipped_no_location = 0
    skipped_no_embedding = 0
    for raw in tqdm(raw_candidates, desc="Transform + standardize + score + embed"):
        try:
            c = transform_candidate(raw)
            apply_experience_scoring(c)
            apply_standardized_titles(c, standardized_titles, client=client)
            add_embeddings_to_candidate(c, client)
            if c.get("location", {}).get("lat") is None or c.get("location", {}).get("lon") is None:
                skipped_no_location += 1
                continue
            if c.get("aggregated_title_embedding") is None:
                skipped_no_embedding += 1
                continue
            processed.append(c)
        except Exception as e:
            tqdm.write(f"Skip post_id={raw.get('post_id')}: {e}")

    print(f"Processed {len(processed)} candidates with valid location and embeddings.")
    if skipped_no_location or skipped_no_embedding:
        print(f"  (Skipped: {skipped_no_location} missing location, {skipped_no_embedding} missing title embedding)")
    if not processed:
        print("Nothing to index.")
        return

    index_errors: list[tuple[str, dict]] = []
    success, failed = bulk_index_candidates(es, processed, chunk_size=50, errors=index_errors)
    print(f"Indexed: {success} ok, {failed} failed.")
    if index_errors:
        max_show = 10
        print(f"First {min(max_show, len(index_errors))} index error(s):")
        for doc_id, err in index_errors[:max_show]:
            reason = err.get("reason", err) if isinstance(err, dict) else err
            err_type = err.get("type", "") if isinstance(err, dict) else ""
            print(f"  post_id={doc_id}: [{err_type}] {reason}")

    index_jobs(es)


def index_jobs(es) -> None:
    """Extract job postings from WordPress and index them into Elasticsearch."""
    print("Extracting job postings from WordPress/MariaDB...")
    try:
        raw_jobs = extract_job_postings()
    except Exception as e:
        print(f"Job extraction failed: {e}")
        return
    print(f"Extracted {len(raw_jobs)} job postings.")
    if not raw_jobs:
        print("No job postings to index.")
        return
    transformed_jobs = []
    for rj in raw_jobs:
        try:
            j = transform_job(rj)
            transformed_jobs.append(j)
        except Exception as e:
            print(f"Skip job post_id={rj.get('post_id')}: {e}")
    if transformed_jobs:
        ok, failed = bulk_index_jobs(es, transformed_jobs)
        print(f"Jobs indexed: {ok} ok, {failed} failed.")
    else:
        print("No job postings could be transformed.")


if __name__ == "__main__":
    main()
