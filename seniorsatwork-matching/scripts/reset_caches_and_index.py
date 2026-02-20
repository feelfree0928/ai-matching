"""
Reset embedding cache, title-mapping cache, and Elasticsearch candidates index
so initial_load.py can run from a completely clean state (full re-embed).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    project_root = os.path.join(os.path.dirname(__file__), "..")
    data_dir = os.path.join(project_root, "data")

    removed = []

    # 1. Embeddings cache (OpenAI text -> vector)
    embeddings_db = os.path.join(data_dir, "embeddings.db")
    if os.path.exists(embeddings_db):
        os.remove(embeddings_db)
        removed.append("data/embeddings.db")
    else:
        print("(data/embeddings.db not found, skipping)")

    # 2. Title mappings cache (raw title -> standardized title via GPT)
    title_db = os.path.join(data_dir, "title_mappings.db")
    if os.path.exists(title_db):
        os.remove(title_db)
        removed.append("data/title_mappings.db")
    else:
        print("(data/title_mappings.db not found, skipping)")

    # 3. Elasticsearch candidates and job_postings indices (use get to avoid HEAD 400 on ES 8.x)
    try:
        from elasticsearch.exceptions import NotFoundError

        from es_layer.indexer import get_es_client
        from es_layer.mappings import CANDIDATES_INDEX, JOBS_INDEX

        es = get_es_client()
        for index_name in (CANDIDATES_INDEX, JOBS_INDEX):
            try:
                es.indices.get(index=index_name)
                es.indices.delete(index=index_name)
                removed.append(f"Elasticsearch index: {index_name}")
            except NotFoundError:
                pass  # index already missing
    except Exception as e:
        print(f"Elasticsearch: could not delete indices ({e})")
        print("  You can delete the 'candidates' and 'job_postings' indices manually if needed.")

    if removed:
        print("Removed (clean slate for re-embed):")
        for path in removed:
            print(f"  - {path}")
        print()
        print("Run a full re-embed with:  python scripts/initial_load.py")
    else:
        print("Nothing to remove (caches and indices were already absent or could not be cleared).")


if __name__ == "__main__":
    main()
