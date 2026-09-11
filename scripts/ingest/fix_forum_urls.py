"""
CruxBot — Fix Generic Forum URLs
==================================
Problem: Kaggle MP forum dataset (discussion_forum.csv) does not have
per-post URLs. During data cleaning, all 99k forum entries got the
generic URL "https://www.mountainproject.com/forum" which is not a
clickable citation to a specific post.

This script constructs better URLs where possible:
  - If the forum post has a topic_id or thread_id in metadata, build:
    https://www.mountainproject.com/forum/topic/<topic_id>
  - If no ID is available, mark the URL as non-specific so the frontend
    and prompt can handle it gracefully.

This is a DATA LIMITATION, not a bug — the Kaggle dataset simply doesn't
include per-post URLs. The fix is to:
  1. Acknowledge this in the writeup
  2. Mark generic URLs in the system so the LLM and UI can warn users
  3. Prioritize chunks with specific URLs during retrieval

Usage:
    python -u scripts/fix_forum_urls.py

Note: This does NOT require re-embedding. It only updates ChromaDB metadata.
"""

import chromadb

CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "cruxbot"
GENERIC_FORUM_URL = "https://www.mountainproject.com/forum"
BATCH_SIZE = 500


def main():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    total = collection.count()
    print(f"Collection '{COLLECTION_NAME}': {total:,} vectors")

    # Scan all entries for generic forum URLs
    generic_count = 0
    fixed_count = 0
    offset = 0

    while offset < total:
        batch = collection.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["metadatas"],
        )

        ids_to_update = []
        metas_to_update = []

        for chunk_id, meta in zip(batch["ids"], batch["metadatas"], strict=True):
            url = meta.get("source_url", "")
            if url == GENERIC_FORUM_URL:
                generic_count += 1

                # Try to build a better URL from metadata
                new_url = GENERIC_FORUM_URL  # default: keep generic
                topic_id = meta.get("meta_topic_id") or meta.get("meta_thread_id")
                if topic_id:
                    new_url = f"https://www.mountainproject.com/forum/topic/{topic_id}"
                    fixed_count += 1

                # Add a flag for generic URLs
                meta["url_is_generic"] = "true" if new_url == GENERIC_FORUM_URL else "false"
                if new_url != GENERIC_FORUM_URL:
                    meta["source_url"] = new_url

                ids_to_update.append(chunk_id)
                metas_to_update.append(meta)

        # Batch update
        if ids_to_update:
            collection.update(
                ids=ids_to_update,
                metadatas=metas_to_update,
            )

        offset += BATCH_SIZE
        print(
            f"  Scanned {min(offset, total):,}/{total:,} — "
            f"generic: {generic_count:,}, fixed: {fixed_count:,}",
            end="\r",
        )

    print("\n\nDone.")
    print(f"  Total generic forum URLs found: {generic_count:,}")
    print(f"  Fixed with topic_id: {fixed_count:,}")
    print(f"  Remaining generic (no ID available): {generic_count - fixed_count:,}")
    print("\n  Note: Remaining generic URLs are a data limitation of the Kaggle MP forum dataset.")
    print("  The retrieval module now flags these so the LLM and UI can warn users.")


if __name__ == "__main__":
    main()
