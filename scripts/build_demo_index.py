"""Build a small, self-contained index so a stranger can run the project.

The full index is 2.5 GB of vectors plus a 277 MB sparse index — too large to
ship, and the reason the predecessor repository could not be run by anyone who
did not already have the data. This produces a subset small enough to attach to
a GitHub release.

Vectors are copied from the existing collection rather than recomputed. The
embeddings already exist and are identical either way; re-encoding 40k chunks
would take minutes and introduce a chance of the demo index disagreeing with
the one the benchmark ran against.

What goes in, in order:

1. Every chunk the benchmark judged. The demo index therefore provably contains
   the passages behind the numbers in the README, so the queries in
   `benchmarks/queries.jsonl` return something real rather than near-misses.
2. A stratified random sample over the rest, keeping each content type's share
   of the corpus. Proportional rather than balanced: a demo that over-samples
   Reddit would answer differently from the system being described.

    python scripts/build_demo_index.py --size 40000

Requires:  pip install -e ".[rag]"
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from cruxbot.config import settings

BATCH = 2000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source-path", type=Path, default=settings.chroma_path)
    parser.add_argument("--source-collection", default=settings.collection)
    parser.add_argument("--out", type=Path, default=Path("data/demo"))
    parser.add_argument("--size", type=int, default=40_000, help="Target chunk count")
    parser.add_argument("--labels", type=Path, default=Path("benchmarks/labeled.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def judged_chunk_ids(path: Path) -> set[str]:
    """Every chunk id the benchmark graded, relevant or not."""
    if not path.exists():
        return set()
    from cruxbot.evaluation.dataset import load_queries

    return {chunk_id for query in load_queries(path) for chunk_id in query.relevance}


def choose(ids: list[str], types: list[str], keep: set[str], size: int, seed: int) -> set[str]:
    """Pick `size` chunk ids: everything judged, then a proportional sample.

    Sampling is seeded so the demo index is reproducible -- a reader who builds
    it themselves should get the same one the release contains.
    """
    selected = {chunk_id for chunk_id in ids if chunk_id in keep}
    remaining = size - len(selected)
    if remaining <= 0:
        return selected

    by_type: dict[str, list[str]] = defaultdict(list)
    for chunk_id, content_type in zip(ids, types, strict=True):
        if chunk_id not in selected:
            by_type[content_type or "unknown"].append(chunk_id)

    total = sum(len(v) for v in by_type.values())
    rng = random.Random(seed)
    for _content_type, candidates in sorted(by_type.items()):
        share = round(remaining * len(candidates) / total) if total else 0
        selected.update(rng.sample(candidates, min(share, len(candidates))))
    return selected


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import chromadb

    from cruxbot.retrieval.sparse import BM25Index

    if not args.source_path.exists():
        print(f"error: no index at {args.source_path}", file=sys.stderr)
        return 1

    source = chromadb.PersistentClient(path=str(args.source_path)).get_collection(
        args.source_collection
    )
    total = source.count()
    print(f"Source: {total:,} chunks in {args.source_path}::{args.source_collection}")

    # One pass to decide what to keep, a second to copy it. Holding every
    # embedding in memory to do it in one pass would cost gigabytes.
    ids: list[str] = []
    types: list[str] = []
    for offset in range(0, total, BATCH):
        batch = source.get(limit=BATCH, offset=offset, include=["metadatas"])
        ids.extend(batch["ids"])
        types.extend((m or {}).get("content_type", "") for m in batch["metadatas"])

    keep = judged_chunk_ids(args.labels)
    print(f"  {len(keep):,} chunks judged by the benchmark; keeping all of them")

    selected = choose(ids, types, keep, args.size, args.seed)
    print(f"  selected {len(selected):,} chunks ({len(selected) / total:.1%} of the corpus)")

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    target = chromadb.PersistentClient(path=str(args.out / "chroma")).get_or_create_collection(
        settings.collection, metadata={"hnsw:space": "cosine"}
    )

    kept_ids: list[str] = []
    kept_docs: list[str] = []
    kept_meta: list[dict[str, str]] = []
    breakdown: Counter[str] = Counter()

    for offset in range(0, total, BATCH):
        batch = source.get(
            limit=BATCH, offset=offset, include=["documents", "metadatas", "embeddings"]
        )
        rows = [
            (i, d, m or {}, e)
            for i, d, m, e in zip(
                batch["ids"],
                batch["documents"],
                batch["metadatas"],
                batch["embeddings"],
                strict=True,
            )
            if i in selected
        ]
        if not rows:
            continue

        target.upsert(
            ids=[r[0] for r in rows],
            documents=[r[1] for r in rows],
            metadatas=[r[2] for r in rows],
            embeddings=[list(r[3]) for r in rows],
        )
        for chunk_id, document, meta, _ in rows:
            kept_ids.append(chunk_id)
            kept_docs.append(document)
            kept_meta.append(meta)
            breakdown[meta.get("content_type", "unknown")] += 1
        print(f"\r  copied {len(kept_ids):,}/{len(selected):,}", end="", flush=True)

    print()
    print("  rebuilding BM25 over the subset ...")
    # Rebuilt rather than filtered: document frequency and average length are
    # corpus-global, so a sliced index would carry the full corpus's statistics
    # and score differently from one actually built on these chunks.
    BM25Index.build(kept_ids, kept_docs, kept_meta).save(args.out / "bm25.pkl")

    size_mb = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"\n{len(kept_ids):,} chunks -> {args.out} ({size_mb:.0f} MB)")
    for content_type, count in breakdown.most_common():
        print(f"    {content_type:<20} {count:>8,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
