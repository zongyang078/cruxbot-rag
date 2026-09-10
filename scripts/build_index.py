"""Build the dense and sparse indexes from the unified corpus.

Full and incremental runs are the same command. Chunk ids are derived from
document ids, so re-running over a subset upserts exactly those chunks:

    # everything
    python scripts/build_index.py --corpus data/unified/cruxbot_unified.jsonl

    # just today's new Reddit posts, into the same index
    python scripts/build_index.py --corpus data/incoming/reddit-2026-09-09.jsonl --skip-sparse

The sparse index is corpus-global (document frequency, average length), so it
is rebuilt rather than patched -- cheap enough at this size to run nightly.

Requires the retrieval extra:  pip install -e ".[rag]"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from cruxbot.config import settings
from cruxbot.indexing.corpus import load_documents
from cruxbot.indexing.pipeline import IndexBuilder, IndexStats, build_sparse_index


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--corpus", required=True, type=Path, help="Unified corpus (.json or .jsonl)"
    )
    parser.add_argument("--chroma-path", type=Path, default=settings.chroma_path)
    parser.add_argument("--collection", default=settings.collection)
    parser.add_argument("--model", default=settings.embedding_model)
    parser.add_argument("--bm25-out", type=Path, default=settings.bm25_cache)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--limit", type=int, help="Index only the first N documents (for a smoke run)"
    )
    parser.add_argument("--skip-dense", action="store_true", help="Only rebuild BM25")
    parser.add_argument("--skip-sparse", action="store_true", help="Only update the vector index")
    return parser.parse_args(argv)


def _documents(args: argparse.Namespace):
    stream = load_documents(args.corpus)
    if args.limit:
        for i, document in enumerate(stream):
            if i >= args.limit:
                return
            yield document
    else:
        yield from stream


def _progress(stats: IndexStats) -> None:
    print(
        f"\r  {stats.chunks:,} chunks from {stats.documents:,} docs "
        f"({stats.chunks_per_second:,.0f}/s)",
        end="",
        flush=True,
    )


def build_dense(args: argparse.Namespace) -> IndexStats:
    from cruxbot.retrieval.dense import ChromaStore, SentenceTransformerEmbedder

    print(f"Loading embedding model {args.model} ...")
    embedder = SentenceTransformerEmbedder(args.model)
    store = ChromaStore(str(args.chroma_path), args.collection, create=True)

    print(f"Indexing into {args.chroma_path} :: {args.collection}")
    stats = IndexBuilder(embedder, store, batch_size=args.batch_size, on_progress=_progress).build(
        _documents(args)
    )
    print()
    return stats


def build_sparse(args: argparse.Namespace) -> None:
    print("Building BM25 index ...")
    started = time.perf_counter()
    index = build_sparse_index(_documents(args))
    index.save(args.bm25_out)
    size_mb = Path(args.bm25_out).stat().st_size / 1024 / 1024
    print(
        f"  {index.n_docs:,} chunks, {len(index.postings):,} terms "
        f"-> {args.bm25_out} ({size_mb:.0f} MB, {time.perf_counter() - started:.0f}s)"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.corpus.exists():
        print(f"error: no corpus at {args.corpus}", file=sys.stderr)
        return 1
    if args.skip_dense and args.skip_sparse:
        print("error: --skip-dense and --skip-sparse leaves nothing to do", file=sys.stderr)
        return 1

    if not args.skip_dense:
        stats = build_dense(args)
        print(
            f"  {stats.documents:,} documents -> {stats.chunks:,} chunks "
            f"in {stats.elapsed_s / 60:.1f} min ({stats.skipped_empty:,} empty, skipped)"
        )
        for content_type, count in sorted(stats.by_content_type.items(), key=lambda kv: -kv[1]):
            print(f"    {content_type:<20} {count:>9,}")

    if not args.skip_sparse:
        build_sparse(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
