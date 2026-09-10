"""Turn a query set into a labelled benchmark by pooling and judging.

For each query: run every configuration under comparison, union their top
results, fetch the passage text, and have a judge grade the pool. Anything not
in the pool is unjudged and counts as irrelevant.

The pool must cover every configuration you intend to compare. A configuration
added afterwards is penalised for anything it uniquely finds, because those
passages were never judged -- so adding one means rebuilding the pool.

    python scripts/build_eval_set.py \
        --queries benchmarks/queries.jsonl \
        --out benchmarks/labeled.jsonl

Requires:  pip install -e ".[rag,eval]"  and ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from cruxbot.config import settings
from cruxbot.evaluation.dataset import load_queries, pool, save_queries
from cruxbot.evaluation.judge import AnthropicJudge, to_relevance

load_dotenv()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--queries", type=Path, default=Path("benchmarks/queries.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("benchmarks/labeled.jsonl"))
    parser.add_argument("--chroma-path", type=Path, default=settings.chroma_path)
    parser.add_argument("--collection", default=settings.collection)
    parser.add_argument("--bm25", type=Path, default=settings.bm25_cache)
    parser.add_argument("--model", default=settings.embedding_model)
    parser.add_argument("--judge-model", default="claude-opus-5")
    parser.add_argument(
        "--depth", type=int, default=10, help="Top results each config contributes to the pool"
    )
    parser.add_argument("--limit", type=int, help="Only process the first N queries")
    parser.add_argument(
        "--dry-run", action="store_true", help="Build pools and report size without judging"
    )
    return parser.parse_args(argv)


def build_retrievers(args: argparse.Namespace):
    """Load components once and return the shared configuration set.

    The pool must cover exactly the configurations that will later be scored,
    so the list comes from `evaluation.configs` rather than being written out
    here -- two hand-maintained copies drift, and a drifted pool silently
    penalises whichever configuration was left out.
    """
    from cruxbot.evaluation.configs import as_callables, build_all
    from cruxbot.retrieval.dense import ChromaStore, DenseRetriever, SentenceTransformerEmbedder
    from cruxbot.retrieval.rerank import CrossEncoderReranker
    from cruxbot.retrieval.sparse import BM25Index

    print(f"Loading {args.model} ...")
    embedder = SentenceTransformerEmbedder(args.model)
    store = ChromaStore(str(args.chroma_path), args.collection)
    print(f"  chroma: {store.count():,} vectors")

    bm25 = BM25Index.load(args.bm25)
    if bm25 is None:
        print(f"error: no usable BM25 index at {args.bm25}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  bm25:   {bm25.n_docs:,} chunks")

    print("Loading reranker ...")
    reranker = CrossEncoderReranker()

    specs = build_all(DenseRetriever(embedder, store), bm25, reranker)
    print(f"  pooling from {len(specs)} configurations: {', '.join(s.name for s in specs)}")
    return as_callables(specs), store


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.queries.exists():
        print(f"error: no query set at {args.queries}", file=sys.stderr)
        return 1

    queries = load_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]
    print(f"{len(queries)} queries from {args.queries}")

    configs, store = build_retrievers(args)
    judge = None if args.dry_run else AnthropicJudge(args.judge_model)

    started = time.perf_counter()
    pool_sizes: list[int] = []

    for i, query in enumerate(queries, 1):
        by_config = {name: fn(query.query, args.depth) for name, fn in configs.items()}
        candidates = pool(by_config, depth=args.depth)
        pool_sizes.append(len(candidates))
        query.pooled_from = sorted(configs)

        if args.dry_run:
            print(f"[{i}/{len(queries)}] {query.query_id}: {len(candidates)} candidates")
            continue

        texts = {
            chunk_id: record["text"]
            for chunk_id, record in store.get(candidates).items()
            if record.get("text")
        }
        judgements = judge.grade(query.query, texts)
        query.relevance = to_relevance(judgements)

        n_relevant = sum(1 for g in query.relevance.values() if g > 0)
        abstained = sum(1 for j in judgements if j.abstained)
        print(
            f"[{i}/{len(queries)}] {query.query_id}: {len(candidates)} pooled, "
            f"{n_relevant} relevant" + (f", {abstained} abstained" if abstained else "")
        )

    elapsed = time.perf_counter() - started
    mean_pool = sum(pool_sizes) / len(pool_sizes) if pool_sizes else 0
    print(f"\nPool: {mean_pool:.0f} candidates per query on average")

    if args.dry_run:
        print("Dry run - nothing judged, nothing written.")
        return 0

    usable = sum(1 for q in queries if q.is_usable())
    save_queries(queries, args.out)
    print(
        f"Judged in {elapsed / 60:.1f} min. "
        f"{usable}/{len(queries)} queries have at least one relevant passage."
    )
    print(f"Written to {args.out}")
    if usable < len(queries):
        print(
            f"  note: {len(queries) - usable} queries found nothing relevant and will be "
            "excluded from metrics -- check whether the corpus covers them at all."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
