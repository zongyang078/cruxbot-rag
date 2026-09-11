"""Score every retrieval configuration against the labelled benchmark.

Each configuration differs from the one above it in exactly one variable, so a
difference between two adjacent rows is attributable to that variable and to
nothing else. This is the property the predecessor's evaluation lacked: it
compared a "v1" against a "v3" that had changed the retriever, the prompt, and
the intent router together, leaving the reported 22-point gain unattributable.

    python scripts/evaluate_retrieval.py --queries benchmarks/labeled.jsonl

Requires:  pip install -e ".[rag,eval]"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cruxbot.config import settings
from cruxbot.evaluation.dataset import load_queries
from cruxbot.evaluation.metrics import DEFAULT_MIN_GRADE
from cruxbot.evaluation.runner import RetrievalConfig, format_table, run_all, summarize


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--queries", type=Path, default=Path("benchmarks/labeled.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results.json"))
    parser.add_argument("--chroma-path", type=Path, default=settings.chroma_path)
    parser.add_argument("--collection", default=settings.collection)
    parser.add_argument("--bm25", type=Path, default=settings.bm25_cache)
    parser.add_argument("--model", default=settings.embedding_model)
    parser.add_argument("--min-grade", type=float, default=DEFAULT_MIN_GRADE)
    parser.add_argument("--rerank-pool", type=int, default=50)
    return parser.parse_args(argv)


def build_configs(args: argparse.Namespace) -> list[RetrievalConfig]:
    """The shared configuration set, ready for the runner.

    Identical to what `build_eval_set.py` pooled from -- see
    `cruxbot.evaluation.configs` for why that identity matters.
    """
    from cruxbot.evaluation.configs import build_all
    from cruxbot.retrieval.dense import ChromaStore, DenseRetriever, SentenceTransformerEmbedder
    from cruxbot.retrieval.rerank import CrossEncoderReranker
    from cruxbot.retrieval.sparse import BM25Index

    print(f"Loading {args.model} ...")
    embedder = SentenceTransformerEmbedder(args.model)
    store = ChromaStore(str(args.chroma_path), args.collection)
    bm25 = BM25Index.load(args.bm25)
    if bm25 is None:
        print(f"error: no usable BM25 index at {args.bm25}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  {store.count():,} vectors, {bm25.n_docs:,} sparse chunks")

    print("Loading reranker ...")
    reranker = CrossEncoderReranker()

    return [
        RetrievalConfig(spec.name, spec.build({}), spec.description)
        for spec in build_all(
            DenseRetriever(embedder, store), bm25, reranker, rerank_pool=args.rerank_pool
        )
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.queries.exists():
        print(f"error: no labelled query set at {args.queries}", file=sys.stderr)
        print("       run scripts/build_eval_set.py first", file=sys.stderr)
        return 1

    queries = load_queries(args.queries)
    scored = [q for q in queries if q.is_usable(args.min_grade)]
    print(
        f"{len(scored)}/{len(queries)} queries usable at grade >= {args.min_grade:g} "
        f"({sum(q.n_relevant(args.min_grade) for q in scored)} relevant passages)"
    )

    configs = build_configs(args)
    print(f"\nEvaluating {len(configs)} configurations ...")
    reports = run_all(configs, queries, ks=(5, 10, 50), min_grade=args.min_grade)

    print()
    print(format_table(reports))

    print("\nBy category (nDCG@10):")
    categories = sorted({c for r in reports for c in r.by_category})
    header = f"  {'configuration':<24}" + "".join(f"{c:>12}" for c in categories)
    print(header)
    for report in reports:
        row = f"  {report.name:<24}"
        for category in categories:
            row += f"{report.by_category.get(category, {}).get('ndcg@10', 0.0):>12.3f}"
        print(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = summarize(reports)
    payload["min_grade"] = args.min_grade
    payload["n_queries"] = len(scored)
    payload["embedding_model"] = args.model
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
