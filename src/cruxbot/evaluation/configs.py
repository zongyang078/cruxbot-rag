"""The canonical set of retrieval configurations under comparison.

Defined once and shared by the pooling script and the evaluation script,
because pooling is only fair if the two agree exactly. When they drift, a
configuration is scored against a pool that never contained its results: every
passage it uniquely found is unjudged, and unjudged means irrelevant. The
configuration then looks worse than it is, and nothing in the output says so.

Adding a configuration therefore means rebuilding the pool and re-judging. That
is the cost of the method, and it is why the list lives here rather than being
written out at each call site.

Each entry differs from its neighbour in exactly one variable, so a difference
between two adjacent rows is attributable to that variable alone.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_RERANK_POOL = 50


@dataclass(slots=True)
class ConfigSpec:
    """A named retrieval configuration and what it adds to the one before it."""

    name: str
    description: str
    build: Callable[[dict[str, Any]], Callable[[str, int], list[str]]]


def _ids(result: Any) -> list[str]:
    return [chunk.chunk_id for chunk in result.chunks]


def build_all(
    dense: Any,
    bm25: Any,
    reranker: Any = None,
    rerank_pool: int = DEFAULT_RERANK_POOL,
) -> list[ConfigSpec]:
    """Construct every configuration from already-loaded components.

    Components are passed in rather than built here so that one embedding model
    and one index serve every configuration -- loading them per configuration
    would make the latency column measure model loading.
    """
    from cruxbot.retrieval.hybrid import HybridRetriever

    specs = [
        ConfigSpec(
            "dense",
            "bi-encoder over ChromaDB, no filtering",
            lambda _: (
                lambda q, k: _ids(HybridRetriever(dense).retrieve(q, top_k=k, use_intent=False))
            ),
        ),
        ConfigSpec(
            "sparse",
            "BM25 over the inverted index",
            lambda _: lambda q, k: [str(h["chunk_id"]) for h in bm25.search(q, top_k=k)],
        ),
        ConfigSpec(
            "hybrid",
            "dense + sparse merged by RRF",
            lambda _: (
                lambda q, k: _ids(
                    HybridRetriever(dense, bm25).retrieve(q, top_k=k, use_intent=False)
                )
            ),
        ),
        ConfigSpec(
            "hybrid+intent",
            "adds the rule-based content-type prior",
            lambda _: lambda q, k: _ids(HybridRetriever(dense, bm25).retrieve(q, top_k=k)),
        ),
    ]

    if reranker is not None:
        specs += [
            ConfigSpec(
                "hybrid+rerank",
                f"cross-encoder over the top {rerank_pool}, no intent prior",
                lambda _: (
                    lambda q, k: _ids(
                        HybridRetriever(
                            dense, bm25, reranker=reranker, rerank_pool=rerank_pool
                        ).retrieve(q, top_k=k, use_intent=False)
                    )
                ),
            ),
            ConfigSpec(
                "hybrid+intent+rerank",
                f"both the intent prior and a cross-encoder over the top {rerank_pool}",
                lambda _: (
                    lambda q, k: _ids(
                        HybridRetriever(
                            dense, bm25, reranker=reranker, rerank_pool=rerank_pool
                        ).retrieve(q, top_k=k)
                    )
                ),
            ),
        ]
    return specs


def as_callables(specs: Sequence[ConfigSpec]) -> dict[str, Callable[[str, int], list[str]]]:
    """Reduce specs to the `name -> retrieve` mapping the pooler wants."""
    return {spec.name: spec.build({}) for spec in specs}
