"""Hybrid retrieval: dense and sparse candidates merged by rank fusion.

Dense retrieval matches meaning: "how do I get stronger fingers" finds a
hangboard protocol that shares no words with the query. Sparse retrieval
matches surface form: a route name or a grade has to appear literally, and an
embedding of a proper noun the model never saw in training is close to noise.

Climbing queries routinely need both in one sentence -- "beginner-friendly
5.10a in Joshua Tree" is one semantic constraint and two literal ones -- so
neither retriever alone is sufficient, and RRF merges them without having to
reconcile their incomparable score scales.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from cruxbot import fusion
from cruxbot import intent as intent_module
from cruxbot.retrieval.dense import DenseRetriever
from cruxbot.retrieval.rerank import Reranker, rerank
from cruxbot.retrieval.sparse import BM25Index
from cruxbot.types import Chunk


@dataclass(slots=True)
class RetrievalResult:
    """Retrieved chunks plus the decisions that produced them.

    The trace fields exist so that evaluation can attribute a bad answer to the
    stage that caused it -- an intent misfire, an empty sparse result, a fusion
    that buried the right chunk -- rather than only observing that the answer
    was wrong.
    """

    chunks: list[Chunk] = field(default_factory=list)
    intent: str | None = None
    content_types: list[str] | None = None
    n_dense: int = 0
    n_sparse: int = 0
    n_reranked: int = 0
    rerank_ms: float = 0.0
    elapsed_ms: float = 0.0


class HybridRetriever:
    """Compose dense and sparse retrieval over a shared corpus."""

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: BM25Index | None = None,
        reranker: Reranker | None = None,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        rerank_pool: int = 50,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        # How many fused candidates the cross-encoder sees. Larger pools raise
        # recall but cost a forward pass each, so this is the main latency dial
        # of the second stage.
        self.rerank_pool = rerank_pool

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int | None = None,
        use_intent: bool = True,
        content_types: Sequence[str] | None = None,
    ) -> RetrievalResult:
        """Retrieve the top_k chunks for a query.

        Args:
            query: The user's question, unmodified.
            top_k: How many chunks to return.
            candidate_k: How many candidates to pull from each retriever before
                fusion. Defaults to 10x top_k, bounded below at 20 -- fusion
                can only promote a chunk that at least one retriever surfaced.
            use_intent: Apply the rule-based content-type prior.
            content_types: Explicit filter; overrides intent detection.
        """
        started = time.perf_counter()
        candidate_k = candidate_k or max(top_k * 10, 20)

        detected = intent_module.detect(query) if use_intent else None
        filter_types = (
            list(content_types)
            if content_types is not None
            else intent_module.content_types_for(detected)
        )

        dense_hits = self.dense.search(query, top_k=candidate_k, content_types=filter_types)
        sparse_hits = (
            self.sparse.search(query, top_k=candidate_k, content_types=filter_types)
            if self.sparse is not None
            else []
        )

        merged = fusion.rrf_merge(
            [dense_hits, sparse_hits],
            weights=[self.dense_weight, self.sparse_weight],
        )

        # A content-type filter can starve the result set -- an "article" query
        # against a corpus with few articles returns almost nothing. Backfill
        # from an unfiltered dense pass rather than answering from thin context.
        needed = self.rerank_pool if self.reranker else top_k
        if filter_types and len(merged) < needed:
            merged = self._backfill(query, merged, needed, candidate_k)

        # Text has to be present before the cross-encoder sees a candidate, so
        # hydration covers the whole rerank pool rather than just the final
        # top_k. One batched lookup either way.
        selected = self._hydrate(merged[:needed])

        rerank_ms = 0.0
        if self.reranker is not None:
            rerank_started = time.perf_counter()
            selected = rerank(self.reranker, query, selected, top_k=top_k)
            rerank_ms = (time.perf_counter() - rerank_started) * 1000
        else:
            selected = selected[:top_k]

        return RetrievalResult(
            chunks=[self._to_chunk(hit) for hit in selected],
            intent=detected,
            content_types=filter_types,
            n_dense=len(dense_hits),
            n_sparse=len(sparse_hits),
            n_reranked=len(merged[:needed]) if self.reranker else 0,
            rerank_ms=rerank_ms,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    def _backfill(
        self,
        query: str,
        merged: list[dict[str, Any]],
        top_k: int,
        candidate_k: int,
    ) -> list[dict[str, Any]]:
        seen = {hit.get("chunk_id") for hit in merged}
        for hit in self.dense.search(query, top_k=candidate_k, content_types=None):
            if len(merged) >= top_k:
                break
            if hit.get("chunk_id") not in seen:
                merged.append(hit)
                seen.add(hit.get("chunk_id"))
        return merged

    def _hydrate(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fill in document text for hits that BM25 found but dense did not.

        The sparse index stores term statistics, not document bodies, so a
        chunk ranked only by BM25 arrives here with no text. One batched
        lookup covers all of them; without it those chunks reach the prompt as
        empty context blocks and the model answers from nothing.
        """
        missing = [hit["chunk_id"] for hit in hits if not hit.get("text") and hit.get("chunk_id")]
        if not missing:
            return hits

        fetched = self.dense.store.get(missing)
        for hit in hits:
            if hit.get("text"):
                continue
            if record := fetched.get(hit.get("chunk_id", "")):
                hit["text"] = record["text"]
                hit["metadata"] = hit.get("metadata") or record["metadata"]
        return hits

    @staticmethod
    def _to_chunk(hit: dict[str, Any]) -> Chunk:
        metadata = hit.get("metadata") or {}
        return Chunk(
            text=str(hit.get("text") or ""),
            chunk_id=str(hit.get("chunk_id", "")),
            source_url=str(metadata.get("source_url", "")),
            score=float(hit.get("rerank_score", hit.get("rrf_score", hit.get("score", 0.0)))),
            metadata=metadata,
        )
