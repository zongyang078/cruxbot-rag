"""Cross-encoder reranking: the second stage of retrieval.

A bi-encoder embeds the query and the document independently, so it can never
condition one on the other -- it compares two summaries written in ignorance of
each other. That independence is what makes it cheap enough to run over the
whole corpus, and also what caps its precision.

A cross-encoder reads the query and document together and scores the pair
directly. It is far too expensive to run over 384k documents, but running it
over the ~50 candidates the first stage surfaced costs one forward pass batch.

This is the trade the two-stage architecture makes: the first stage is
optimized for recall over everything, the second for precision over a handful.
It also changes how the first stage should be judged -- its job is to get the
right document into the candidate pool at all, not to rank it first, so
Recall@50 matters there and nDCG@10 matters here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class Reranker(Protocol):
    """Scores how well each document answers a query."""

    name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Return one relevance score per text. Higher is more relevant."""
        ...


def rerank(
    reranker: Reranker,
    query: str,
    hits: Sequence[dict[str, Any]],
    top_k: int | None = None,
    text_key: str = "text",
) -> list[dict[str, Any]]:
    """Reorder hits by cross-encoder score, best first.

    Hits with no text are scored last rather than dropped: a missing body is a
    hydration bug, and silently discarding the chunk would hide it while
    quietly shrinking the context the model gets.

    Returns copies annotated with "rerank_score"; the input is not mutated.
    """
    if not hits:
        return []

    texts = [str(hit.get(text_key) or "") for hit in hits]
    scorable = [i for i, text in enumerate(texts) if text]

    scores = [float("-inf")] * len(hits)
    if scorable:
        for position, score in zip(
            scorable, reranker.score(query, [texts[i] for i in scorable]), strict=True
        ):
            scores[position] = score

    ranked = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
    reordered = []
    for i in ranked:
        hit = dict(hits[i])
        hit["rerank_score"] = scores[i]
        reordered.append(hit)

    return reordered[:top_k] if top_k is not None else reordered


class CrossEncoderReranker:
    """Reranker backed by a sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str | None = None, batch_size: int = 32) -> None:
        from sentence_transformers import CrossEncoder

        from cruxbot.config import settings

        self.name = model_name or settings.reranker_model
        self.batch_size = batch_size
        self._model = CrossEncoder(self.name)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        return [float(s) for s in self._model.predict(pairs, batch_size=self.batch_size)]
