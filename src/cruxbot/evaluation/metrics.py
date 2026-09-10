"""Ranking metrics for retrieval evaluation.

These measure the retriever alone, against known-relevant documents, with no
language model involved. That separation is the point of the harness: an
end-to-end answer score cannot tell you whether a bad answer came from the
retriever missing the passage or the generator misusing it.

Relevance is graded, not binary. An LLM judge naturally produces a scale, and
collapsing it loses the distinction between a passage that answers the question
and one that is merely on-topic. Binary labels are the special case where every
gain is 1.

Which metric answers which question:

  Recall@k     Did the relevant documents reach the top k at all? This is how
               the first stage of a two-stage retriever should be judged --
               with k set to the rerank pool size, since anything outside it is
               invisible to the second stage.
  MRR          How far must a user read before the first relevant result?
  nDCG@k       How good is the ordering, given that relevance is graded and
               earlier positions matter more? This is the second-stage metric.
  Precision@k  What fraction of what was shown was worth showing?

Pure module: no I/O, no heavy dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

# Relevance grade -> gain, for judged labels. Kept explicit rather than using
# the grade as the gain directly, so the exponential convention (2^g - 1) can
# be swapped in without hunting for call sites.
Relevance = Mapping[str, float]


# Grades at or above this count as a retrieval success for the binary metrics.
# 2 = "useful but partial"; grade 1 is "on topic but does not address the
# question", which is not something the retriever should be credited for
# finding. Set it to 1 to reproduce the looser convention where any positive
# grade counts.
DEFAULT_MIN_GRADE = 2.0


def threshold(relevance: Relevance, min_grade: float = DEFAULT_MIN_GRADE) -> dict[str, float]:
    """Drop grades below `min_grade`, keeping the rest at full gain.

    Applied before the binary metrics (Recall, Precision, MRR) so that they
    measure what the caller means by relevant. nDCG is left graded: its whole
    purpose is to weight a 3 above a 2, and thresholding first would discard
    exactly that information.
    """
    return {k: v for k, v in relevance.items() if v >= min_grade}


def _gain(relevance: Relevance, doc_id: str) -> float:
    return max(relevance.get(doc_id, 0.0), 0.0)


def recall_at_k(retrieved: Sequence[str], relevance: Relevance, k: int) -> float:
    """Fraction of the total available gain that appears in the top k.

    Returns 0.0 when nothing is relevant, rather than dividing by zero. A query
    with no known relevant document carries no information about the retriever
    and should be excluded upstream, not silently scored as perfect.
    """
    total = sum(g for g in relevance.values() if g > 0)
    if total <= 0:
        return 0.0
    found = sum(_gain(relevance, doc_id) for doc_id in retrieved[:k])
    return min(found / total, 1.0)


def precision_at_k(retrieved: Sequence[str], relevance: Relevance, k: int) -> float:
    """Fraction of the top k that is relevant at all.

    Graded relevance is thresholded here: precision asks "was this worth
    showing", which is a yes-or-no question.
    """
    if k <= 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if _gain(relevance, doc_id) > 0)
    return hits / k


def reciprocal_rank(retrieved: Sequence[str], relevance: Relevance, k: int | None = None) -> float:
    """1 / (rank of the first relevant result), or 0.0 if none is found."""
    for position, doc_id in enumerate(retrieved if k is None else retrieved[:k], start=1):
        if _gain(relevance, doc_id) > 0:
            return 1.0 / position
    return 0.0


def dcg_at_k(retrieved: Sequence[str], relevance: Relevance, k: int) -> float:
    """Discounted cumulative gain: sum of gain / log2(rank + 1)."""
    return sum(
        _gain(relevance, doc_id) / math.log2(position + 1)
        for position, doc_id in enumerate(retrieved[:k], start=1)
    )


def ndcg_at_k(retrieved: Sequence[str], relevance: Relevance, k: int) -> float:
    """DCG divided by the DCG of the best possible ranking of the same labels.

    Normalizing makes the score comparable across queries with different
    numbers of relevant documents -- without it, a query with ten relevant
    passages outscores one with two regardless of ranking quality.
    """
    ideal_order = sorted((g for g in relevance.values() if g > 0), reverse=True)
    ideal = sum(
        gain / math.log2(position + 1) for position, gain in enumerate(ideal_order[:k], start=1)
    )
    if ideal <= 0:
        return 0.0
    return dcg_at_k(retrieved, relevance, k) / ideal


def evaluate_query(
    retrieved: Sequence[str],
    relevance: Relevance,
    ks: Sequence[int] = (5, 10, 50),
    min_grade: float = DEFAULT_MIN_GRADE,
) -> dict[str, float]:
    """Score one ranked result list against its labels.

    Recall, Precision and MRR use the thresholded labels; nDCG uses the graded
    ones. Returns a flat dict so results across queries can be averaged by key
    without knowing which metrics were computed.
    """
    binary = threshold(relevance, min_grade)

    scores: dict[str, float] = {"mrr": reciprocal_rank(retrieved, binary)}
    for k in ks:
        scores[f"recall@{k}"] = recall_at_k(retrieved, binary, k)
        scores[f"precision@{k}"] = precision_at_k(retrieved, binary, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevance, k)
    return scores


def aggregate(per_query: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Average each metric across queries.

    Macro-averaged: every query counts once regardless of how many relevant
    documents it has, so a handful of heavily-labelled queries cannot dominate
    the reported figure.
    """
    if not per_query:
        return {}
    keys = sorted({key for scores in per_query for key in scores})
    return {key: sum(scores.get(key, 0.0) for scores in per_query) / len(per_query) for key in keys}
