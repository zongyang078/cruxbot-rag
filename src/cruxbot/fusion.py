"""Reciprocal Rank Fusion for merging ranked retrieval results.

RRF (Cormack et al., SIGIR 2009) merges ranked lists by rank position rather
than by raw score. This matters here because BM25 scores are unbounded term
weights while dense scores are cosine distances -- there is no principled way
to normalize one onto the other, but their *ranks* are directly comparable.

Pure module: no I/O, no heavy dependencies.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

# The damping constant from the original paper. Large k flattens the weight
# curve, so a document ranked 1st is only modestly favoured over one ranked
# 10th -- this is what makes RRF robust to a single list being badly wrong.
DEFAULT_K = 60


def _default_key(doc: dict[str, Any]) -> str:
    """Identify a document across lists.

    Falls back to a text prefix when no chunk_id is present so that fusion
    still deduplicates rather than double-counting the same passage.
    """
    return doc.get("chunk_id") or doc.get("id") or str(doc.get("text", ""))[:120]


def rrf_merge(
    ranked_lists: Sequence[Iterable[dict[str, Any]]],
    weights: Sequence[float] | None = None,
    k: int = DEFAULT_K,
    key: Callable[[dict[str, Any]], str] = _default_key,
) -> list[dict[str, Any]]:
    """Merge N ranked lists into one, ordered by descending RRF score.

    score(d) = sum over lists L containing d of  weight[L] / (k + rank(d, L))

    Args:
        ranked_lists: Ranked result lists, best-first. May be different lengths.
        weights: Per-list weight; defaults to 1.0 each. Must match length.
        k: RRF damping constant.
        key: Extracts the identity of a document, for cross-list dedup.

    Returns:
        A single list of the input dicts, each annotated with "rrf_score".
        Ties are broken by first appearance, so the result is deterministic.

    Raises:
        ValueError: if `weights` is given but does not match `ranked_lists`.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights has length {len(weights)} but there are {len(ranked_lists)} lists"
        )

    scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}
    order: dict[str, int] = {}

    for weight, ranked in zip(weights, ranked_lists, strict=True):
        for rank, doc in enumerate(ranked):
            doc_key = key(doc)
            scores[doc_key] = scores.get(doc_key, 0.0) + weight / (k + rank)
            if doc_key not in docs:
                docs[doc_key] = doc
                order[doc_key] = len(order)

    merged = []
    for doc_key, score in sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]])):
        doc = dict(docs[doc_key])
        doc["rrf_score"] = score
        merged.append(doc)
    return merged
