"""The labelled query set that retrieval is scored against.

Labels are built by **pooling**, the standard TREC construction: every
configuration under comparison contributes its top results for a query, the
union is judged, and anything unjudged counts as irrelevant.

Pooling is not a convenience -- it is what makes the comparison fair. Labelling
from one configuration's output would guarantee that configuration wins, since
every document it ranked highly would be judged and every document only its
rivals found would be an unjudged zero. The known cost is the mirror image: a
configuration added *after* judging is penalised for anything it uniquely
finds, so the pool must be rebuilt when a new contender enters.

The alternative -- generating a synthetic query from each passage and treating
that passage as its own ground truth -- is cheaper and needs no judge, but the
query inherits the passage's vocabulary. That inflates BM25 against dense
retrieval, which is precisely the comparison this harness exists to make.

Pure module: no I/O beyond reading and writing JSONL.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class LabeledQuery:
    """One query and the graded relevance of the passages judged for it."""

    query_id: str
    query: str
    category: str = ""
    # chunk_id -> relevance grade. 0 = judged irrelevant, which is information;
    # absent = never judged, which is not. Both score as zero gain, but only
    # the first says a human or judge looked.
    relevance: dict[str, float] = field(default_factory=dict)
    # Which configurations contributed candidates to the pool that was judged.
    # Recorded so a later run can tell whether its results were in scope.
    pooled_from: list[str] = field(default_factory=list)

    def n_relevant(self, min_grade: float = 2.0) -> int:
        """How many passages meet the relevance bar the metrics use."""
        return sum(1 for grade in self.relevance.values() if grade >= min_grade)

    def is_usable(self, min_grade: float = 2.0) -> bool:
        """True when this query can discriminate between retrievers.

        A query whose pool contains nothing at or above the relevance bar
        scores every configuration at zero, so it is excluded from reported
        metrics rather than counted as a universal failure. The bar must match
        the one the metrics use, or a query kept here is a guaranteed zero
        there.
        """
        return self.n_relevant(min_grade) > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "category": self.category,
            "relevance": self.relevance,
            "pooled_from": self.pooled_from,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> LabeledQuery:
        raw = record.get("relevance") or {}
        return cls(
            query_id=str(record.get("query_id", "")),
            query=str(record.get("query", "")),
            category=str(record.get("category") or ""),
            relevance={str(k): float(v) for k, v in dict(raw).items()},  # type: ignore[arg-type]
            pooled_from=[str(c) for c in (record.get("pooled_from") or [])],  # type: ignore[union-attr]
        )


def load_queries(path: str | Path) -> list[LabeledQuery]:
    """Read a JSONL query set.

    Raises:
        FileNotFoundError: if the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No query set at {path}")
    with path.open(encoding="utf-8") as handle:
        return [LabeledQuery.from_dict(json.loads(line)) for line in handle if line.strip()]


def save_queries(queries: Iterable[LabeledQuery], path: str | Path) -> int:
    """Write a JSONL query set, returning the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(query.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def usable(queries: Iterable[LabeledQuery], min_grade: float = 2.0) -> Iterator[LabeledQuery]:
    """Yield only the queries that can distinguish one retriever from another."""
    return (query for query in queries if query.is_usable(min_grade))


def pool(
    results_by_config: Mapping[str, Iterable[str]],
    depth: int,
) -> list[str]:
    """Union the top `depth` chunk ids from each configuration.

    Order is by first appearance, so the pool is deterministic and a judge
    reading it top-down sees the most widely-agreed candidates first.
    """
    seen: dict[str, None] = {}
    for chunk_ids in results_by_config.values():
        for chunk_id in list(chunk_ids)[:depth]:
            seen.setdefault(chunk_id, None)
    return list(seen)
