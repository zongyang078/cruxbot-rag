"""Run retrieval configurations over a labelled query set and score them.

The unit of comparison is a `RetrievalConfig`: a named setting of the retrieval
pipeline that differs from its neighbours in exactly one variable. That
constraint is the whole point. The predecessor project reported that a change
from "v1" to "v3" improved results by 22 points, but v3 changed the retriever
*and* the prompt *and* the intent router simultaneously, so the improvement
could not be attributed to any of them.

Latency is recorded per query alongside quality, because the two trade against
each other -- a reranker that adds four points of nDCG and half a second is a
different proposition from one that adds four points and forty milliseconds,
and a table showing only the first number cannot tell them apart.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from cruxbot.evaluation.dataset import LabeledQuery, usable
from cruxbot.evaluation.metrics import DEFAULT_MIN_GRADE, aggregate, evaluate_query


@dataclass(slots=True)
class RetrievalConfig:
    """A named retrieval setting to evaluate.

    `retrieve` takes a query and a result count, and returns ranked chunk ids.
    Everything else about how those ids were produced is the configuration's
    own business, which is what lets dense-only, BM25-only, hybrid, and
    reranked pipelines be compared through one interface.
    """

    name: str
    retrieve: Any  # Callable[[str, int], Sequence[str]]
    description: str = ""


@dataclass(slots=True)
class QueryResult:
    """What one configuration did on one query."""

    query_id: str
    category: str
    retrieved: list[str]
    scores: dict[str, float]
    latency_ms: float
    error: str = ""


@dataclass(slots=True)
class ConfigReport:
    """Aggregated results for one configuration."""

    name: str
    description: str = ""
    n_queries: int = 0
    n_errors: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    results: list[QueryResult] = field(default_factory=list)


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile.

    Written out rather than using `statistics.quantiles`, which interpolates
    and needs at least two data points -- neither is wanted for a latency
    figure quoted in a report.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(fraction * len(ordered)), len(ordered) - 1)
    return ordered[index]


def run_config(
    config: RetrievalConfig,
    queries: Sequence[LabeledQuery],
    ks: Sequence[int] = (5, 10, 50),
    top_k: int | None = None,
    min_grade: float = DEFAULT_MIN_GRADE,
) -> ConfigReport:
    """Evaluate one configuration over every usable query.

    `top_k` defaults to the largest cutoff being measured: retrieving fewer
    results than a metric's k would report a recall ceiling imposed by the
    harness rather than by the retriever.

    `min_grade` sets the relevance bar for both steps, and has to: it selects
    which queries can discriminate at all, and it thresholds the labels the
    binary metrics are computed from. Passing it to one and not the other keeps
    queries that are guaranteed zeros, or drops queries that are not.
    """
    scored = list(usable(queries, min_grade))
    depth = top_k or max(ks)

    results: list[QueryResult] = []
    for query in scored:
        started = time.perf_counter()
        try:
            retrieved = list(config.retrieve(query.query, depth))
            error = ""
        except Exception as exc:  # noqa: BLE001 - one bad query must not void the run
            retrieved, error = [], f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000

        results.append(
            QueryResult(
                query_id=query.query_id,
                category=query.category,
                retrieved=retrieved,
                # A failed query scores zero rather than being dropped, so a
                # configuration cannot raise its average by erroring out.
                scores=evaluate_query(retrieved, query.relevance, ks=ks, min_grade=min_grade),
                latency_ms=latency_ms,
                error=error,
            )
        )

    latencies = [r.latency_ms for r in results]
    by_category: dict[str, dict[str, float]] = {}
    for category in sorted({r.category for r in results if r.category}):
        subset = [r.scores for r in results if r.category == category]
        by_category[category] = aggregate(subset)

    return ConfigReport(
        name=config.name,
        description=config.description,
        n_queries=len(results),
        n_errors=sum(1 for r in results if r.error),
        metrics=aggregate([r.scores for r in results]),
        by_category=by_category,
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        results=results,
    )


def run_all(
    configs: Sequence[RetrievalConfig],
    queries: Sequence[LabeledQuery],
    ks: Sequence[int] = (5, 10, 50),
    min_grade: float = DEFAULT_MIN_GRADE,
) -> list[ConfigReport]:
    """Evaluate every configuration over the same query set."""
    return [run_config(config, queries, ks=ks, min_grade=min_grade) for config in configs]


def format_table(
    reports: Sequence[ConfigReport],
    columns: Sequence[str] = ("recall@50", "recall@10", "ndcg@10", "mrr"),
) -> str:
    """Render reports as a markdown table, best configuration per column bolded."""
    if not reports:
        return "(no results)"

    headers = ["configuration", *columns, "p50 ms", "p95 ms"]
    best = {
        column: max((r.metrics.get(column, 0.0) for r in reports), default=0.0)
        for column in columns
    }

    rows = []
    for report in reports:
        cells = [report.name]
        for column in columns:
            value = report.metrics.get(column, 0.0)
            rendered = f"{value:.3f}"
            cells.append(f"**{rendered}**" if value == best[column] and value > 0 else rendered)
        cells.append(f"{report.latency_p50_ms:.0f}")
        cells.append(f"{report.latency_p95_ms:.0f}")
        rows.append(cells)

    widths = [max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)) + " |",
        "| " + " | ".join("-" * w for w in widths) + " |",
    ]
    lines += [
        "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths, strict=True)) + " |"
        for row in rows
    ]
    return "\n".join(lines)


def summarize(reports: Sequence[ConfigReport]) -> dict[str, Any]:
    """Serializable summary of a full evaluation run."""
    return {
        "configurations": [
            {
                "name": r.name,
                "description": r.description,
                "n_queries": r.n_queries,
                "n_errors": r.n_errors,
                "metrics": r.metrics,
                "by_category": r.by_category,
                "latency_p50_ms": round(r.latency_p50_ms, 1),
                "latency_p95_ms": round(r.latency_p95_ms, 1),
            }
            for r in reports
        ],
        "table": format_table(reports),
    }


__all__ = [
    "ConfigReport",
    "QueryResult",
    "RetrievalConfig",
    "format_table",
    "run_all",
    "run_config",
    "summarize",
]
